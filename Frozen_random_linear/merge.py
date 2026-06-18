"""
Multimodal Time-Series + Text: Train THEN Infer (single process, no save/load)
==============================================================================
Random-frozen-encoder variant.

  - ts_encoder : frozen RANDOM nn.Linear(1, 384)   (reused in-memory; never saved/loaded)
  - model      : Qwen3-0.6B + LoRA adapters (4-bit) (trained, then for_inference)
  - projector  : trainable MLP (d_ts -> d_llm)      (trained, then .eval())

After QLoRA fine-tuning, the SAME in-memory objects are switched to inference
mode and the evaluation loop runs immediately, writing the per-task result CSVs.
No projector.pt / ts_encoder.pt is written, no merged model is saved, no reload.
"""

import os
# Set Unsloth/inductor flags BEFORE importing unsloth so its patching sees them.
os.environ["UNSLOTH_DISABLE_GEMMA_PATCH"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import unsloth  # import first so its monkey-patches apply cleanly
import logging
import json
import ast

import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import Trainer
from torch.utils.data import Dataset

# ============================================================
# 0. Logging Setup
# ============================================================
os.makedirs("logs", exist_ok=True)
log_file = "logs/timemoe_training_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting the combined random-frozen train+infer script...")

# ============================================================
# 1. Frozen Random Linear Encoder (replaces Time-MoE)
# ============================================================
logging.info("Creating frozen random linear encoder...")
D_TS = 384
ts_encoder = nn.Linear(1, D_TS).to("cuda")
for param in ts_encoder.parameters():
    param.requires_grad = False
ts_encoder.eval()
logging.info(f"Frozen linear encoder created. Hidden dim = {D_TS}")

# ============================================================
# 2. Load Qwen3-0.6B with Unsloth + QLoRA
# ============================================================
max_seq_length = 8192
dtype = None
load_in_4bit = True

logging.info("Loading Qwen3-0.6B model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
D_LLM = model.config.hidden_size  # 1024 for Qwen3-0.6B
logging.info(f"Qwen3-0.6B loaded. Hidden dim (d_llm) = {D_LLM}")

# ============================================================
# 3. Register <ts> as a Special Token
# ============================================================
logging.info("Adding <ts> as a special token...")
special_tokens_dict = {"additional_special_tokens": ["<ts>"]}
num_added = tokenizer.add_special_tokens(special_tokens_dict)
logging.info(f"Added {num_added} special token(s). Vocab size now: {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))

TS_TOKEN_ID = tokenizer.convert_tokens_to_ids("<ts>")
logging.info(f"<ts> token ID = {TS_TOKEN_ID}")

# ============================================================
# 4. Apply LoRA via Unsloth
# ============================================================
logging.info("Applying PEFT with LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

# ============================================================
# 5. Projection Layer (d_ts -> d_llm)
# ============================================================
logging.info("Creating MLP projector for encoder -> LLM embedding alignment...")


class TimeSeriesProjector(nn.Module):
    """Linear(d_ts, 4*d_ts) -> GELU -> Linear(4*d_ts, d_llm) -> LayerNorm(d_llm)."""

    def __init__(self, d_ts, d_llm):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_ts, 4 * d_ts),
            nn.GELU(),
            nn.Linear(4 * d_ts, d_llm),
            nn.LayerNorm(d_llm),
        )

    def forward(self, x):
        return self.mlp(x)


projector = TimeSeriesProjector(D_TS, D_LLM).to("cuda")

# ============================================================
# 6. Load Training Dataset
# ============================================================
logging.info("Loading training dataset...")
raw_dataset = concatenate_datasets([
    load_dataset("csv", data_files="data/train/anomaly_detection.csv", split="train"),
    load_dataset("csv", data_files="data/train/classification.csv",    split="train"),
    load_dataset("csv", data_files="data/train/multiple_choice.csv",   split="train"),
    load_dataset("csv", data_files="data/train/true_false.csv",        split="train"),
    load_dataset("csv", data_files="data/train/forecasting.csv",       split="train"),
    load_dataset("csv", data_files="data/train/imputation.csv",        split="train"),
])
raw_dataset = raw_dataset.shuffle(seed=42)
raw_dataset = raw_dataset.filter(
    lambda x: (
        x["QA_list"] is not None
        and isinstance(x["QA_list"], str)
        and len(x["QA_list"].strip()) > 0
        and x["time_series_norm"] is not None
        and x["question_text"] is not None
        and x["statistical_features"] is not None
    )
)
logging.info(f"Dataset loaded: {len(raw_dataset)} samples after filtering.")


# ============================================================
# 7. Custom Dataset with Embedding Splicing
# ============================================================
class TimeSeriesTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, ts_encoder, projector,
                 ts_token_id, max_seq_length, d_llm):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.ts_encoder = ts_encoder  # frozen random linear encoder
        self.projector = projector
        self.ts_token_id = ts_token_id
        self.max_seq_length = max_seq_length
        self.d_llm = d_llm

    def __len__(self):
        return len(self.data)

    def _parse_list(self, raw):
        if isinstance(raw, list):
            return raw
        try:
            return ast.literal_eval(raw)
        except Exception:
            return json.loads(raw)

    def _parse_stats(self, raw):
        if isinstance(raw, dict):
            return raw
        try:
            return ast.literal_eval(raw)
        except Exception:
            return json.loads(raw)

    def __getitem__(self, idx):
        row = self.data[idx]

        ts_norm = self._parse_list(row["time_series_norm"])
        stats = self._parse_stats(row["statistical_features"])
        question_text = row["question_text"]
        qa_list = row["QA_list"]

        # Encode time series with frozen random linear projection: [T] -> [T, 1] -> [T, 384]
        ts_tensor = torch.tensor(ts_norm, dtype=torch.float32).to("cuda").unsqueeze(-1)
        with torch.no_grad():
            ts_embeds = self.ts_encoder(ts_tensor)  # [T, 384]

        stats_prompt = (
            f"\nThe above is the normalized time series data. "
            f"Its raw data has the following statistical information: "
            f"minimum: {stats['min']}, maximum: {stats['max']}, "
            f"median: {stats['median']}, mean: {stats['mean']}."
        )

        assistant_marker = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"
        if assistant_marker in qa_list:
            answer_part = qa_list.split(assistant_marker, 1)[1]
            if end_marker in answer_part:
                answer_part = answer_part.split(end_marker, 1)[0]
        else:
            answer_part = "I cannot determine the answer."

        full_text = (
            f"<|im_start|>user\n"
            f"{question_text}{stats_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{answer_part.strip()}<|im_end|>"
        )

        encoding = self.tokenizer(
            full_text, return_tensors="pt", truncation=False, add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)

        ts_positions = (input_ids == self.ts_token_id).nonzero(as_tuple=True)[0]
        if len(ts_positions) == 0:
            error_msg = (
                f"ERROR: <ts> token NOT FOUND in sample {idx}!\n"
                f"  Token ID for <ts>: {self.ts_token_id}\n"
                f"  Input IDs (first 50): {input_ids[:50].tolist()}\n"
                f"  Decoded (first 200): {self.tokenizer.decode(input_ids[:50])[:200]}\n"
                f"  Check that question_text contains the literal string '<ts>'."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        ts_pos = ts_positions[0].item()

        T = ts_embeds.shape[0]
        labels_before = input_ids[:ts_pos]
        labels_ts = torch.full((T,), -100, dtype=torch.long)
        labels_after = input_ids[ts_pos + 1:]
        labels = torch.cat([labels_before, labels_ts, labels_after], dim=0)

        assistant_token_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        assistant_len = len(assistant_token_ids)
        label_list = labels.tolist()
        assistant_start = -1
        for i in range(len(label_list) - assistant_len + 1):
            if label_list[i:i + assistant_len] == assistant_token_ids:
                assistant_start = i + assistant_len
                break
        if assistant_start > 0:
            labels[:assistant_start] = -100

        total_len = labels.shape[0]
        if total_len > self.max_seq_length:
            labels = labels[:self.max_seq_length]
            total_len = self.max_seq_length

        attention_mask = torch.ones(total_len, dtype=torch.long)

        return {
            "input_ids": input_ids.cpu(),
            "ts_embeds": ts_embeds.detach().cpu(),
            "ts_pos": torch.tensor(ts_pos, dtype=torch.long),
            "labels": labels.cpu(),
            "attention_mask": attention_mask.cpu(),
        }


# ============================================================
# 8. Custom Data Collator
# ============================================================
class EmbeddingCollator:
    def __init__(self, pad_token_id, d_ts):
        self.pad_token_id = pad_token_id
        self.d_ts = d_ts

    def __call__(self, features):
        max_ids_len = max(f["input_ids"].shape[0] for f in features)
        max_ts_len = max(f["ts_embeds"].shape[0] for f in features)
        max_label_len = max(f["labels"].shape[0] for f in features)

        batch_input_ids, batch_ts_embeds, batch_ts_pos = [], [], []
        batch_ts_len, batch_labels, batch_masks = [], [], []

        for f in features:
            ids = f["input_ids"]
            ids_pad = max_ids_len - ids.shape[0]
            if ids_pad > 0:
                ids = torch.cat([ids, torch.full((ids_pad,), self.pad_token_id, dtype=torch.long)])
            batch_input_ids.append(ids)

            ts = f["ts_embeds"]
            batch_ts_len.append(ts.shape[0])
            ts_pad = max_ts_len - ts.shape[0]
            if ts_pad > 0:
                ts = torch.cat([ts, torch.zeros(ts_pad, self.d_ts)], dim=0)
            batch_ts_embeds.append(ts)

            batch_ts_pos.append(f["ts_pos"])

            labels = f["labels"]
            mask = f["attention_mask"]
            lab_pad = max_label_len - labels.shape[0]
            if lab_pad > 0:
                labels = torch.cat([labels, torch.full((lab_pad,), -100, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(lab_pad, dtype=torch.long)])
            batch_labels.append(labels)
            batch_masks.append(mask)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "ts_embeds": torch.stack(batch_ts_embeds),
            "ts_pos": torch.stack(batch_ts_pos),
            "ts_len": torch.tensor(batch_ts_len, dtype=torch.long),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_masks),
        }


# ============================================================
# 9. Custom Trainer (projector + splicing inside compute_loss)
# ============================================================
class MultimodalTrainer(Trainer):
    def __init__(self, projector, **kwargs):
        super().__init__(**kwargs)
        self.projector = projector

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device
        input_ids = inputs["input_ids"].to(device)
        ts_embeds = inputs["ts_embeds"].to(device)
        ts_pos = inputs["ts_pos"].to(device)
        ts_len = inputs["ts_len"].to(device)
        labels = inputs["labels"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        B = input_ids.shape[0]
        embed_layer = model.get_input_embeddings()

        all_combined = []
        max_combined_len = labels.shape[1]

        for i in range(B):
            text_emb = embed_layer(input_ids[i])

            T_i = ts_len[i].item()
            ts_emb_i = ts_embeds[i, :T_i, :]
            ts_proj_i = self.projector(ts_emb_i)  # gradient flows here

            pos_i = ts_pos[i].item()
            combined = torch.cat([
                text_emb[:pos_i],
                ts_proj_i,
                text_emb[pos_i + 1:],
            ], dim=0)

            if combined.shape[0] < max_combined_len:
                pad_len = max_combined_len - combined.shape[0]
                combined = torch.cat([
                    combined,
                    torch.zeros(pad_len, combined.shape[1], device=device),
                ], dim=0)
            elif combined.shape[0] > max_combined_len:
                combined = combined[:max_combined_len]

            all_combined.append(combined)

        inputs_embeds = torch.stack(all_combined, dim=0)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss


# ============================================================
# 10. Build Dataset & Training Args
# ============================================================
logging.info("Building custom dataset...")
train_dataset = TimeSeriesTextDataset(
    hf_dataset=raw_dataset,
    tokenizer=tokenizer,
    ts_encoder=ts_encoder,
    projector=projector,
    ts_token_id=TS_TOKEN_ID,
    max_seq_length=max_seq_length,
    d_llm=D_LLM,
)

logging.info("Setting up training arguments...")
training_args = UnslothTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=2000,
    warmup_steps=500,
    learning_rate=5e-5,
    embedding_learning_rate=0,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    seed=0,
    output_dir="outputs",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

# ============================================================
# 11. Initialize Trainer with Projector in the Optimizer
# ============================================================
logging.info("Initializing the multimodal trainer...")
collator = EmbeddingCollator(
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    d_ts=D_TS,
)
trainer = MultimodalTrainer(
    projector=projector,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=collator,
    args=training_args,
)

original_create_optimizer = trainer.create_optimizer


def create_optimizer_with_projector():
    original_create_optimizer()
    trainer.optimizer.add_param_group({
        "params": list(projector.parameters()),
        "lr": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
    })
    logging.info("Projector parameters added to optimizer.")


trainer.create_optimizer = create_optimizer_with_projector

# ============================================================
# 12. Log GPU Stats & Train
# ============================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
logging.info(f"{start_gpu_memory} GB of memory reserved before training.")

logging.info("Starting model training...")
try:
    trainer_stats = trainer.train()
    logging.info("Training completed successfully!")
except Exception as e:
    logging.error("An error occurred during training:", exc_info=True)
    raise

# ============================================================
# 13. Switch the SAME objects to inference mode (no save/load)
# ============================================================
logging.info("Switching to inference mode (reusing in-memory model + projector + ts_encoder)...")
del trainer
torch.cuda.empty_cache()

FastLanguageModel.for_inference(model)  # LoRA adapters stay active
model.eval()
projector.eval()
ts_encoder.eval()  # already frozen + random; reused exactly as trained

EOS_TOKEN_ID = tokenizer.eos_token_id
IM_END_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
STOP_TOKEN_IDS = {EOS_TOKEN_ID}
if IM_END_TOKEN_ID is not None and IM_END_TOKEN_ID != tokenizer.unk_token_id:
    STOP_TOKEN_IDS.add(IM_END_TOKEN_ID)
logging.info(f"Stop token IDs: {STOP_TOKEN_IDS}")


# ============================================================
# 14. Inference Helpers
# ============================================================
def parse_list(raw):
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


def parse_stats(raw):
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


def parse_qa_list(qa_string):
    if not qa_string or (isinstance(qa_string, float) and pd.isna(qa_string)):
        return None
    try:
        cleaned = qa_string.strip().replace("\n", " ").replace("\r", " ")
        if not cleaned.startswith("{"):
            cleaned = "{" + cleaned + "}"
        return json.loads(cleaned)
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def encode_time_series(ts_norm_list):
    """Encode normalized time series with the frozen random linear encoder -> [T, D_TS]."""
    ts_tensor = torch.tensor(ts_norm_list, dtype=torch.float32).to("cuda").unsqueeze(-1)  # [T, 1]
    with torch.no_grad():
        ts_embeds = ts_encoder(ts_tensor)  # [T, 384]
    return ts_embeds


@torch.no_grad()
def generate_from_embeds(model, inputs_embeds, attention_mask,
                         max_new_tokens=8192, stop_token_ids=None):
    """
    Manual autoregressive loop (Qwen3+Unsloth does not support generate(inputs_embeds=...)).
    Prefill with inputs_embeds, then decode token-by-token with the KV cache.
    Explicit position_ids carry the device so forward() doesn't deref input_ids=None.
    """
    if stop_token_ids is None:
        stop_token_ids = {EOS_TOKEN_ID}

    device = inputs_embeds.device
    seq_len = inputs_embeds.shape[1]

    generated_ids = []
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    generated_ids.append(next_token_id.item())
    if next_token_id.item() in stop_token_ids:
        return generated_ids

    current_pos = seq_len
    for _ in range(max_new_tokens - 1):
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(1, 1, dtype=torch.long, device=device),
        ], dim=1)
        step_position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=device)

        outputs = model(
            input_ids=next_token_id.unsqueeze(0),
            attention_mask=attention_mask,
            position_ids=step_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_ids.append(next_token_id.item())
        current_pos += 1
        if next_token_id.item() in stop_token_ids:
            break

    return generated_ids


@torch.no_grad()
def generate_answer_multimodal(question_text, stats, ts_norm_list):
    """Encode TS -> project -> splice into <ts> position -> manual generate."""
    ts_embeds = encode_time_series(ts_norm_list)   # [T, D_TS]
    ts_proj = projector(ts_embeds.to("cuda"))      # [T, D_LLM]

    stats_prompt = (
        f"\nThe above is the normalized time series data. "
        f"Its raw data has the following statistical information: "
        f"minimum: {stats['min']}, maximum: {stats['max']}, "
        f"median: {stats['median']}, mean: {stats['mean']}."
    )
    full_prompt = (
        f"<|im_start|>user\n"
        f"{question_text}{stats_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    encoding = tokenizer(
        full_prompt, return_tensors="pt", truncation=False, add_special_tokens=True,
    )
    input_ids = encoding["input_ids"].squeeze(0).to("cuda")

    ts_positions = (input_ids == TS_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(ts_positions) == 0:
        print("WARNING: <ts> not found in prompt. Falling back to text-only generation.")
        outputs = model.generate(
            input_ids=input_ids.unsqueeze(0),
            max_new_tokens=8192,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            do_sample=False,
        )
        generated_tokens = outputs[:, input_ids.shape[0]:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded[0].strip().replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    ts_pos = ts_positions[0].item()

    embed_layer = model.get_input_embeddings()
    text_emb = embed_layer(input_ids)  # [seq_len, D_LLM]

    combined_emb = torch.cat([
        text_emb[:ts_pos],
        ts_proj,
        text_emb[ts_pos + 1:],
    ], dim=0).unsqueeze(0)  # [1, new_seq_len, D_LLM]

    attention_mask = torch.ones(1, combined_emb.shape[1], dtype=torch.long, device="cuda")

    generated_ids = generate_from_embeds(
        model=model,
        inputs_embeds=combined_emb,
        attention_mask=attention_mask,
        max_new_tokens=8192,
        stop_token_ids=STOP_TOKEN_IDS,
    )
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


# ============================================================
# 15. Test Datasets & Execution Loop
# ============================================================
base_data_path = "/weka/s225635478/CADE/data/test"
test_files = {
    # Uncomment any tasks you want evaluated. Kept matching your inference script:
     "Classification": "classification.csv",
     "Anomaly": "anomaly_detection.csv",
     "Forecasting": "forecasting.csv",
    "Imputation": "imputation.csv",
     "Multiple_choice": "multiple_choice.csv",
     "True_false": "true_false.csv",
}

output_dir = "results/random_frozen_6tasks"
os.makedirs(output_dir, exist_ok=True)

for task_name, file_name in test_files.items():
    file_path = os.path.join(base_data_path, file_name)
    if not os.path.exists(file_path):
        logging.info(f"Skipping {task_name}: File not found ({file_path}).")
        continue

    logging.info(f">>> Running Task: {task_name}")
    df = pd.read_csv(file_path)
    predictions = []

    for index, row in df.iterrows():
        qa_list_raw = row.get("QA_list")
        application_domain = row.get("application_domain", "")
        task_type = row.get("task_type", "")
        time_series_norm_raw = row.get("time_series_norm")
        question_text_raw = row.get("question_text")
        stats_raw = row.get("statistical_features")

        has_multimodal = (
            time_series_norm_raw is not None
            and not (isinstance(time_series_norm_raw, float) and pd.isna(time_series_norm_raw))
            and question_text_raw is not None
            and not (isinstance(question_text_raw, float) and pd.isna(question_text_raw))
            and stats_raw is not None
            and not (isinstance(stats_raw, float) and pd.isna(stats_raw))
        )

        if has_multimodal:
            try:
                ts_norm_list = parse_list(time_series_norm_raw)
                stats = parse_stats(stats_raw)
                question_text = str(question_text_raw)
                if stats is None:
                    predictions.append("Error: Could not parse statistical_features")
                    continue
                ans = generate_answer_multimodal(question_text, stats, ts_norm_list)
                predictions.append(ans)
            except Exception as e:
                predictions.append(f"Error: {str(e)}")
                print(f"Row {index}: Error during multimodal generation: {e}")
                continue
        else:
            if not qa_list_raw or (isinstance(qa_list_raw, float) and pd.isna(qa_list_raw)):
                predictions.append("Error: No QA_list found")
                continue
            qa_dict = parse_qa_list(qa_list_raw)
            if qa_dict is None:
                predictions.append("Error: Could not parse QA_list")
                print(f"Row {index}: Failed to parse: {str(qa_list_raw)[:100]}...")
                continue
            question = qa_dict.get("question", "")
            if not question:
                predictions.append("Error: No question found in QA_list")
                continue

            full_context = f"Application Domain: {application_domain}\nTask Type: {task_type}\n\n{question}"
            full_prompt = f"<|im_start|>user\n{full_context}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=8192,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.1,
                do_sample=False,
            )
            generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            response = decoded[0].strip().replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            predictions.append(response)

        if index % 10 == 0:
            print(f"[{task_name}] Processed {index}/{len(df)} rows...")

    df["model_response"] = predictions
    output_filename = os.path.join(output_dir, f"results_{task_name.lower()}.csv")
    df.to_csv(output_filename, index=False)
    logging.info(f"Saved results to {output_filename}")

logging.info("--- All tasks completed successfully ---")