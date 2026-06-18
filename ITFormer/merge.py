"""
Multimodal Time-Series + Text: Train AND Infer in ONE Run (Q-Former Version)
============================================================================
This script merges the training pipeline and the inference pipeline into a
single process. After training, it does NOT save weights to disk and reload
them. Instead it keeps the trained `model` (Qwen3 + LoRA), `qformer`, and
`projector` in GPU memory and uses them directly to generate the result CSVs.

Pipeline (identical to the separate scripts):
  1. Time-MoE (50M) encodes normalized time series -> embeddings [B, T, 384]
  2. Text is tokenized; text embeddings (with <ts> tokens) come from the LLM
     embedding layer
  3. Q-Former performs cross-modal interaction (self-attn on user-turn text,
     cross-attn between learnable prefix tokens and time-series features)
     -> prefix tokens [B, prefix_num, D_LLM]
  4. A learned linear projection maps Q-Former output -> D_LLM
  5. Projected prefix embeddings replace the <ts> token(s) in the text sequence
  6. Combined embeddings -> Qwen3-0.6B (QLoRA fine-tune, then generation)

Inference note: Qwen3ForCausalLM (Unsloth-patched) does not support
model.generate(inputs_embeds=...). A manual autoregressive loop is used:
prefill with inputs_embeds, then decode token-by-token with the KV cache.
"""

# Set inference-related env vars BEFORE importing unsloth so the manual
# autoregressive decode loop behaves predictably.
import os
os.environ.setdefault("UNSLOTH_DISABLE_GEMMA_PATCH", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import unsloth
import logging
import json
import ast
import glob
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import AutoModelForCausalLM
from torch.utils.data import Dataset

# Import the Q-Former module
from qformer.qformer import ITFormerAdapted

# ============================================================
# 0. Logging Setup
# ============================================================
os.makedirs("logs", exist_ok=True)
log_file = "logs/timemoe_qformer_train_and_infer_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting combined train + inference script (Q-Former version)...")

# ============================================================
# 1. Load and Prepare the Time-MoE Encoder (FROZEN)
# ============================================================
logging.info("Loading Time-MoE encoder (frozen)...")

# ---- Compatibility patch (needed by both training and inference) ----
from transformers import DynamicCache
if not hasattr(DynamicCache, 'get_usable_length'):
    def _get_usable_length(self, new_seq_length=0, layer_idx=0):
        """Compatibility shim: old get_usable_length(new_seq_len, layer_idx) -> new get_seq_length(layer_idx)."""
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length
    logging.info("Patched DynamicCache: added get_usable_length shim.")

timemoe_full = AutoModelForCausalLM.from_pretrained(
    "Maple728/TimeMoE-50M",
    device_map="cuda",
    trust_remote_code=True,
)

for param in timemoe_full.parameters():
    param.requires_grad = False
timemoe_full.eval()

timemoe_encoder = timemoe_full
D_TS = 384  # TimeMoE-50M hidden dim

logging.info(f"Time-MoE backbone loaded. Hidden dim = {D_TS}")

# ============================================================
# 2. Load Qwen3-0.6B with Unsloth + QLoRA
# ============================================================
# Load with the larger context so the SAME in-memory model can be used both
# for training (dataset truncated at TRAIN_MAX_SEQ_LENGTH) and for generation.
max_seq_length = 8192
TRAIN_MAX_SEQ_LENGTH = 8192  # dataset truncation length used during training
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
logging.info(f"Qwen3-0.6B loaded. Hidden dim (D_LLM) = {D_LLM}")

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

# Stop tokens for generation (used later at inference time)
EOS_TOKEN_ID = tokenizer.eos_token_id
IM_END_TOKEN_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
STOP_TOKEN_IDS = {EOS_TOKEN_ID}
if IM_END_TOKEN_ID is not None and IM_END_TOKEN_ID != tokenizer.unk_token_id:
    STOP_TOKEN_IDS.add(IM_END_TOKEN_ID)
logging.info(f"Stop token IDs: {STOP_TOKEN_IDS}")

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
# 5. Define the Q-Former + Linear Projection
# ============================================================
logging.info("Creating Q-Former and projection layer...")

PREFIX_NUM = 20  # number of learnable prefix tokens output by Q-Former


class QFormerArgs:
    """Configuration for ITFormerAdapted."""
    def __init__(self):
        self.it_d_model = D_LLM   # query/instruction dimension matches LLM hidden size
        self.it_d_ts = D_TS       # time-series memory dimension from TimeMoE
        self.it_qk_dim = D_LLM    # inner attention dimension aligned with LLM space
        self.it_n_heads = 8
        self.it_layers = 4
        self.it_dropout = 0.1
        self.prefix_num = PREFIX_NUM


qformer_args = QFormerArgs()
qformer = ITFormerAdapted(qformer_args).to("cuda")

qformer_params = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
logging.info(f"Q-Former created. Trainable params: {qformer_params:,}")
logging.info(f"  it_d_model (D_LLM) = {qformer_args.it_d_model}")
logging.info(f"  it_d_ts    (D_TS)  = {qformer_args.it_d_ts}")
logging.info(f"  it_qk_dim          = {qformer_args.it_qk_dim}")
logging.info(f"  prefix_num         = {PREFIX_NUM}")


class TimeSeriesProjector(nn.Module):
    """Linear projection from Q-Former output space to LLM embedding space."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.proj(x)


projector = TimeSeriesProjector(D_LLM, D_LLM).to("cuda")
proj_params = sum(p.numel() for p in projector.parameters())
logging.info(f"Projector: {D_LLM} -> {D_LLM} (trainable params: {proj_params:,})")

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
# 7. Custom Dataset
# ============================================================
class TimeSeriesTextDataset(Dataset):
    """
    For each sample:
      1. Encode normalized time series with Time-MoE -> [T, D_TS]
      2. Build prompt text (with <ts> placeholder), tokenize
      3. Return raw data; embedding splicing + Q-Former happen in compute_loss
    """

    def __init__(self, hf_dataset, tokenizer, timemoe_encoder,
                 ts_token_id, max_seq_length, prefix_num):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.timemoe = timemoe_encoder
        self.ts_token_id = ts_token_id
        self.max_seq_length = max_seq_length
        self.prefix_num = prefix_num

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

        # --- A. Parse columns ---
        ts_norm = self._parse_list(row["time_series_norm"])
        stats = self._parse_stats(row["statistical_features"])
        question_text = row["question_text"]  # contains <ts> placeholder
        qa_list = row["QA_list"]

        # --- B. Encode time series with Time-MoE (FROZEN) ---
        ts_tensor = torch.tensor([ts_norm], dtype=torch.float32).to("cuda")
        with torch.no_grad():
            ts_outputs = self.timemoe(
                input_ids=ts_tensor,
                output_hidden_states=True,
                use_cache=False,
            )
        if hasattr(ts_outputs, 'hidden_states') and ts_outputs.hidden_states is not None:
            ts_embeds = ts_outputs.hidden_states[-1]  # [1, T, 384]
        else:
            ts_embeds = ts_outputs.last_hidden_state
        ts_embeds = ts_embeds.squeeze(0)  # [T, D_TS]

        # --- C. Build the full prompt text ---
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

        # --- D. Tokenize ---
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)  # [seq_len]

        # --- E. Find <ts> position ---
        ts_positions = (input_ids == self.ts_token_id).nonzero(as_tuple=True)[0]
        if len(ts_positions) == 0:
            error_msg = (
                f"ERROR: <ts> token NOT FOUND in sample {idx}!\n"
                f"  Token ID for <ts>: {self.ts_token_id}\n"
                f"  Input IDs (first 50): {input_ids[:50].tolist()}\n"
                f"  Decoded text (first 200 chars): "
                f"{self.tokenizer.decode(input_ids[:50])[:200]}\n"
                f"  Check that question_text contains the literal string '<ts>'."
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        ts_pos = ts_positions[0].item()

        # --- E2. Find end of user turn ---
        im_end_token_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        im_end_id = im_end_token_ids[0]  # single token for <|im_end|>
        im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        user_end_pos = None
        for pos in im_end_positions:
            if pos.item() > ts_pos:
                user_end_pos = pos.item() + 1  # +1 to include the <|im_end|> token itself
                break
        if user_end_pos is None:
            user_end_pos = input_ids.shape[0]
            logging.warning(f"Sample {idx}: Could not find <|im_end|> after <ts>, "
                            f"using full sequence for Q-Former input.")

        # --- F. Build labels ---
        T_prefix = self.prefix_num
        labels_before = input_ids[:ts_pos]
        labels_qf = torch.full((T_prefix,), -100, dtype=torch.long)  # ignore Q-Former positions
        labels_after = input_ids[ts_pos + 1:]
        labels = torch.cat([labels_before, labels_qf, labels_after], dim=0)

        # Mask user portion (only train on assistant's response)
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

        # --- G. Truncate if needed ---
        if total_len > self.max_seq_length:
            labels = labels[:self.max_seq_length]
            total_len = self.max_seq_length

        attention_mask = torch.ones(total_len, dtype=torch.long)

        return {
            "input_ids": input_ids.cpu(),                  # [seq_len] (original, with <ts>)
            "ts_embeds": ts_embeds.detach().cpu(),         # [T, D_TS=384]
            "ts_pos": torch.tensor(ts_pos, dtype=torch.long),
            "user_end_pos": torch.tensor(user_end_pos, dtype=torch.long),
            "labels": labels.cpu(),                        # [total_len] (spliced with prefix_num)
            "attention_mask": attention_mask.cpu(),
        }


# ============================================================
# 8. Custom Data Collator
# ============================================================
class EmbeddingCollator:
    """Pads all fields to the max length in the batch."""

    def __init__(self, pad_token_id, d_ts):
        self.pad_token_id = pad_token_id
        self.d_ts = d_ts

    def __call__(self, features):
        max_ids_len = max(f["input_ids"].shape[0] for f in features)
        max_ts_len = max(f["ts_embeds"].shape[0] for f in features)
        max_label_len = max(f["labels"].shape[0] for f in features)

        batch_input_ids = []
        batch_ts_embeds = []
        batch_ts_pos = []
        batch_user_end_pos = []
        batch_ts_len = []
        batch_labels = []
        batch_masks = []

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
            batch_user_end_pos.append(f["user_end_pos"])

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
            "user_end_pos": torch.stack(batch_user_end_pos),
            "ts_len": torch.tensor(batch_ts_len, dtype=torch.long),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_masks),
        }


# ============================================================
# 9. Custom Trainer with Q-Former Integration
# ============================================================
from transformers import Trainer


class MultimodalQFormerTrainer(Trainer):
    """
    Custom trainer that integrates Q-Former into the forward pass.
    See header docstring for the per-sample flow.
    """

    def __init__(self, qformer, projector, prefix_num, **kwargs):
        super().__init__(**kwargs)
        self.qformer = qformer
        self.projector = projector
        self.prefix_num = prefix_num

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device
        input_ids = inputs["input_ids"].to(device)           # [B, seq_len]
        ts_embeds = inputs["ts_embeds"].to(device)           # [B, max_T, D_TS]
        ts_pos = inputs["ts_pos"].to(device)                 # [B]
        user_end_pos = inputs["user_end_pos"].to(device)     # [B]
        ts_len = inputs["ts_len"].to(device)                 # [B]
        labels = inputs["labels"].to(device)                 # [B, label_len]
        attention_mask = inputs["attention_mask"].to(device)  # [B, label_len]

        B = input_ids.shape[0]

        embed_layer = model.get_input_embeddings()

        all_combined = []
        max_combined_len = labels.shape[1]

        for i in range(B):
            # ---- Step 1: text embeddings from LLM embedding layer ----
            text_emb = embed_layer(input_ids[i])  # [seq_len, D_LLM]

            # ---- Step 2: time-series memory (trimmed to actual length) ----
            T_i = ts_len[i].item()
            ts_mem_i = ts_embeds[i, :T_i, :]  # [T_i, D_TS]

            # ---- Step 3: Q-Former cross-modal interaction (user turn only) ----
            uep_i = user_end_pos[i].item()
            user_emb = text_emb[:uep_i, :]  # [user_turn_len, D_LLM]

            qf_out = self.qformer(
                x=user_emb.unsqueeze(0),       # [1, user_turn_len, D_LLM]
                memory=ts_mem_i.unsqueeze(0),  # [1, T_i, D_TS]
            )  # [1, prefix_num, D_LLM]
            qf_out = qf_out.squeeze(0)  # [prefix_num, D_LLM]

            # ---- Step 4: Linear projection ----
            qf_proj = self.projector(qf_out)  # [prefix_num, D_LLM]

            # ---- Step 5: Splice — replace <ts> token with prefix tokens ----
            pos_i = ts_pos[i].item()
            combined = torch.cat([
                text_emb[:pos_i],        # text before <ts>
                qf_proj,                 # Q-Former prefix tokens (replaces <ts>)
                text_emb[pos_i + 1:],    # text after <ts>
            ], dim=0)

            # ---- Step 6: Pad or truncate to max_combined_len ----
            if combined.shape[0] < max_combined_len:
                pad_len = max_combined_len - combined.shape[0]
                combined = torch.cat([
                    combined,
                    torch.zeros(pad_len, combined.shape[1], device=device),
                ], dim=0)
            elif combined.shape[0] > max_combined_len:
                combined = combined[:max_combined_len]

            all_combined.append(combined)

        inputs_embeds = torch.stack(all_combined, dim=0)  # [B, max_combined_len, D_LLM]

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
    timemoe_encoder=timemoe_encoder,
    ts_token_id=TS_TOKEN_ID,
    max_seq_length=TRAIN_MAX_SEQ_LENGTH,
    prefix_num=PREFIX_NUM,
)

logging.info("Setting up training arguments...")
training_args = UnslothTrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=2000,
    warmup_steps=500,
    learning_rate=5e-5,
    embedding_learning_rate=1e-5,
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
# 11. Initialize Trainer
# ============================================================
logging.info("Initializing the multimodal Q-Former trainer...")

collator = EmbeddingCollator(
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    d_ts=D_TS,
)

trainer = MultimodalQFormerTrainer(
    qformer=qformer,
    projector=projector,
    prefix_num=PREFIX_NUM,
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=collator,
    args=training_args,
)

# Add Q-Former + projector parameters to the optimizer
original_create_optimizer = trainer.create_optimizer


def create_optimizer_with_extra_params():
    """Inject Q-Former and projector parameters into the trainer's optimizer."""
    original_create_optimizer()
    trainer.optimizer.add_param_group({
        "params": list(qformer.parameters()),
        "lr": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
    })
    trainer.optimizer.add_param_group({
        "params": list(projector.parameters()),
        "lr": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
    })
    logging.info("Q-Former and projector parameters added to optimizer.")


trainer.create_optimizer = create_optimizer_with_extra_params

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
# 13. Switch to Inference (NO save / NO reload)
# ============================================================
# We reuse the in-memory trained objects directly:
#   - `model`     : Qwen3-0.6B + trained LoRA adapters (and resized embeddings)
#   - `qformer`   : trained Q-Former
#   - `projector` : trained linear projector
# This skips the merge/save and the from_pretrained reload entirely.
logging.info("Switching trained model to inference mode (no checkpoint reload)...")
FastLanguageModel.for_inference(model)   # enables Unsloth fast inference path
model.eval()
qformer.eval()
projector.eval()
logging.info("Model, Q-Former, and projector set to eval mode.")

# ============================================================
# 14. Inference Helper Functions
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
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


def parse_qa_list(qa_string):
    if not qa_string or pd.isna(qa_string):
        return None
    try:
        cleaned = qa_string.strip()
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        if not cleaned.startswith('{'):
            cleaned = '{' + cleaned + '}'
        return json.loads(cleaned)
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def encode_time_series(ts_norm_list):
    """Encode normalized time series with frozen Time-MoE -> [T, D_TS]."""
    ts_tensor = torch.tensor([ts_norm_list], dtype=torch.float32).to("cuda")
    with torch.no_grad():
        ts_outputs = timemoe_encoder(
            input_ids=ts_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
    if hasattr(ts_outputs, 'hidden_states') and ts_outputs.hidden_states is not None:
        ts_embeds = ts_outputs.hidden_states[-1]
    else:
        ts_embeds = ts_outputs.last_hidden_state
    return ts_embeds.squeeze(0)  # [T, 384]


# ============================================================
# 15. Manual Autoregressive Generation with inputs_embeds
#     (Qwen3ForCausalLM / Unsloth does not support
#      model.generate(inputs_embeds=...))
# ============================================================
@torch.no_grad()
def generate_from_embeds(model, inputs_embeds, attention_mask,
                         max_new_tokens=2048, stop_token_ids=None):
    """
    Step 1 (prefill):  forward(inputs_embeds=combined_emb) -> logits + KV cache
    Step 2+ (decode):  forward(input_ids=last_token, past_key_values=cache)
    Stops at a stop token or max_new_tokens.
    """
    if stop_token_ids is None:
        stop_token_ids = {EOS_TOKEN_ID}

    device = inputs_embeds.device
    seq_len = inputs_embeds.shape[1]

    generated_ids = []

    # --- Prefill ---
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated_ids.append(next_token_id.item())

    if next_token_id.item() in stop_token_ids:
        return generated_ids

    current_pos = seq_len

    # --- Decode ---
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
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_token_id.item())
        current_pos += 1

        if next_token_id.item() in stop_token_ids:
            break

    return generated_ids


def generate_answer_multimodal(question_text, stats, ts_norm_list):
    """Full multimodal Q-Former generation (see header for the flow)."""
    # --- A. Encode time series with TimeMoE ---
    ts_embeds = encode_time_series(ts_norm_list)       # [T, D_TS]

    # --- B. Build prompt (same format as training, no answer) ---
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

    # --- C. Tokenize ---
    encoding = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"].squeeze(0).to("cuda")  # [seq_len]

    # --- D. Find <ts> position ---
    ts_positions = (input_ids == TS_TOKEN_ID).nonzero(as_tuple=True)[0]
    if len(ts_positions) == 0:
        print("WARNING: <ts> token not found in prompt. Falling back to text-only generation.")
        outputs = model.generate(
            input_ids=input_ids.unsqueeze(0),
            max_new_tokens=8192,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            do_sample=False,
        )
        input_length = input_ids.shape[0]
        generated_tokens = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        response = decoded[0].strip()
        return response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    ts_pos = ts_positions[0].item()

    # --- E. Text embeddings ---
    embed_layer = model.get_input_embeddings()
    text_emb = embed_layer(input_ids)  # [seq_len, D_LLM]

    # --- F. User turn boundary ---
    im_end_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_token_ids[0]
    im_end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]

    user_end_pos = None
    for pos in im_end_positions:
        if pos.item() > ts_pos:
            user_end_pos = pos.item() + 1
            break
    if user_end_pos is None:
        user_end_pos = input_ids.shape[0]
        print("WARNING: Could not find <|im_end|> after <ts>, using full sequence for Q-Former input.")

    # --- G. Q-Former cross-modal interaction ---
    user_emb = text_emb[:user_end_pos, :]  # [user_turn_len, D_LLM]
    with torch.no_grad():
        qf_out = qformer(
            x=user_emb.unsqueeze(0),                       # [1, user_turn_len, D_LLM]
            memory=ts_embeds.unsqueeze(0).to("cuda"),      # [1, T, D_TS]
        )  # [1, prefix_num, D_LLM]
    qf_out = qf_out.squeeze(0)  # [prefix_num, D_LLM]

    # --- H. Linear projection ---
    with torch.no_grad():
        qf_proj = projector(qf_out)  # [prefix_num, D_LLM]

    # --- I. Splice prefix tokens in place of <ts> ---
    combined_emb = torch.cat([
        text_emb[:ts_pos],        # text before <ts>
        qf_proj,                  # Q-Former prefix tokens
        text_emb[ts_pos + 1:],    # text after <ts>
    ], dim=0).unsqueeze(0)        # [1, new_seq_len, D_LLM]

    combined_len = combined_emb.shape[1]
    attention_mask = torch.ones(1, combined_len, dtype=torch.long, device="cuda")

    # --- J. Generate ---
    generated_ids = generate_from_embeds(
        model=model,
        inputs_embeds=combined_emb,
        attention_mask=attention_mask,
        max_new_tokens=8192,
        stop_token_ids=STOP_TOKEN_IDS,
    )

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return response


# ============================================================
# 16. Define Test Datasets
# ============================================================
base_data_path = "/weka/s225635478/CADE/data/test"
test_files = {
    "Classification": "classification.csv",
    "Anomaly": "anomaly_detection.csv",
    "Forecasting": "forecasting.csv",
    "Multiple_choice": "multiple_choice.csv",
    "True_false": "true_false.csv",
    "Imputation": "imputation.csv",
}

# ============================================================
# 17. Execution Loop -> write result CSVs
# ============================================================
logging.info("Starting inference over test datasets...")
for task_name, file_name in test_files.items():
    file_path = os.path.join(base_data_path, file_name)

    if not os.path.exists(file_path):
        print(f"Skipping {task_name}: File not found.")
        continue

    print(f"\n>>> Running Task: {task_name}")
    df = pd.read_csv(file_path)

    predictions = []

    for index, row in df.iterrows():
        qa_list_raw = row.get('QA_list')
        application_domain = row.get('application_domain', '')
        task_type = row.get('task_type', '')
        time_series_norm_raw = row.get('time_series_norm')
        question_text_raw = row.get('question_text')
        stats_raw = row.get('statistical_features')

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
            if not qa_list_raw or pd.isna(qa_list_raw):
                predictions.append("Error: No QA_list found")
                continue

            qa_dict = parse_qa_list(qa_list_raw)
            if qa_dict is None:
                predictions.append("Error: Could not parse QA_list")
                print(f"Row {index}: Failed to parse: {str(qa_list_raw)[:100]}...")
                continue

            question = qa_dict.get('question', '')
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
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            response = decoded[0].strip()
            response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            predictions.append(response)

        if index % 10 == 0:
            print(f"Processed {index}/{len(df)} rows...")

    df['model_response'] = predictions
    output_dir = "results/qformer"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"results_{task_name.lower()}.csv")
    df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")
    logging.info(f"Saved results to {output_filename}")

print("\n--- Train + inference completed successfully ---")
logging.info("Train + inference completed successfully.")