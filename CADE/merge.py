"""
Combined Train + In-Memory Inference for the AUX_LOSS_WEIGHT ablation
=====================================================================
One run = one (QUEUE_SIZE, AUX_LOSS_WEIGHT) pair. The script:
  1. Loads a FRESH Qwen3-0.6B + QLoRA, trainable ts_encoder, and projector
  2. Trains with the SupCon auxiliary loss (weight set via --aux_loss_weight)
  3. Runs inference IN-MEMORY on the just-trained model (no save/reload from
     disk, no vocab-size config patching) and writes per-task result CSVs

For this ablation QUEUE_SIZE is held fixed (512, the best value from the
earlier queue sweep) and AUX_LOSS_WEIGHT is swept. weight=0.0 (no aux loss)
and weight=0.1 (the incumbent) are already done; this runs the rest.

Run ONE config per process (see submit_all.sh). Do NOT loop configs inside a
single process: re-initializing Unsloth models in one process leaks GPU
memory and trips global-patch state. A fresh process per config is clean and
trivially parallelizable across GPUs.

Usage:
    python run_queue_sweep.py --queue_size 512 --aux_loss_weight 0.25 \
        --output_root results/qsweep \
        --test_data_dir /weka/s225635478/CADE/data/test
"""
import unsloth  # must be imported before transformers/torch

import argparse
import os
import ast
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainingArguments
from transformers import Trainer


# ============================================================
# Fixed hyperparameters (held CONSTANT across the whole ablation)
# ============================================================
# NOTE: AUX_LOSS_WEIGHT is no longer a global; it is a CLI arg threaded
# through train() and into the trainer as self.aux_loss_weight.
AUX_LOSS_TEMPERATURE = 0.07
QUEUE_WARMUP_MIN = 16            # min entries before aux loss is computed
ANCHORED_TASKS = {"classification"}

CLASSIFICATION_LABELS = [
    "Downstairs", "Freeze", "Jogging", "No freeze",
    "Sitting", "Standing", "Upstairs", "Walking",
]

D_TS = 384
MAX_SEQ_LENGTH = 8192


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--queue_size", type=int, default=512,
                   help="Memory-bank capacity. Fixed at 512 for the weight ablation.")
    p.add_argument("--aux_loss_weight", type=float, default=0.1,
                   help="SupCon auxiliary-loss weight to test (e.g. 0.01/0.05/0.25/0.5). "
                        "Set 0.0 to disable the aux term entirely.")
    p.add_argument("--output_root", type=str, default="results/qsweep",
                   help="Per-config results go to {output_root}/qsize_{q}_w{weight}/.")
    p.add_argument("--train_data_dir", type=str, default="data/train")
    p.add_argument("--test_data_dir", type=str,
                   default="/home/s225635478/scratch/CADE/data/test")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--save_model", action="store_true",
                   help="Also save merged LLM + projector + ts_encoder (off by "
                        "default to save disk; results CSVs are the deliverable).")
    return p.parse_args()


def weight_tag(aux_loss_weight):
    """Filesystem-safe tag for a weight value: 0.01->'0.01', 0.5->'0.5', 0.0->'0'."""
    return f"{aux_loss_weight:g}"


# ============================================================
# Projector (identical to original)
# ============================================================
class TimeSeriesProjector(nn.Module):
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


# ============================================================
# Dataset (identical logic; raw TS + class_key)
# ============================================================
class TimeSeriesTextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, ts_token_id, max_seq_length, d_llm):
        self.data = hf_dataset
        self.tokenizer = tokenizer
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

    def _extract_class_key(self, task_type, application_domain, answer_part):
        if task_type == "classification":
            for cls in sorted(CLASSIFICATION_LABELS, key=len, reverse=True):
                if cls in answer_part:
                    return ("classification", cls)
            return None
        return None

    def __getitem__(self, idx):
        row = self.data[idx]

        ts_norm = self._parse_list(row["time_series_norm"])
        stats = self._parse_stats(row["statistical_features"])
        question_text = row["question_text"]
        qa_list = row["QA_list"]
        task_type = row["task_type"]
        application_domain = row["application_domain"]

        ts_raw = torch.tensor(ts_norm, dtype=torch.float32)
        T = ts_raw.shape[0]

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

        class_key = None
        if task_type in ANCHORED_TASKS:
            class_key = self._extract_class_key(task_type, application_domain, answer_part)

        encoding = self.tokenizer(
            full_text, return_tensors="pt", truncation=False, add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)

        ts_positions = (input_ids == self.ts_token_id).nonzero(as_tuple=True)[0]
        if len(ts_positions) == 0:
            raise ValueError(f"<ts> token not found in sample {idx}.")
        ts_pos = ts_positions[0].item()

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
            "ts_raw": ts_raw.cpu(),
            "ts_pos": torch.tensor(ts_pos, dtype=torch.long),
            "labels": labels.cpu(),
            "attention_mask": attention_mask.cpu(),
            "task_type": task_type,
            "class_key": class_key,
        }


# ============================================================
# Collator (identical)
# ============================================================
class EmbeddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_ids_len = max(f["input_ids"].shape[0] for f in features)
        max_ts_len = max(f["ts_raw"].shape[0] for f in features)
        max_label_len = max(f["labels"].shape[0] for f in features)

        batch_input_ids, batch_ts_raw, batch_ts_pos = [], [], []
        batch_ts_len, batch_labels, batch_masks = [], [], []

        for f in features:
            ids = f["input_ids"]
            ids_pad = max_ids_len - ids.shape[0]
            if ids_pad > 0:
                ids = torch.cat([ids, torch.full((ids_pad,), self.pad_token_id, dtype=torch.long)])
            batch_input_ids.append(ids)

            ts = f["ts_raw"]
            batch_ts_len.append(ts.shape[0])
            ts_pad = max_ts_len - ts.shape[0]
            if ts_pad > 0:
                ts = torch.cat([ts, torch.zeros(ts_pad)])
            batch_ts_raw.append(ts)

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
            "ts_raw": torch.stack(batch_ts_raw),
            "ts_pos": torch.stack(batch_ts_pos),
            "ts_len": torch.tensor(batch_ts_len, dtype=torch.long),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_masks),
            "task_type": [f["task_type"] for f in features],
            "class_key": [f["class_key"] for f in features],
        }


# ============================================================
# Trainer (QUEUE_SIZE and AUX_LOSS_WEIGHT are instance attributes)
# ============================================================
class MultimodalTrainer(Trainer):
    def __init__(self, projector, ts_encoder, label_targets, class_to_id,
                 id_to_target, queue_size, aux_loss_weight, **kwargs):
        super().__init__(**kwargs)
        self.projector = projector
        self.ts_encoder = ts_encoder
        self.label_targets = label_targets
        self.class_to_id = class_to_id
        self.id_to_target = id_to_target

        self.queue_size = queue_size          # fixed (512) for this ablation
        self.aux_loss_weight = aux_loss_weight  # <-- swept parameter
        self.queue_projections = []
        self.queue_class_ids = []
        self.queue_ptr = 0

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def _enqueue(self, projection_vec, class_id):
        if len(self.queue_projections) < self.queue_size:
            self.queue_projections.append(projection_vec)
            self.queue_class_ids.append(class_id)
        else:
            self.queue_projections[self.queue_ptr] = projection_vec
            self.queue_class_ids[self.queue_ptr] = class_id
            self.queue_ptr = (self.queue_ptr + 1) % self.queue_size

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = model.device
        input_ids = inputs["input_ids"].to(device)
        ts_raw = inputs["ts_raw"].to(device)
        ts_pos = inputs["ts_pos"].to(device)
        ts_len = inputs["ts_len"].to(device)
        labels = inputs["labels"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        class_keys = inputs["class_key"]

        B = input_ids.shape[0]
        embed_layer = model.get_input_embeddings()

        all_combined = []
        max_combined_len = labels.shape[1]

        anchored_pooled_projections = []
        anchored_class_ids = []

        for i in range(B):
            text_emb = embed_layer(input_ids[i])

            T_i = ts_len[i].item()
            ts_i = ts_raw[i, :T_i].unsqueeze(-1)
            ts_encoded_i = self.ts_encoder(ts_i)
            ts_proj_i = self.projector(ts_encoded_i)

            ck = class_keys[i]
            if ck is not None and ck in self.class_to_id:
                anchored_pooled_projections.append(ts_proj_i.mean(dim=0))
                anchored_class_ids.append(self.class_to_id[ck])

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
                    torch.zeros(pad_len, combined.shape[1], device=device, dtype=combined.dtype),
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
        ce_loss = outputs.loss

        # When the aux term is disabled, skip all contrastive bookkeeping so the
        # weight=0.0 control is a clean "CE only" run.
        if self.aux_loss_weight == 0.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        if len(anchored_pooled_projections) == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        if len(self.queue_projections) < QUEUE_WARMUP_MIN:
            for proj_vec, cid in zip(anchored_pooled_projections, anchored_class_ids):
                self._enqueue(proj_vec.detach(), cid)
            return (ce_loss, outputs) if return_outputs else ce_loss

        batch_projections = torch.stack(anchored_pooled_projections, dim=0).float()

        pool_class_ids_list = list(self.queue_class_ids)
        pool_class_ids = torch.tensor(pool_class_ids_list, device=device)
        pool_targets = torch.stack(
            [self.id_to_target[cid] for cid in pool_class_ids_list], dim=0
        ).to(device).float()

        batch_proj_norm = F.normalize(batch_projections, dim=-1)
        pool_targets_norm = F.normalize(pool_targets, dim=-1)

        sim = batch_proj_norm @ pool_targets_norm.T / AUX_LOSS_TEMPERATURE

        batch_class_ids_tensor = torch.tensor(anchored_class_ids, device=device)
        mask = (batch_class_ids_tensor.unsqueeze(1) == pool_class_ids.unsqueeze(0)).float()

        sim_max = sim.max(dim=-1, keepdim=True).values
        sim_stable = sim - sim_max.detach()
        exp_sim = sim_stable.exp()
        log_prob = sim_stable - exp_sim.sum(dim=-1, keepdim=True).log()

        num_positives = mask.sum(dim=-1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=-1) / num_positives.clamp(min=1)

        valid_rows = num_positives > 0
        if valid_rows.sum() == 0:
            aux_loss = torch.tensor(0.0, device=device)
        else:
            aux_loss = -mean_log_prob_pos[valid_rows].mean()

        total_loss = ce_loss + self.aux_loss_weight * aux_loss

        for proj_vec, cid in zip(anchored_pooled_projections, anchored_class_ids):
            self._enqueue(proj_vec.detach(), cid)

        return (total_loss, outputs) if return_outputs else total_loss


# ============================================================
# Build a fresh model/tokenizer/encoder/projector + label targets
# ============================================================
def build_components():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-0.6B",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    d_llm = model.config.hidden_size

    tokenizer.add_special_tokens({"additional_special_tokens": ["<ts>"]})
    model.resize_token_embeddings(len(tokenizer))
    ts_token_id = tokenizer.convert_tokens_to_ids("<ts>")

    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407, use_rslora=True, loftq_config=None,
    )

    ts_encoder = nn.Linear(1, D_TS).to("cuda")
    ts_encoder.train()
    projector = TimeSeriesProjector(D_TS, d_llm).to("cuda")

    # Label-target lookup tables (depend only on the frozen base embeddings;
    # identical across runs because embedding_learning_rate=0).
    label_targets, class_to_id = {}, {}
    embed = model.get_input_embeddings()
    for cid, cls in enumerate(CLASSIFICATION_LABELS):
        token_ids = tokenizer.encode(cls.lower(), add_special_tokens=False)
        with torch.no_grad():
            vec = embed(torch.tensor(token_ids, device="cuda")).mean(dim=0).detach().float()
        key = ("classification", cls)
        label_targets[key] = vec
        class_to_id[key] = cid
    id_to_target = {cid: label_targets[k] for k, cid in class_to_id.items()}
    assert len(label_targets) == 8

    return model, tokenizer, ts_token_id, d_llm, ts_encoder, projector, \
        label_targets, class_to_id, id_to_target


# ============================================================
# Training
# ============================================================
def train(queue_size, aux_loss_weight, train_data_dir):
    (model, tokenizer, ts_token_id, d_llm, ts_encoder, projector,
     label_targets, class_to_id, id_to_target) = build_components()

    raw_dataset = concatenate_datasets([
        load_dataset("csv", data_files=f"{train_data_dir}/anomaly_detection.csv", split="train"),
        load_dataset("csv", data_files=f"{train_data_dir}/classification.csv",    split="train"),
        load_dataset("csv", data_files=f"{train_data_dir}/multiple_choice.csv",   split="train"),
        load_dataset("csv", data_files=f"{train_data_dir}/true_false.csv",        split="train"),
        load_dataset("csv", data_files=f"{train_data_dir}/forecasting.csv",       split="train"),
        load_dataset("csv", data_files=f"{train_data_dir}/imputation.csv",        split="train"),
    ]).shuffle(seed=42)

    raw_dataset = raw_dataset.filter(
        lambda x: (
            x["QA_list"] is not None and isinstance(x["QA_list"], str)
            and len(x["QA_list"].strip()) > 0
            and x["time_series_norm"] is not None
            and x["question_text"] is not None
            and x["statistical_features"] is not None
        )
    )

    train_dataset = TimeSeriesTextDataset(
        raw_dataset, tokenizer, ts_token_id, MAX_SEQ_LENGTH, d_llm)

    w_tag = weight_tag(aux_loss_weight)
    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=2000,
        warmup_steps=500,
        learning_rate=5e-5,
        embedding_learning_rate=0,   # keeps cached label targets valid
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=0,
        output_dir=f"outputs/qsize_{queue_size}_w{w_tag}",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    collator = EmbeddingCollator(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    trainer = MultimodalTrainer(
        projector=projector, ts_encoder=ts_encoder,
        label_targets=label_targets, class_to_id=class_to_id,
        id_to_target=id_to_target, queue_size=queue_size,
        aux_loss_weight=aux_loss_weight,
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset, data_collator=collator, args=training_args,
    )

    original_create_optimizer = trainer.create_optimizer

    def create_optimizer_with_extra_params():
        original_create_optimizer()
        trainer.optimizer.add_param_group({
            "params": list(projector.parameters()),
            "lr": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
        })
        trainer.optimizer.add_param_group({
            "params": list(ts_encoder.parameters()),
            "lr": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
        })

    trainer.create_optimizer = create_optimizer_with_extra_params

    logging.info(f"[qsize={queue_size} w={w_tag}] starting training on "
                 f"{len(raw_dataset)} samples")
    trainer.train()
    logging.info(f"[qsize={queue_size} w={w_tag}] training done")

    return model, tokenizer, ts_token_id, projector, ts_encoder


# ============================================================
# In-memory inference (no save/reload; reuses the trained objects)
# ============================================================
@torch.no_grad()
def generate_from_embeds(model, inputs_embeds, attention_mask,
                         stop_token_ids, max_new_tokens):
    device = inputs_embeds.device
    seq_len = inputs_embeds.shape[1]
    generated_ids = []

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=True, return_dict=True)
    past_key_values = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    generated_ids.append(next_token_id.item())
    if next_token_id.item() in stop_token_ids:
        return generated_ids

    current_pos = seq_len
    for _ in range(max_new_tokens - 1):
        attention_mask = torch.cat(
            [attention_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1)
        step_position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=device)
        outputs = model(input_ids=next_token_id.unsqueeze(0),
                        attention_mask=attention_mask, position_ids=step_position_ids,
                        past_key_values=past_key_values, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_ids.append(next_token_id.item())
        current_pos += 1
        if next_token_id.item() in stop_token_ids:
            break
    return generated_ids


def _parse_list(raw):
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


def _parse_stats(raw):
    if isinstance(raw, dict):
        return raw
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return json.loads(raw)


def _parse_qa_list(qa_string):
    if not qa_string or pd.isna(qa_string):
        return None
    try:
        cleaned = qa_string.strip().replace('\n', ' ').replace('\r', ' ')
        if not cleaned.startswith('{'):
            cleaned = '{' + cleaned + '}'
        return json.loads(cleaned)
    except Exception:
        return None


def run_inference(model, tokenizer, ts_token_id, projector, ts_encoder,
                  test_data_dir, out_dir, max_new_tokens):
    FastLanguageModel.for_inference(model)
    ts_encoder.eval()
    projector.eval()

    eos_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {eos_id}
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        stop_ids.add(im_end_id)

    def gen_multimodal(question_text, stats, ts_norm_list):
        ts_tensor = torch.tensor(ts_norm_list, dtype=torch.float32).to("cuda").unsqueeze(-1)
        with torch.no_grad():
            ts_embeds = ts_encoder(ts_tensor)
            ts_proj = projector(ts_embeds)

        stats_prompt = (
            f"\nThe above is the normalized time series data. "
            f"Its raw data has the following statistical information: "
            f"minimum: {stats['min']}, maximum: {stats['max']}, "
            f"median: {stats['median']}, mean: {stats['mean']}."
        )
        full_prompt = (
            f"<|im_start|>user\n{question_text}{stats_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tokenizer(full_prompt, return_tensors="pt",
                              truncation=False, add_special_tokens=True
                              )["input_ids"].squeeze(0).to("cuda")

        ts_positions = (input_ids == ts_token_id).nonzero(as_tuple=True)[0]
        if len(ts_positions) == 0:
            return "Error: <ts> token not found"
        ts_pos = ts_positions[0].item()

        embed_layer = model.get_input_embeddings()
        text_emb = embed_layer(input_ids)
        combined = torch.cat([text_emb[:ts_pos], ts_proj, text_emb[ts_pos + 1:]],
                             dim=0).unsqueeze(0)
        attn = torch.ones(1, combined.shape[1], dtype=torch.long, device="cuda")
        gen_ids = generate_from_embeds(model, combined, attn, stop_ids, max_new_tokens)
        resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return resp.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    test_files = {
        "Classification": "classification.csv",
        "Anomaly": "anomaly_detection.csv",
        "Forecasting": "forecasting.csv",
        "Multiple_choice": "multiple_choice.csv",
        "True_false": "true_false.csv",
        "Imputation": "imputation.csv",
    }

    os.makedirs(out_dir, exist_ok=True)
    for task_name, file_name in test_files.items():
        file_path = os.path.join(test_data_dir, file_name)
        if not os.path.exists(file_path):
            logging.warning(f"Skipping {task_name}: {file_path} not found")
            continue

        df = pd.read_csv(file_path)
        predictions = []
        for index, row in df.iterrows():
            ts_raw = row.get("time_series_norm")
            q_raw = row.get("question_text")
            stats_raw = row.get("statistical_features")
            has_mm = (
                ts_raw is not None and not (isinstance(ts_raw, float) and pd.isna(ts_raw))
                and q_raw is not None and not (isinstance(q_raw, float) and pd.isna(q_raw))
                and stats_raw is not None and not (isinstance(stats_raw, float) and pd.isna(stats_raw))
            )
            try:
                if has_mm:
                    stats = _parse_stats(stats_raw)
                    if stats is None:
                        predictions.append("Error: Could not parse statistical_features")
                        continue
                    predictions.append(
                        gen_multimodal(str(q_raw), stats, _parse_list(ts_raw)))
                else:
                    qa_dict = _parse_qa_list(row.get("QA_list"))
                    if qa_dict is None:
                        predictions.append("Error: Could not parse QA_list")
                        continue
                    question = qa_dict.get("question", "")
                    ctx = (f"Application Domain: {row.get('application_domain','')}\n"
                           f"Task Type: {row.get('task_type','')}\n\n{question}")
                    prompt = f"<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n"
                    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         use_cache=True, pad_token_id=eos_id,
                                         eos_token_id=eos_id, temperature=0.1, do_sample=False)
                    gen = out[:, inputs["input_ids"].shape[1]:]
                    resp = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
                    predictions.append(
                        resp.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip())
            except Exception as e:
                predictions.append(f"Error: {e}")
            if index % 10 == 0:
                logging.info(f"[{task_name}] {index}/{len(df)}")

        df["model_response"] = predictions
        out_path = os.path.join(out_dir, f"results_{task_name.lower()}.csv")
        df.to_csv(out_path, index=False)
        logging.info(f"Saved {out_path}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s:%(message)s")
    w_tag = weight_tag(args.aux_loss_weight)
    logging.info(f"=== ABLATION RUN: AUX_LOSS_WEIGHT={args.aux_loss_weight}, "
                 f"QUEUE_SIZE={args.queue_size} (fixed) ===")

    model, tokenizer, ts_token_id, projector, ts_encoder = train(
        args.queue_size, args.aux_loss_weight, args.train_data_dir)

    out_dir = os.path.join(args.output_root, f"qsize_{args.queue_size}_w{w_tag}")
    run_inference(model, tokenizer, ts_token_id, projector, ts_encoder,
                  args.test_data_dir, out_dir, args.max_new_tokens)

    if args.save_model:
        save_dir = os.path.join("models", f"qsize_{args.queue_size}_w{w_tag}")
        os.makedirs(os.path.join(save_dir, "projector"), exist_ok=True)
        torch.save(projector.state_dict(), os.path.join(save_dir, "projector", "projector.pt"))
        torch.save(ts_encoder.state_dict(), os.path.join(save_dir, "projector", "ts_encoder.pt"))
        model.save_pretrained_merged(os.path.join(save_dir, "final_model_merged"),
                                     tokenizer, save_method="merged_16bit")
        logging.info(f"Saved model artifacts to {save_dir}")

    logging.info(f"=== DONE: AUX_LOSS_WEIGHT={args.aux_loss_weight}, "
                 f"QUEUE_SIZE={args.queue_size} ===")


if __name__ == "__main__":
    main()