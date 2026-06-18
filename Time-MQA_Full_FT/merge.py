"""
Qwen3-0.6B FULL FINE-TUNE + Infer in ONE Run (no save / no reload)
==================================================================
Full-parameter fine-tuning version of the original QLoRA script. Differences
from the LoRA version:
  * Model is loaded in bf16 (load_in_4bit=False) with full_finetuning=True.
    You cannot full-fine-tune 4-bit weights, so QLoRA is off.
  * The get_peft_model(...) / LoRA block is removed entirely. ALL parameters
    are trainable.
  * Learning rate lowered to 2e-5 (full FT forgets faster than LoRA, so a
    gentler LR is safer). embedding_learning_rate kept since embeddings are
    now trainable too.
  * adamw_8bit kept to keep the optimizer-state footprint small (~6 GB static
    for 0.6B vs ~10 GB with fp32 Adam).

After training, the in-memory model + tokenizer are used directly for
generation. Output is the per-task result CSVs in results/qwen3-0.6B-full/.

VRAM note: at batch 4 / seq 8192 this lands around ~12-18 GB. If you OOM,
drop per_device_train_batch_size to 1-2 and raise gradient_accumulation_steps
to keep the effective batch size constant.
"""

# Inference-related env vars must be set BEFORE importing unsloth.
import os
os.environ.setdefault("UNSLOTH_DISABLE_GEMMA_PATCH", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import unsloth
import logging
import re
import ast
import json
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

# ============================================================
# 0. Logging Setup
# ============================================================
log_file = "training_log.txt"  # created automatically if it doesn't exist
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("Starting combined FULL fine-tune + inference script (Qwen3)...")

# ============================================================
# 1. Load Dataset
# ============================================================
logging.info("Loading dataset...")
dataset = concatenate_datasets([
    load_dataset("csv", data_files="data/train/anomaly_detection.csv", split="train"),
    load_dataset("csv", data_files="data/train/classification.csv",    split="train"),
    load_dataset("csv", data_files="data/train/multiple_choice.csv",   split="train"),
    load_dataset("csv", data_files="data/train/true_false.csv",        split="train"),
    load_dataset("csv", data_files="data/train/forecasting.csv",       split="train"),
    load_dataset("csv", data_files="data/train/imputation.csv",        split="train"),
])
dataset = dataset.shuffle(seed=42)
dataset = dataset.filter(
    lambda x: x["QA_list"] is not None
    and isinstance(x["QA_list"], str)
    and len(x["QA_list"].strip()) > 0
)
logging.info(f"Dataset loaded: {len(dataset)} samples after filtering.")

# ============================================================
# 2. Load Model & Tokenizer (FULL FINE-TUNE, bf16 — no 4-bit)
#    Reused for training + inference.
# ============================================================
max_seq_length = 8192
dtype = None          # auto: bf16 on Ampere+, fp16 otherwise
load_in_4bit = False  # MUST be False for full fine-tuning

logging.info("Loading model and tokenizer for full fine-tuning...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    full_finetuning=True,   # <-- enables full-parameter fine-tuning in Unsloth
)

# ============================================================
# 3. (No LoRA / no PEFT)
# ============================================================
# Full fine-tuning trains every parameter, so there is no get_peft_model call.
# Gradient checkpointing still helps a lot with activation memory:
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# Quick sanity check on trainable parameter count.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
logging.info(f"Trainable params: {trainable:,} / {total:,} "
             f"({100 * trainable / total:.1f}%)")

# ============================================================
# 4. Training Arguments
# ============================================================
logging.info("Setting up training arguments...")
training_args = UnslothTrainingArguments(
    per_device_train_batch_size=4,     # drop to 1-2 if you OOM
    gradient_accumulation_steps=8,     # raise this to keep effective batch size
    max_steps=2000,
    warmup_steps=500,
    learning_rate=2e-5,                # lowered for full FT (was 5e-5 for LoRA)
    embedding_learning_rate=1e-5,      # embeddings are trainable now
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",                # keeps optimizer-state memory ~1.2 GB
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    seed=0,
    output_dir="outputs",
    report_to="none",
    save_strategy="no",                # disable checkpoint saving
)


def formatting_func(examples):
    texts = examples["QA_list"]
    if isinstance(texts, str):
        return [texts]
    return list(texts)


# ============================================================
# 5. Initialize Trainer
# ============================================================
logging.info("Initializing the trainer...")
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=training_args,
)

# ============================================================
# 6. Log GPU Stats & Train
# ============================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
logging.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
logging.info(f"{start_gpu_memory} GB of memory reserved.")

logging.info("Starting model training (full fine-tune)...")
try:
    trainer_stats = trainer.train()
    logging.info("Training completed successfully!")
except Exception:
    logging.error("An error occurred during training:", exc_info=True)
    raise  # don't proceed to inference if training failed

# Peak memory report — handy for confirming the VRAM estimate.
peak = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
logging.info(f"Peak reserved GPU memory during training: {peak} GB "
             f"({100 * peak / max_memory:.1f}% of {max_memory} GB).")

# ============================================================
# 7. Switch to Inference (NO save / NO reload)
# ============================================================
# The fully fine-tuned `model` stays in GPU memory and is used directly.
logging.info("Switching trained model to inference mode (no checkpoint reload)...")
FastLanguageModel.for_inference(model)
model.eval()

# ============================================================
# 8. Generation Helper
# ============================================================
def generate_answer(prompt_text):
    full_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
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
    return response


# ============================================================
# 9. QA_list Parser
# ============================================================
def parse_qa_list(qa_string):
    if not qa_string or pd.isna(qa_string):
        return None

    # Try clean JSON first
    for attempt in [qa_string.strip(), '{' + qa_string.strip() + '}']:
        try:
            return json.loads(attempt)
        except Exception:
            pass

    # Fallback: extract question / answer via string matching
    try:
        s = qa_string.strip()
        q_match = re.search(r'"question"\s*:\s*"', s)
        a_match = re.search(r'",\s*"answer"\s*:\s*"', s)

        if q_match and a_match:
            question = s[q_match.end():a_match.start()]
            answer_start = a_match.end()
            answer = s[answer_start:].rstrip().rstrip('}').rstrip('"')
            return {"question": question, "answer": answer}
        elif q_match:
            question = s[q_match.end():].rstrip().rstrip('}').rstrip('"')
            return {"question": question}
    except Exception:
        pass

    return None


# ============================================================
# 10. Test Datasets
# ============================================================
base_data_path = "/weka/s225635478/CADE/data/test"
test_files = {
    "Classification": "classification.csv",
    "Anomaly": "anomaly_detection.csv",
    "Forecasting": "forecasting.csv",
    "Imputation": "imputation.csv",
    "Multiple_choice": "multiple_choice.csv",
    "True_false": "true_false.csv",
}

# ============================================================
# 11. Execution Loop -> write result CSVs
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
        qa_list = row.get('QA_list')
        application_domain = row.get('application_domain', '')
        task_type = row.get('task_type', '')

        if not qa_list or pd.isna(qa_list):
            predictions.append("Error: No QA_list found")
            continue

        qa_dict = parse_qa_list(qa_list)

        if qa_dict is None:
            predictions.append("Error: Could not parse QA_list")
            print(f"Row {index}: Failed to parse: {qa_list[:100]}...")
            continue

        question = qa_dict.get('question', '')
        if not question:
            predictions.append("Error: No question found in QA_list")
            continue

        full_context = f"Application Domain: {application_domain}\nTask Type: {task_type}\n\n{question}"

        # Format constraint for forecasting / imputation tasks
        if task_name in ["Forecasting", "Imputation"]:
            full_context += (
                "\nPlease respond with ONLY the predicted values in a Python list "
                "format [v1, v2, ..., vN], where N equals the exact number of "
                "requested points. No explanations."
            )

        ans = generate_answer(full_context)
        predictions.append(ans)

        if index % 10 == 0:
            print(f"Processed {index}/{len(df)} rows...")

    df['model_response'] = predictions
    output_dir = "results/qwen+full_finetuning"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"results_{task_name.lower()}.csv")
    df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}")
    logging.info(f"Saved results to {output_filename}")

print("\n--- Full fine-tune + inference completed successfully ---")
logging.info("Full fine-tune + inference completed successfully.")