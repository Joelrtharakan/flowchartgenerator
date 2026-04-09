"""
train_phi3.py — Fine-tune Phi-3-mini on mermaid flowchart data using QLoRA
GPU REQUIRED  (CUDA).  Run on: local RTX / A100 / Colab T4 / RunPod.
"""

import os
import sys
import torch
import psycopg2
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

# ─────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────
load_dotenv() # Load variables from .env file into os.environ

# ─────────────────────────────────────────────
# 1. DATABASE — fetch training pairs
# ─────────────────────────────────────────────
print("\n[1/6] Fetching data from PostgreSQL …")

try:
    conn = psycopg2.connect(
        host     = os.environ["DB_HOST"],
        user     = os.environ["DB_USER"],
        password = os.environ["DB_PASSWORD"],
        dbname   = os.environ.get("DB_NAME", "postgres"),
        port     = int(os.environ.get("DB_PORT", 5432)),
    )
except KeyError as e:
    sys.exit(
        f"\n❌  Missing env var: {e}\n"
        "    Export DB_HOST, DB_USER, DB_PASSWORD before running.\n"
    )

cur = conn.cursor()
cur.execute("SELECT prompt, mermaid FROM flowcharts")
rows = cur.fetchall()
cur.close()
conn.close()

print(f"    Loaded {len(rows)} rows from DB")
if len(rows) == 0:
    sys.exit("❌  No rows in flowcharts table. Aborting.")

# ─────────────────────────────────────────────
# 2. FORMAT + QUALITY FILTER
#    Phi-3 chat template:
#    <|user|>\n{prompt}\n<|end|>\n<|assistant|>\n{mermaid}<|end|>
# ─────────────────────────────────────────────
print("\n[2/6] Filtering and formatting …")

def format_sample(prompt: str, mermaid: str) -> str:
    return (
        f"<|user|>\n{prompt.strip()}\n<|end|>\n"
        f"<|assistant|>\n{mermaid.strip()}<|end|>"
    )

JUNK_PATTERNS = [
    "Do something", "do something",
    "TODO", "placeholder",
    "...",
    "graph TD\n",   # empty graph — no nodes
]

data, skipped = [], 0
for p, m in rows:
    if any(pat in m for pat in JUNK_PATTERNS) or len(m.strip()) < 30:
        skipped += 1
        continue
    data.append({"text": format_sample(p, m)})

print(f"    Kept: {len(data)}  |  Skipped (low quality): {skipped}")
if len(data) == 0:
    sys.exit("❌  All samples filtered. Fix your data first.")

dataset = Dataset.from_list(data)

# ─────────────────────────────────────────────
# 3. TOKENIZER
# ─────────────────────────────────────────────
print("\n[3/6] Loading tokenizer …")

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Phi-3 has no dedicated pad token — reuse eos
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ─────────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────────
print("\n[4/6] Loading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ─────────────────────────────────────────────
# 5. LoRA CONFIG
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # expect ~1-2% — normal for QLoRA

# ─────────────────────────────────────────────
# 6. TOKENISE + TRAIN/VAL SPLIT
# ─────────────────────────────────────────────
print("\n[5/6] Tokenising and splitting …")

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation     = True,
        padding        = "max_length",
        max_length     = MAX_LENGTH,
        return_tensors = None,
    )
    result["labels"] = result["input_ids"].copy()  # collator masks pad → -100
    return result

# 90 / 10 split — fixed seed for reproducibility
split     = dataset.train_test_split(test_size=0.1, seed=42)
train_raw = split["train"]
val_raw   = split["test"]
print(f"    Train: {len(train_raw)}  |  Val: {len(val_raw)}")

train_dataset = train_raw.map(tokenize, remove_columns=["text"], desc="Tokenising train")
val_dataset   = val_raw.map(tokenize,   remove_columns=["text"], desc="Tokenising val")
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Masks padding positions in labels so they don't contribute to loss
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ─────────────────────────────────────────────
# 7. TRAINING ARGUMENTS
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./temp",

    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,

    learning_rate=2e-4,

    logging_steps=10,
    save_steps=50,

    fp16=False,
    bf16=False,

    evaluation_strategy="no",

    report_to="none",
)

# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────
print("\n[6/6] Starting training …\n")

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    data_collator = data_collator,
    processing_class = tokenizer,
)

trainer.train()

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
BASE_DIR = "models"
os.makedirs(BASE_DIR, exist_ok=True)

existing = os.listdir(BASE_DIR)
versions = [int(v.replace("v", "")) for v in existing if v.startswith("v")]

next_version = max(versions) + 1 if versions else 1

SAVE_DIR = f"{BASE_DIR}/v{next_version}"
os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅  Done.  Model saved → {SAVE_DIR}/")
print("   Next step: evaluate_phi3.py  →  deploy")