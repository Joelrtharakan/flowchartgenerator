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
    prepare_model_for_kbit_training,
)

# ─────────────────────────────────────────────
# 0. GPU GUARD — fail fast if no CUDA
# ─────────────────────────────────────────────
load_dotenv() # Load variables from .env file into os.environ

if not torch.cuda.is_available():
    sys.exit(
        "\n❌  No CUDA GPU detected.\n"
        "    This script requires a GPU.  Options:\n"
        "      • Google Colab  → Runtime > Change runtime type > T4 GPU (free)\n"
        "      • RunPod / vast.ai → cheapest GPU instance\n"
        "      • Local RTX 3060 / 3090 / 4090\n"
    )

GPU_NAME = torch.cuda.get_device_name(0)
USE_BF16 = torch.cuda.is_bf16_supported()   # Ampere+ (A100, RTX 30xx/40xx)
USE_FP16 = not USE_BF16                     # Older cards (T4, V100, RTX 20xx)
DTYPE    = torch.bfloat16 if USE_BF16 else torch.float16

print(f"✅  GPU   : {GPU_NAME}")
print(f"   VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"   dtype : {'bfloat16' if USE_BF16 else 'float16'}")

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

from transformers import BitsAndBytesConfig

# ─────────────────────────────────────────────
# 4. MODEL  (QLoRA — 4-bit on GPU)
# ─────────────────────────────────────────────
print("\n[4/6] Loading model in 4-bit (QLoRA) …")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=DTYPE
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype       = DTYPE,  # bf16 or fp16 for LoRA adapter weights
    device_map        = "auto", # fills all available GPU VRAM automatically
)

# Mandatory before applying LoRA to a quantised model
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)
model.config.use_cache = False  # must be False for gradient checkpointing

# ─────────────────────────────────────────────
# 5. LoRA CONFIG
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r              = 16,        # rank — well-suited for ~500 samples
    lora_alpha     = 32,        # effective scale = alpha/r = 2.0
    target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
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
# 7. TRAINING ARGUMENTS  (GPU-optimised)
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = "./phi3-finetuned",

    # Epochs & batching
    num_train_epochs            = 3,
    per_device_train_batch_size = 4,   # bump to 8 if VRAM > 16 GB
    gradient_accumulation_steps = 4,   # effective batch = 16

    # Optimiser
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 5,
    weight_decay                = 0.01,
    max_grad_norm               = 1.0,
    optim                       = "paged_adamw_32bit",  # saves GPU VRAM

    # Precision — auto-set from GPU capability
    fp16                        = USE_FP16,
    bf16                        = USE_BF16,

    # Evaluation
    eval_strategy               = "steps",
    eval_steps                  = 50,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,

    # Logging & checkpoints
    logging_steps               = 10,
    save_steps                  = 50,
    save_total_limit            = 2,
    report_to                   = "none",  # swap to "wandb" if you use W&B

    # GPU memory savers
    gradient_checkpointing      = True,
    dataloader_pin_memory       = True,
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
SAVE_DIR = "./phi3-finetuned"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅  Done.  Model saved → {SAVE_DIR}/")
print(f"   Peak GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("   Next step: evaluate_phi3.py  →  deploy")