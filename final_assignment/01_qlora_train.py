"""
Final Assignment - Step 01: QLoRA Domain Adaptation of Llama-3.2-3B.

Fine-tunes Llama-3.2-3B-Instruct on train_silver patent claims using 4-bit
quantization (QLoRA) via peft + bitsandbytes. The goal is domain adaptation to
the dense linguistic style of patent claims and Y02 classification logic.

Architecture mirrors M3_3_Finetune_opt_bnb_peft notebook exactly:
  - 4-bit NF4 quantization via BitsAndBytesConfig
  - LoRA adapters on q_proj and v_proj (r=16, alpha=32)
  - Freeze base weights, enable gradient checkpointing
  - CausalLM training with classification prompt format

Run on HPC: sbatch slurm_qlora.sh (GPU required, ~40GB VRAM)
Output: ./qlora_patent_adapter/ (LoRA adapter weights)
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'qlora_patent_adapter')
PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'patents_50k_green.parquet')

# Smaller model for HPC feasibility (3B instead of 8B)
# Can change to "meta-llama/Llama-3.1-8B-Instruct" if sufficient VRAM
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print(f"Model: {MODEL_NAME}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 4-bit quantization config (matches M3_3_NLG_8 and M3_3_Finetune_opt_bnb_peft notebooks)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\nLoading base model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Freeze all base model parameters (mirrors M3_3_Finetune_opt_bnb_peft exactly)
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# LoRA config (mirrors M3_3_Finetune_opt_bnb_peft notebook)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and format training data
print("\nPreparing training data...")
df = pd.read_parquet(PARQUET_PATH)
train_df = df[df['split'] == 'train_silver'].sample(10000, random_state=42)

def format_claim(row):
    """Format patent claim as a classification prompt for causal LM training."""
    label_text = "yes" if row['is_green_silver'] == 1 else "no"
    return (
        f"Claim: {row['text'][:400]}\n"
        f"Is this a green technology patent under CPC Y02? Answer: {label_text}"
    )

train_df = train_df.copy()
train_df['formatted'] = train_df.apply(format_claim, axis=1)

print(f"Training samples: {len(train_df)}")
print("Example prompt:")
print(train_df['formatted'].iloc[0][:300])

# Tokenize
print("\nTokenizing...")
encodings = tokenizer(
    train_df['formatted'].tolist(),
    truncation=True,
    max_length=256,
    padding='max_length',
    return_tensors='pt',
)

ds = Dataset.from_dict({
    'input_ids':      encodings['input_ids'].tolist(),
    'attention_mask': encodings['attention_mask'].tolist(),
})
print(f"Dataset size: {len(ds)} samples")

# Training (mirrors M3_3_Finetune_opt_bnb_peft notebook)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=200,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to='none',
)

trainer = Trainer(
    model=model,
    train_dataset=ds,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False

print("\nStarting QLoRA training...")
trainer.train()

print(f"\nSaving LoRA adapter to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("QLoRA training complete.")
print(f"Adapter saved to: {os.path.abspath(OUTPUT_DIR)}")
