"""
Final Assignment - Step 00: QLoRA Inference on 100 High-Risk Claims.

Loads the trained QLoRA adapter (from 01_qlora_train.py) and runs inference
on the 100 high-risk claims selected in Assignment 2.

The classification decision is made by comparing logits for the tokens
"yes" and "no" at the next-token position after "Answer:", which is how
the model was trained. This produces a clean 0/1 label per claim.

The resulting qlora_predictions_100.csv is read by 02_mas_qlora.py to
inject the QLoRA model's assessment into the Advocate agent's task.

Run on HPC: sbatch slurm_qlora_inference.sh (GPU required, ~5-10 min)
Output: qlora_predictions_100.csv
"""

import os
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

OUTPUT_DIR  = os.path.dirname(__file__)
ADAPTER_DIR = os.path.join(OUTPUT_DIR, 'qlora_patent_adapter')
INPUT_CSV   = os.path.join(OUTPUT_DIR, '..', 'assignment_2', 'hitl_green_100.csv')
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, 'qlora_predictions_100.csv')

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

if not os.path.isdir(ADAPTER_DIR):
    raise FileNotFoundError(
        f"Adapter not found at {ADAPTER_DIR}. "
        "Run 01_qlora_train.py first."
    )

print(f"Adapter directory: {ADAPTER_DIR}")
print(f"Base model:        {BASE_MODEL}")
print(f"CUDA available:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"VRAM:              {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 4-bit quantization (same config as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\nLoading base model with 4-bit quantization...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.pad_token = tokenizer.eos_token

# Get token IDs for "yes" and "no" as they appear after "Answer: "
# Encoded with a leading space to match the training format
yes_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
no_id  = tokenizer.encode(" no",  add_special_tokens=False)[0]
print(f"\nClassification token IDs — ' yes': {yes_id}, ' no': {no_id}")

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} claims from {INPUT_CSV}")
print("\nRunning QLoRA inference on 100 claims...")
print("-" * 60)

records = []
with torch.no_grad():
    for idx, row in df.iterrows():
        # Same prompt format used during training in 01_qlora_train.py
        prompt = (
            f"Claim: {str(row['text'])[:400]}\n"
            f"Is this a green technology patent under CPC Y02? Answer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Compare logits for "yes" vs "no" at the answer position
        outputs   = model(**inputs)
        logits    = outputs.logits[0, -1, :]
        yes_logit = logits[yes_id].item()
        no_logit  = logits[no_id].item()
        qlora_label = 1 if yes_logit > no_logit else 0

        # Generate a short continuation for transparency in the Advocate task
        gen = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw_output = tokenizer.decode(
            gen[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        records.append({
            'doc_id':       row['doc_id'],
            'qlora_label':  qlora_label,
            'qlora_output': raw_output,
        })

        label_str = "GREEN    " if qlora_label == 1 else "NOT GREEN"
        print(
            f"  [{idx + 1:3d}/100] {label_str} "
            f"(yes={yes_logit:+.2f}, no={no_logit:+.2f}) "
            f"raw='{raw_output}'"
        )

df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False)

print("-" * 60)
print(f"\nSaved: {OUTPUT_CSV}")
print(f"QLoRA label distribution: {df_out['qlora_label'].value_counts().to_dict()}")
print(f"  GREEN:     {(df_out['qlora_label'] == 1).sum()} / 100")
print(f"  NOT GREEN: {(df_out['qlora_label'] == 0).sum()} / 100")
print("\nQLoRA inference complete. Next step: sbatch slurm_mas_final.sh")
