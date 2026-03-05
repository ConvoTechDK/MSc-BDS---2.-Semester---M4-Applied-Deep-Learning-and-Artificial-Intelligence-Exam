"""
Assignment 2 - Step 04: Fine-Tune PatentSBERTa for Binary Classification.

Merges the 100 HITL gold labels into the dataset (gold overrides silver for those 100),
then fine-tunes PatentSBERTa using HuggingFace Trainer for binary green/not-green
classification.

Settings (per assignment spec):
  max_seq_length = 256
  epochs = 1
  learning_rate = 2e-5

Evaluates on eval_silver AND gold_100 separately.

Run on HPC: sbatch slurm_finetune_a2.sh
Run locally: python 04_finetune_patentsbert.py
Output: ./results_a2/ (model checkpoint), pushed to Peter512/patentsbert-green-a2
"""

import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'patents_50k_green.parquet')
GOLD_CSV     = os.path.join(os.path.dirname(__file__), 'hitl_green_100_labeled.csv')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'results_a2')
HF_REPO      = "Peter512/patentsbert-green-a2"

print("Loading datasets...")
df_main = pd.read_parquet(PARQUET_PATH)
df_gold = pd.read_csv(GOLD_CSV)

# Merge gold labels: gold overrides silver for the 100 HITL items
df_main = df_main.copy()
df_main['is_green_gold'] = df_main['is_green_silver']

gold_lookup = dict(zip(df_gold['doc_id'].astype(str), df_gold['is_green_human'].astype(int)))
df_main['is_green_gold'] = df_main.apply(
    lambda row: gold_lookup.get(str(row['doc_id']), int(row['is_green_silver'])),
    axis=1
)

train_df = df_main[df_main['split'] == 'train_silver'].reset_index(drop=True)
eval_df  = df_main[df_main['split'] == 'eval_silver'].reset_index(drop=True)

# Gold_100: the 100 HITL-labeled claims with human labels
gold_df = df_gold[['doc_id', 'text', 'is_green_human']].copy()
gold_df = gold_df.rename(columns={'is_green_human': 'is_green_gold'})
gold_df['is_green_gold'] = gold_df['is_green_gold'].astype(int)

# Add the 100 gold items to training (they come from pool_unlabeled, not train_silver)
gold_train = gold_df[['text', 'is_green_gold']].copy()
train_df = pd.concat([train_df, gold_train], ignore_index=True)

print(f"train (silver + gold_100): {len(train_df)} rows")
print(f"eval_silver:  {len(eval_df)} rows")
print(f"gold_100:     {len(gold_df)} rows")
print(f"\nLabel distribution (train): {train_df['is_green_gold'].value_counts().to_dict()}")

# Build HuggingFace datasets
def make_hf_dataset(df_src, label_col='is_green_gold'):
    return Dataset.from_dict({
        'text':   df_src['text'].tolist(),
        'labels': df_src[label_col].astype(int).tolist(),
    })

train_ds = make_hf_dataset(train_df)
eval_ds  = make_hf_dataset(eval_df)
gold_ds  = make_hf_dataset(gold_df)

# Tokenize (matches M3_2_Transformermodels_NLU_FineTuning_huggingface_v3 pattern)
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa')

def tokenize(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256,
    )

print("Tokenizing datasets...")
train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
eval_ds  = eval_ds.map(tokenize,  batched=True, desc="Tokenizing eval")
gold_ds  = gold_ds.map(tokenize,  batched=True, desc="Tokenizing gold_100")

# Load model for sequence classification
print("\nLoading PatentSBERTa for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    'AI-Growth-Lab/PatentSBERTa',
    num_labels=2,
    ignore_mismatched_sizes=True,
)

# Evaluation metrics: accuracy + F1 + precision + recall
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy':  accuracy_score(labels, predictions),
        'f1':        f1_score(labels, predictions, average='binary'),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall':    recall_score(labels, predictions, average='binary'),
    }

# Training arguments (matches assignment spec and HF notebook pattern)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
    logging_steps=100,
    report_to='none',
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning (1 epoch)...")
trainer.train()

# Evaluate on eval_silver
print("\nEvaluation on eval_silver:")
eval_results = trainer.evaluate()
print(eval_results)

# Evaluate on gold_100
print("\nEvaluation on gold_100:")
gold_results = trainer.evaluate(eval_dataset=gold_ds)
print(gold_results)

# Push to HuggingFace Hub
print(f"\nPushing model to HuggingFace Hub: {HF_REPO}")
trainer.model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"Model pushed to: https://huggingface.co/{HF_REPO}")

print("\nFine-tuning complete.")
print(f"eval_silver  accuracy={eval_results.get('eval_accuracy','N/A'):.4f}  F1={eval_results.get('eval_f1','N/A'):.4f}  P={eval_results.get('eval_precision','N/A'):.4f}  R={eval_results.get('eval_recall','N/A'):.4f}")
print(f"gold_100     accuracy={gold_results.get('eval_accuracy','N/A'):.4f}  F1={gold_results.get('eval_f1','N/A'):.4f}  P={gold_results.get('eval_precision','N/A'):.4f}  R={gold_results.get('eval_recall','N/A'):.4f}")
