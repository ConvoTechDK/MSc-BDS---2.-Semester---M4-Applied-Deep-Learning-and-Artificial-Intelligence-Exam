"""
Final Assignment - Step 03: Final PatentSBERTa Fine-Tuning.

Performs the final fine-tuning of PatentSBERTa using gold labels from the
QLoRA-powered MAS with exception-based HITL (from 02_mas_qlora.py).

Same architecture as Assignments 2 and 3 fine-tuning scripts.

Run on HPC: sbatch slurm_mas_final.sh (as part of that pipeline)
Output: ./results_final/ (model checkpoint), pushed to Peter512/patentsbert-green-final
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
FINAL_CSV    = os.path.join(os.path.dirname(__file__), 'final_labels_100.csv')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'results_final')
HF_REPO      = "Peter512/patentsbert-green-final"

print("Loading datasets...")
df_main  = pd.read_parquet(PARQUET_PATH)
df_final = pd.read_csv(FINAL_CSV)

# Merge final gold labels: gold overrides silver for the 100 claims
df_main = df_main.copy()
df_main['label'] = df_main['is_green_silver']

gold_lookup = dict(zip(df_final['doc_id'].astype(str), df_final['is_green_gold'].astype(int)))
df_main['label'] = df_main.apply(
    lambda row: gold_lookup.get(str(row['doc_id']), int(row['is_green_silver'])),
    axis=1
)

train_df = df_main[df_main['split'] == 'train_silver'].reset_index(drop=True)
eval_df  = df_main[df_main['split'] == 'eval_silver'].reset_index(drop=True)

# Gold_100 with QLoRA MAS labels
gold_df = df_final[['doc_id', 'text', 'is_green_gold']].rename(
    columns={'is_green_gold': 'label'}
).copy()
gold_df['label'] = gold_df['label'].astype(int)

# Add the 100 gold items to training (they come from pool_unlabeled, not train_silver)
gold_train = gold_df[['text', 'label']].copy()
train_df = pd.concat([train_df, gold_train], ignore_index=True)

print(f"train (silver + gold_100): {len(train_df)} rows")
print(f"eval_silver:  {len(eval_df)} rows")
print(f"gold_100:     {len(gold_df)} rows")

# HuggingFace datasets
def make_hf_dataset(df_src, label_col='label'):
    return Dataset.from_dict({
        'text':   df_src['text'].tolist(),
        'labels': df_src[label_col].astype(int).tolist(),
    })

train_ds = make_hf_dataset(train_df)
eval_ds  = make_hf_dataset(eval_df)
gold_ds  = make_hf_dataset(gold_df)

# Tokenize
print("\nTokenizing...")
tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa')

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
eval_ds  = eval_ds.map(tokenize,  batched=True, desc="Tokenizing eval")
gold_ds  = gold_ds.map(tokenize,  batched=True, desc="Tokenizing gold_100")

# Load model
print("\nLoading PatentSBERTa...")
model = AutoModelForSequenceClassification.from_pretrained(
    'AI-Growth-Lab/PatentSBERTa',
    num_labels=2,
    ignore_mismatched_sizes=True,
)

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

print("\nStarting final fine-tuning (1 epoch)...")
trainer.train()

print("\nEvaluation on eval_silver:")
eval_results = trainer.evaluate()
print(eval_results)

print("\nEvaluation on gold_100 (QLoRA MAS labels):")
gold_results = trainer.evaluate(eval_dataset=gold_ds)
print(gold_results)

print(f"\nPushing final model to HuggingFace Hub: {HF_REPO}")
trainer.model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"Final model: https://huggingface.co/{HF_REPO}")

print("\nFinal assignment complete.")
print(f"eval_silver  accuracy={eval_results.get('eval_accuracy','N/A'):.4f}  F1={eval_results.get('eval_f1','N/A'):.4f}  P={eval_results.get('eval_precision','N/A'):.4f}  R={eval_results.get('eval_recall','N/A'):.4f}")
print(f"gold_100     accuracy={gold_results.get('eval_accuracy','N/A'):.4f}  F1={gold_results.get('eval_f1','N/A'):.4f}  P={gold_results.get('eval_precision','N/A'):.4f}  R={gold_results.get('eval_recall','N/A'):.4f}")
