"""
Post-hoc evaluation: get exact F1/Precision/Recall for all 4 model versions.

No retraining. Loads each fine-tuned model from HuggingFace and evaluates
on the held-out eval_silver set (5,000 claims).

Run on HPC: sbatch slurm_eval.sh
Run locally (CPU, ~15 min): python eval_all_models.py

Output: prints a clean comparison table with exact metrics for all 4 models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset as TorchDataset

PARQUET_PATH = os.path.join(os.path.dirname(__file__), 'patents_50k_green.parquet')
A2_EMB_PATH  = os.path.join(os.path.dirname(__file__), 'assignment_2', 'eval_emb.npy')
CLF_PATH     = os.path.join(os.path.dirname(__file__), 'assignment_2', 'baseline_clf.pkl')

FINE_TUNED_MODELS = [
    ("Assignment 2 (GPT-4o-mini HITL)",   "Peter512/patentsbert-green-a2"),
    ("Assignment 3 (CrewAI MAS)",          "Peter512/patentsbert-green-a3"),
    ("Final (QLoRA MAS + Targeted HITL)", "Peter512/patentsbert-green-final"),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load dataset
print("\nLoading eval_silver...")
df = pd.read_parquet(PARQUET_PATH)
eval_df = df[df['split'] == 'eval_silver'].reset_index(drop=True)
y_true = eval_df['is_green_silver'].astype(int).tolist()
texts  = eval_df['text'].tolist()
print(f"eval_silver: {len(eval_df)} rows")

results = {}

# ── 1. Baseline (frozen embeddings + Logistic Regression) ────────────
print("\n[1/4] Baseline: Frozen PatentSBERTa + Logistic Regression")
if os.path.exists(A2_EMB_PATH) and os.path.exists(CLF_PATH):
    eval_emb = np.load(A2_EMB_PATH)
    clf = pickle.load(open(CLF_PATH, 'rb'))
    y_pred = clf.predict(eval_emb)
    results["Baseline"] = {
        'f1':        f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall':    recall_score(y_true, y_pred, average='binary'),
        'accuracy':  accuracy_score(y_true, y_pred),
    }
    print(classification_report(y_true, y_pred, target_names=['not_green', 'green']))
else:
    print("  Precomputed eval_emb.npy or baseline_clf.pkl not found.")
    print("  Run 01_baseline.py first, or copy from HPC.")
    results["Baseline"] = None


class TextDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        item = {k: v.squeeze(0) for k, v in enc.items()
                if k in ('input_ids', 'attention_mask', 'token_type_ids')}
        return item


def evaluate_hf_model(model_id, texts, y_true, device, batch_size=32, max_length=256):
    """Load a fine-tuned AutoModelForSequenceClassification and evaluate on texts."""
    print(f"  Loading tokenizer and model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained('AI-Growth-Lab/PatentSBERTa', use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(device)

    print(f"  Running inference on {len(texts)} examples...")
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    return all_preds


# ── 2-4. Fine-tuned models ──────────────────────────────────────────
for i, (label, model_id) in enumerate(FINE_TUNED_MODELS, start=2):
    print(f"\n[{i}/4] {label}")
    y_pred = evaluate_hf_model(model_id, texts, y_true, device)
    results[label] = {
        'f1':        f1_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall':    recall_score(y_true, y_pred, average='binary'),
        'accuracy':  accuracy_score(y_true, y_pred),
    }
    print(classification_report(y_true, y_pred, target_names=['not_green', 'green']))

# ── Summary table ───────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL COMPARISON TABLE (eval_silver, 5,000 claims)")
print("="*70)
print(f"{'Model':<42} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6}")
print("-"*70)

row_order = ["Baseline"] + [label for label, _ in FINE_TUNED_MODELS]
for label in row_order:
    r = results.get(label)
    if r:
        print(f"{label:<42} {r['f1']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['accuracy']:>6.4f}")
    else:
        print(f"{label:<42} {'N/A':>6}")

print("="*70)
print("\nCopy these F1 values into report.md, assignment READMEs, and the HF model cards.")
