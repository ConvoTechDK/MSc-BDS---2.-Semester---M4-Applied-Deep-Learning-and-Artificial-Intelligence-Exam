"""
Assignment 2 - Step 01: Baseline Model (Frozen PatentSBERTa + Logistic Regression).

Loads patents_50k_green.parquet, extracts frozen embeddings from PatentSBERTa,
trains a Logistic Regression classifier on train_silver, and evaluates on eval_silver.

Run: python 01_baseline.py
Output: baseline_clf.pkl, train_emb.npy, eval_emb.npy, pool_emb.npy
"""

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support

PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'patents_50k_green.parquet')
OUTPUT_DIR   = os.path.dirname(__file__)

print("Loading dataset...")
df = pd.read_parquet(PARQUET_PATH)

train = df[df['split'] == 'train_silver'].reset_index(drop=True)
eval_ = df[df['split'] == 'eval_silver'].reset_index(drop=True)
pool  = df[df['split'] == 'pool_unlabeled'].reset_index(drop=True)

print(f"  train_silver:   {len(train)} rows")
print(f"  eval_silver:    {len(eval_)} rows")
print(f"  pool_unlabeled: {len(pool)} rows")

# Load PatentSBERTa (frozen - no fine-tuning here)
# Use MPS (Apple Silicon GPU) when available for faster encoding
import torch
import torch
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"\nLoading PatentSBERTa (device={device})...")
model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa', device=device)

# batch_size=32 is safer on MPS to avoid memory pressure
BATCH_SIZE = 32

# Extract embeddings (mirrors M4_PatentSBERTa_For_PatentSearch.ipynb pattern)
print("Encoding train_silver embeddings (this takes a few minutes)...")
train_emb = model.encode(
    train['text'].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print("Encoding eval_silver embeddings...")
eval_emb = model.encode(
    eval_['text'].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print("Encoding pool_unlabeled embeddings...")
pool_emb = model.encode(
    pool['text'].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print(f"\nEmbedding shapes:")
print(f"  train: {train_emb.shape}")
print(f"  eval:  {eval_emb.shape}")
print(f"  pool:  {pool_emb.shape}")

# Train Logistic Regression on frozen embeddings
print("\nTraining Logistic Regression classifier...")
clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
clf.fit(train_emb, train['is_green_silver'])

# Evaluate on eval_silver
print("\nEvaluation on eval_silver:")
y_pred = clf.predict(eval_emb)
y_prob = clf.predict_proba(eval_emb)[:, 1]

print(classification_report(
    eval_['is_green_silver'],
    y_pred,
    target_names=['not_green', 'green'],
))

precision, recall, f1, _ = precision_recall_fscore_support(
    eval_['is_green_silver'], y_pred, average='binary'
)
print(f"Summary: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# Save model and embeddings for reuse in Steps 02 and 04
print("\nSaving artifacts...")
pickle.dump(clf, open(os.path.join(OUTPUT_DIR, 'baseline_clf.pkl'), 'wb'))
np.save(os.path.join(OUTPUT_DIR, 'train_emb.npy'), train_emb)
np.save(os.path.join(OUTPUT_DIR, 'eval_emb.npy'),  eval_emb)
np.save(os.path.join(OUTPUT_DIR, 'pool_emb.npy'),  pool_emb)
print("Saved: baseline_clf.pkl, train_emb.npy, eval_emb.npy, pool_emb.npy")
print(f"\nBaseline F1 (eval_silver): {f1:.4f}")
