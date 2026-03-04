"""
Assignment 2 - Step 02: Uncertainty Sampling (Identify High-Risk Examples).

Uses the baseline classifier to predict p_green on pool_unlabeled,
computes uncertainty scores u = 1 - 2*|p - 0.5|, and exports the top 100
most uncertain claims to hitl_green_100.csv for human-in-the-loop review.

Run: python 02_uncertainty_sampling.py (requires 01_baseline.py to have run first)
Output: hitl_green_100.csv
"""

import os
import pickle
import numpy as np
import pandas as pd

PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'patents_50k_green.parquet')
OUTPUT_DIR   = os.path.dirname(__file__)

print("Loading dataset and baseline model...")
df  = pd.read_parquet(PARQUET_PATH)
clf = pickle.load(open(os.path.join(OUTPUT_DIR, 'baseline_clf.pkl'), 'rb'))

# Load precomputed pool embeddings from Step 01
pool_emb = np.load(os.path.join(OUTPUT_DIR, 'pool_emb.npy'))
pool = df[df['split'] == 'pool_unlabeled'].reset_index(drop=True)

print(f"Pool size: {len(pool)} rows")
print(f"Pool embeddings shape: {pool_emb.shape}")

# Compute predicted probability of being green
print("\nComputing p_green for pool_unlabeled...")
p_green = clf.predict_proba(pool_emb)[:, 1]

# Uncertainty score: u = 1 - 2*|p - 0.5|
# u = 1 means maximally uncertain (p = 0.5)
# u ~ 0 means confident (p ~ 0 or p ~ 1)
u = 1.0 - 2.0 * np.abs(p_green - 0.5)

print(f"p_green stats: min={p_green.min():.4f}, max={p_green.max():.4f}, mean={p_green.mean():.4f}")
print(f"u score stats: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")

# Add scores to pool dataframe
pool = pool.copy()
pool['p_green'] = p_green
pool['u'] = u

# Select top 100 highest-uncertainty examples
top100 = pool.nlargest(100, 'u').reset_index(drop=True)

print(f"\nTop 100 uncertainty stats:")
print(f"  u range: [{top100['u'].min():.4f}, {top100['u'].max():.4f}]")
print(f"  p_green range: [{top100['p_green'].min():.4f}, {top100['p_green'].max():.4f}]")
print(f"  Silver label distribution: {top100['is_green_silver'].value_counts().to_dict()}")

# Export CSV with empty labeling columns for HITL step
output_cols = [
    'doc_id', 'text', 'p_green', 'u',
    'llm_green_suggested', 'llm_confidence', 'llm_rationale',
    'is_green_human', 'human_notes',
]
top100['llm_green_suggested'] = ''
top100['llm_confidence']      = ''
top100['llm_rationale']       = ''
top100['is_green_human']      = ''
top100['human_notes']         = ''

output_path = os.path.join(OUTPUT_DIR, 'hitl_green_100.csv')
top100[output_cols].to_csv(output_path, index=False)
print(f"\nExported: {output_path}")
print(f"Shape: {top100.shape}")
print("\nFirst 5 rows preview:")
print(top100[['doc_id', 'p_green', 'u']].head().to_string())
