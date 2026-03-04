"""
Assignment 2 - Step 00: Create the patents_50k_green.parquet dataset.

Downloads the AI-Growth-Lab/patents_claims_1.5m_traim_test dataset from HuggingFace,
creates a balanced 50k sample (25k green, 25k not-green) using CPC Y02* codes as
silver labels, and saves it to parquet with train/eval/pool splits.

Run: python 00_create_dataset.py
Output: patents_50k_green.parquet (in project root, shared by all assignments)
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset

# Output path (one level up, shared across assignments)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'patents_50k_green.parquet')

print("Loading dataset from HuggingFace: AI-Growth-Lab/patents_claims_1.5m_traim_test")
print("Using selective column loading to stay within RAM limits (only id, text, Y02* cols)...")

# Load the dataset using HuggingFace's Arrow-backed format, then slim it down
# BEFORE converting to pandas to avoid out-of-memory on machines with limited RAM.
ds = load_dataset("AI-Growth-Lab/patents_claims_1.5m_traim_test", split="train")

print(f"Dataset loaded: {len(ds)} rows, {len(ds.column_names)} columns")

# Identify Y02* columns from the dataset schema (no pandas conversion yet)
y02_cols = [c for c in ds.column_names if c.startswith('Y02')]
print(f"\nY02 columns found: {len(y02_cols)}")
print(f"Examples: {y02_cols[:5]}")

# Select only the columns we need, then convert to pandas (much smaller in RAM)
keep_cols = ['id', 'text'] + y02_cols
print(f"\nSelecting {len(keep_cols)} columns before converting to pandas...")
df = ds.select_columns(keep_cols).to_pandas()
print(f"Slim DataFrame shape: {df.shape}")

# Create silver label: 1 if any Y02* column is 1
df['is_green_silver'] = (df[y02_cols].sum(axis=1) > 0).astype(int)
print(f"\nGreen distribution:")
print(df['is_green_silver'].value_counts())

# Balanced 50k: 25k green, 25k not-green
print("\nSampling 25k green and 25k not-green...")
green     = df[df['is_green_silver'] == 1].sample(25000, random_state=42)
not_green = df[df['is_green_silver'] == 0].sample(25000, random_state=42)
df50k = pd.concat([green, not_green]).sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only the columns needed downstream
df50k = df50k[['id', 'text', 'is_green_silver']].rename(columns={'id': 'doc_id'})

# Create splits:
#   train_silver:   first 35k - used for training the baseline and fine-tuned models
#   eval_silver:    next 5k   - held-out evaluation set
#   pool_unlabeled: last 10k  - pool for uncertainty sampling / active learning
df50k['split'] = 'train_silver'
df50k.loc[35000:39999, 'split'] = 'eval_silver'
df50k.loc[40000:,      'split'] = 'pool_unlabeled'

print("\nSplit distribution:")
print(df50k['split'].value_counts())
print(f"\nLabel distribution within each split:")
print(df50k.groupby('split')['is_green_silver'].mean().round(3))

# Save to parquet
output_abs = os.path.abspath(OUTPUT_PATH)
df50k.to_parquet(output_abs, index=False)
print(f"\nSaved to: {output_abs}")
print(f"File size: {os.path.getsize(output_abs) / 1e6:.1f} MB")
print(f"Shape: {df50k.shape}")
