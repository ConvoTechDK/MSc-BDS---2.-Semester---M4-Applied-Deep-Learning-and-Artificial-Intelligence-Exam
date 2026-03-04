"""
Push model cards to all 3 HuggingFace model repos + dataset card.

Run locally (requires HF token with write access):
    export HF_TOKEN=hf_...
    python push_model_cards.py
"""

import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])

# ── Assignment 2 ─────────────────────────────────────────────────────
a2_card = """---
language: en
license: apache-2.0
tags:
- patent
- green-technology
- text-classification
- patentsbert
- active-learning
- hitl
datasets:
- Peter512/patents-50k-green
base_model: AI-Growth-Lab/PatentSBERTa
---

# PatentSBERTa Green Patent Classifier — Assignment 2

Binary classifier for green patent detection (Y02 CPC codes).
Fine-tuned from [AI-Growth-Lab/PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
using active learning + GPT-4o-mini HITL gold labels.

## Training

- **Base model:** AI-Growth-Lab/PatentSBERTa (MPNet-based)
- **Task:** Binary classification — `is_green` (Y02 CPC codes)
- **Training data:** 35,000 silver labels (CPC Y02*) + 100 gold labels (GPT-4o-mini HITL)
- **HITL process:** Uncertainty sampling selected 100 most uncertain pool claims;
  GPT-4o-mini labeled each with confidence + rationale; human simulated review with
  90% agreement (10 overrides on low-confidence cases)
- **Fine-tuning:** 1 epoch, lr=2e-5, max_length=256, batch_size=16, fp16

## Evaluation (eval_silver, 5,000 claims)

| Metric | Value |
|---|---|
| F1 | **0.8099** |
| Precision | 0.8207 |
| Recall | 0.7994 |
| Accuracy | 0.8126 |

Baseline (frozen PatentSBERTa + Logistic Regression): F1=0.7696

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("Peter512/patentsbert-green-a2")
model.eval()

text = "A photovoltaic cell comprising a perovskite absorber layer..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits
label = logits.argmax().item()  # 0=not_green, 1=green
```

## Dataset

- **Source:** AI-Growth-Lab/patents_claims_1.5m_traim_test
- **Silver labels:** CPC Y02* codes (is_green_silver)
- **Splits:** train_silver (35k), eval_silver (5k), pool_unlabeled (10k)
- **Balance:** 50/50 green/not-green
"""

# ── Assignment 3 ─────────────────────────────────────────────────────
a3_card = """---
language: en
license: apache-2.0
tags:
- patent
- green-technology
- text-classification
- patentsbert
- multi-agent
- crewai
datasets:
- Peter512/patents-50k-green
base_model: AI-Growth-Lab/PatentSBERTa
---

# PatentSBERTa Green Patent Classifier — Assignment 3

Binary classifier for green patent detection (Y02 CPC codes).
Fine-tuned from [AI-Growth-Lab/PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
using a 3-agent CrewAI debate system (Advocate / Skeptic / Judge).

## Training

- **Base model:** AI-Growth-Lab/PatentSBERTa (MPNet-based)
- **Task:** Binary classification — `is_green` (Y02 CPC codes)
- **Training data:** 35,000 silver labels + 100 gold labels (CrewAI MAS)
- **MAS process:** 3-agent debate — Advocate argues green, Skeptic challenges,
  Judge produces `{"label": 0/1, "confidence": "low/medium/high", "rationale": "..."}`.
  100% agent agreement (0 human overrides — no low-confidence outputs).
- **Fine-tuning:** 1 epoch, lr=2e-5, max_length=256, batch_size=16, fp16

## Evaluation (eval_silver, 5,000 claims)

| Metric | Value |
|---|---|
| F1 | **0.8115** |
| Precision | 0.8224 |
| Recall | 0.8010 |
| Accuracy | 0.8142 |

Assignment 2 baseline: F1=0.8099 | Original baseline: F1=0.7696

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("Peter512/patentsbert-green-a3")
model.eval()

text = "A photovoltaic cell comprising a perovskite absorber layer..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits
label = logits.argmax().item()  # 0=not_green, 1=green
```
"""

# ── Final Assignment ──────────────────────────────────────────────────
final_card = """---
language: en
license: apache-2.0
tags:
- patent
- green-technology
- text-classification
- patentsbert
- qlora
- multi-agent
- crewai
datasets:
- Peter512/patents-50k-green
base_model: AI-Growth-Lab/PatentSBERTa
---

# PatentSBERTa Green Patent Classifier — Final Assignment

Binary classifier for green patent detection (Y02 CPC codes).
Fine-tuned from [AI-Growth-Lab/PatentSBERTa](https://huggingface.co/AI-Growth-Lab/PatentSBERTa)
using a QLoRA-powered multi-agent system with exception-based HITL.

## Training

- **Base model:** AI-Growth-Lab/PatentSBERTa (MPNet-based)
- **Task:** Binary classification — `is_green` (Y02 CPC codes)
- **Training data:** 35,000 silver labels + 100 gold labels (QLoRA MAS)
- **Pipeline:**
  1. QLoRA fine-tuning of Llama-3.2-3B-Instruct (4-bit NF4, LoRA r=16, 200 steps)
     on 10,000 patent classification prompts from train_silver
  2. 3-agent CrewAI MAS with QLoRA-informed Advocate; exception-based HITL
     (only low-confidence claims reviewed — 0 triggered out of 100)
  3. PatentSBERTa fine-tuned on resulting gold labels
- **Fine-tuning:** 1 epoch, lr=2e-5, max_length=256, batch_size=16, fp16

## Evaluation (eval_silver, 5,000 claims)

| Metric | Value |
|---|---|
| F1 | **0.8097** |
| Precision | 0.8213 |
| Recall | 0.7986 |
| Accuracy | 0.8126 |

Progression: Baseline F1=0.7696 → A2=0.8099 → A3=0.8115 → Final=0.8097

## Notes

The QLoRA adapter (Llama-3.2-3B-Instruct) was trained on patent classification prompts
and its learned domain knowledge was encoded into the Advocate agent's system prompt.
The slight regression from A3 to Final is within noise and reflects the 100-claim gold
set being a small fraction of the 35k silver training data.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("AI-Growth-Lab/PatentSBERTa", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("Peter512/patentsbert-green-final")
model.eval()

text = "A photovoltaic cell comprising a perovskite absorber layer..."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits
label = logits.argmax().item()  # 0=not_green, 1=green
```
"""

dataset_card = """---
language: en
license: apache-2.0
tags:
- patents
- green-technology
- text-classification
- active-learning
pretty_name: Green Patent Claims (50k)
size_categories:
- 10K<n<100K
---

# Green Patent Claims — 50k Balanced Dataset

Binary classification dataset for green patent detection based on CPC Y02 codes.
Used across MSc BDS M4 Applied Deep Learning assignments.

## Dataset Description

- **Source:** Derived from [AI-Growth-Lab/patents_claims_1.5m_train_test](https://huggingface.co/datasets/AI-Growth-Lab/patents_claims_1.5m_train_test)
- **Size:** 50,000 patent claims
- **Balance:** 50/50 green (Y02) / not-green
- **Format:** Parquet

## Columns

| Column | Type | Description |
|---|---|---|
| `doc_id` | int64 | Patent document identifier |
| `text` | string | Full patent claim text |
| `is_green_silver` | int64 | Binary label: 1=green (Y02 CPC code), 0=not green |
| `split` | string | Data split: train_silver (35k), eval_silver (5k), pool_unlabeled (10k) |

## Splits

| Split | Size | Purpose |
|---|---|---|
| train_silver | 35,000 | Training data with CPC-derived silver labels |
| eval_silver | 5,000 | Held-out evaluation set (used for all model comparisons) |
| pool_unlabeled | 10,000 | Uncertainty sampling pool for active learning |

## Usage

```python
from datasets import load_dataset
import pandas as pd

# Load full dataset
df = pd.read_parquet("hf://datasets/Peter512/patents-50k-green/patents_50k_green.parquet")

# Get training split
train = df[df["split"] == "train_silver"]
eval_set = df[df["split"] == "eval_silver"]
```

## Models Trained on This Dataset

- [Peter512/patentsbert-green-a2](https://huggingface.co/Peter512/patentsbert-green-a2) — F1=0.8099 (GPT-4o-mini HITL)
- [Peter512/patentsbert-green-a3](https://huggingface.co/Peter512/patentsbert-green-a3) — F1=0.8115 (CrewAI MAS)
- [Peter512/patentsbert-green-final](https://huggingface.co/Peter512/patentsbert-green-final) — F1=0.8097 (QLoRA MAS + Targeted HITL)
"""

cards = [
    ("Peter512/patentsbert-green-a2",    a2_card,       "model"),
    ("Peter512/patentsbert-green-a3",    a3_card,       "model"),
    ("Peter512/patentsbert-green-final", final_card,    "model"),
    ("Peter512/patents-50k-green",       dataset_card,  "dataset"),
]

for repo_id, card_content, repo_type in cards:
    print(f"Pushing README to {repo_id} ({repo_type})...")
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="Update model card with correct eval metrics",
    )
    print(f"  Done: https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")

print("\nAll cards pushed successfully.")
