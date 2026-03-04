# M4: Applied Deep Learning and Artificial Intelligence — Exam Portfolio

MSc BDS, Aalborg University Business School

## Overview

This portfolio implements an end-to-end green patent detection pipeline, progressively improving label quality through active learning, human-in-the-loop (HITL) validation, multi-agent debate, and QLoRA domain adaptation.

The central artifact is a fine-tuned PatentSBERTa binary classifier (green vs. not-green) evaluated at four stages:

| Model | Training Data | F1 | Precision | Recall |
|---|---|---|---|---|
| 1. Baseline | Frozen PatentSBERTa + Logistic Regression | 0.7696 | 0.7845 | 0.7553 |
| 2. Assignment 2 | Silver + Gold (GPT-4o-mini HITL, 100 claims) | 0.8099 | 0.8207 | 0.7994 |
| 3. Assignment 3 | Silver + Gold (CrewAI MAS, 100 claims) | 0.8115 | 0.8224 | 0.8010 |
| 4. Final | Silver + Gold (QLoRA MAS + Targeted HITL, 100 claims) | 0.8097 | 0.8213 | 0.7986 |

All models evaluated on `eval_silver` (5,000 held-out claims). Evaluation script: `eval_all_models.py`.

## Repository Structure

```
.
├── assignment_1/                  # Manual SGD + Self-Attention (notebook)
│   ├── assignment_1.ipynb         # Main notebook
│   ├── attention_shift.png        # Generated embedding-shift visualization
│   └── README.md
├── assignment_2/                  # Baseline + Uncertainty Sampling + GPT-4o-mini HITL + Fine-tuning
│   ├── 00_create_dataset.py       # Download and create patents_50k_green.parquet
│   ├── 01_baseline.py             # Frozen PatentSBERTa + Logistic Regression
│   ├── 02_uncertainty_sampling.py # Select top-100 uncertain claims, export hitl_green_100.csv
│   ├── 03_hitl_llm.py             # GPT-4o-mini labeling + human review simulation
│   ├── 04_finetune_patentsbert.py # Fine-tune PatentSBERTa on silver + gold labels
│   ├── hitl_green_100.csv         # 100 uncertain claims (before labeling)
│   ├── hitl_green_100_labeled.csv # 100 claims with GPT-4o-mini + human labels
│   ├── baseline_clf.pkl           # Serialized Logistic Regression baseline
│   ├── train_emb.npy              # Precomputed PatentSBERTa train embeddings
│   ├── eval_emb.npy               # Precomputed PatentSBERTa eval embeddings
│   ├── pool_emb.npy               # Precomputed PatentSBERTa pool embeddings
│   ├── slurm_baseline.sh          # SLURM job for baseline embedding extraction
│   ├── slurm_uncertainty_hitl.sh  # SLURM job for uncertainty sampling + HITL
│   ├── slurm_finetune_a2.sh       # SLURM job for fine-tuning (GPU)
│   └── README.md
├── assignment_3/                  # 3-Agent CrewAI Debate System + Fine-tuning
│   ├── 03_mas_crewai.py           # Advocate / Skeptic / Judge debate on 100 claims
│   ├── 04_finetune_v2.py          # Fine-tune PatentSBERTa with MAS gold labels
│   ├── mas_labels_100.csv         # CrewAI debate output (100 labeled claims)
│   ├── slurm_mas_a3.sh            # SLURM job for MAS (no GPU)
│   ├── slurm_finetune_a3.sh       # SLURM job for fine-tuning (GPU)
│   └── README.md
├── final_assignment/              # QLoRA Training + QLoRA-Powered MAS + Targeted HITL + Final Fine-tuning
│   ├── 01_qlora_train.py          # QLoRA fine-tuning of Llama-3.2-3B-Instruct (4-bit NF4, LoRA r=16)
│   ├── 00_qlora_inference.py      # Run QLoRA adapter on 100 claims; compare yes/no logits
│   ├── 02_mas_qlora.py            # CrewAI MAS: Advocate informed by QLoRA predictions
│   ├── 04_hitl_review.py          # Interactive exception-based HITL (low-confidence claims only)
│   ├── 03_finetune_final.py       # Final PatentSBERTa fine-tuning with QLoRA MAS gold labels
│   ├── qlora_predictions_100.csv  # QLoRA model predictions on 100 claims
│   ├── final_labels_100_raw.csv   # Raw MAS output before HITL review
│   ├── final_labels_100.csv       # Final gold labels after HITL (is_green_gold column)
│   ├── slurm_qlora.sh             # SLURM job for QLoRA training (GPU, ~12 min)
│   ├── slurm_qlora_inference.sh   # SLURM job for QLoRA inference (GPU, ~10 min)
│   ├── slurm_mas_final.sh         # SLURM job for QLoRA-powered MAS (no GPU)
│   ├── slurm_finetune_final.sh    # SLURM job for final fine-tuning (GPU)
│   └── README.md
├── report/                        # 2-page report
│   ├── report.md                  # Source
│   └── report.pdf                 # Compiled PDF (max 2 pages)
├── logs/                          # SLURM execution logs (26 files documenting full pipeline)
├── other_files/                   # Supplementary scripts (not part of submission pipeline)
│   ├── eval_all_models.py         # Post-hoc evaluation of all 4 models on eval_silver
│   ├── slurm_eval.sh              # SLURM job for eval_all_models.py
│   └── push_model_cards.py        # Push model cards to HuggingFace Hub
├── patents_50k_green.parquet      # Dataset (25k green + 25k not-green, 50k total)
└── requirements.txt               # Python dependencies
```

## Assignments

### Assignment 1: SGD Mechanics + Self-Attention
`assignment_1/assignment_1.ipynb`

- **Part A:** Manual SGD on the insurance dataset — forward pass, loss, gradient, weight update for 3 samples, verified with numpy
- **Part B:** Scaled dot-product self-attention from scratch demonstrating that "bank" gets different contextualized representations in financial vs. geographic contexts

### Assignment 2: Green Patent Detection — Active Learning + HITL
`assignment_2/`

1. `01_baseline.py` — Frozen PatentSBERTa embeddings + Logistic Regression (F1=0.7696)
2. `02_uncertainty_sampling.py` — Select top-100 most uncertain pool claims (`u = 1 - 2*|p - 0.5|`)
3. `03_hitl_llm.py` — GPT-4o-mini labels 100 claims; human simulation overrides 4 low-confidence cases
4. `04_finetune_patentsbert.py` — PatentSBERTa fine-tuned on silver + gold (F1=0.8099)

### Assignment 3: Advanced Architecture — Multi-Agent System
`assignment_3/`

- 3-agent CrewAI debate: Advocate (argues green) / Skeptic (challenges) / Judge (final JSON decision)
- Relabels the same 100 high-risk claims from Assignment 2
- PatentSBERTa fine-tuned with MAS gold labels (F1=0.8115)

### Final Assignment: QLoRA-Powered MAS + Targeted HITL
`final_assignment/`

1. `01_qlora_train.py` — QLoRA fine-tuning of Llama-3.2-3B-Instruct (4-bit NF4, LoRA r=16, 200 steps)
2. `00_qlora_inference.py` — Run QLoRA adapter on 100 claims; compare yes/no logits for clean 0/1 labels
3. `02_mas_qlora.py` — CrewAI MAS with QLoRA-informed Advocate; Skeptic + Judge on GPT-4o-mini
4. `04_hitl_review.py` — Interactive exception-based HITL (only low-confidence claims; 0 triggered)
5. `03_finetune_final.py` — Final PatentSBERTa fine-tuned with QLoRA MAS gold labels (F1=0.8097)

## HuggingFace

- Dataset: https://huggingface.co/datasets/Peter512/patents-50k-green
- Assignment 2 model: https://huggingface.co/Peter512/patentsbert-green-a2
- Assignment 3 model: https://huggingface.co/Peter512/patentsbert-green-a3
- Final model: https://huggingface.co/Peter512/patentsbert-green-final

## Setup

```bash
pip install -r requirements.txt
```

GPU jobs (fine-tuning, QLoRA) are submitted via the SLURM scripts included in each assignment folder. The `patents_50k_green.parquet` dataset is included in this repository; it can also be loaded from HuggingFace:

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/Peter512/patents-50k-green/patents_50k_green.parquet")
```
