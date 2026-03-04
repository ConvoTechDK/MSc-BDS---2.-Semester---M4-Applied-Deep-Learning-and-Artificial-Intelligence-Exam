# Assignment 2: Green Patent Detection — Active Learning + LLM->Human HITL

**Course:** MSc BDS - M4: Applied Deep Learning and Artificial Intelligence
**Deadline:** Monday 16 February 2026
**HuggingFace Model:** https://huggingface.co/Peter512/patentsbert-green-a2

## Overview

This assignment implements a full active learning pipeline for green patent detection:

1. **Baseline:** Frozen PatentSBERTa embeddings + Logistic Regression
2. **Uncertainty Sampling:** Select 100 most uncertain claims for human review
3. **HITL:** GPT-4o-mini labels each claim; human simulates final review
4. **Fine-tuning:** PatentSBERTa fine-tuned on silver + gold labels

## Files

| File | Description |
|---|---|
| `00_create_dataset.py` | Download and create `patents_50k_green.parquet` (25k green + 25k not-green) |
| `01_baseline.py` | Extract PatentSBERTa embeddings, train Logistic Regression, evaluate |
| `02_uncertainty_sampling.py` | Compute u = 1 - 2*|p - 0.5|, export top-100 to `hitl_green_100.csv` |
| `03_hitl_llm.py` | GPT-4o-mini labels + human review simulation |
| `04_finetune_patentsbert.py` | Fine-tune PatentSBERTa with HuggingFace Trainer |
| `slurm_finetune_a2.sh` | SLURM job script for AAU HPC |

## How to Run

### Local (Steps 00-03)

```bash
pip install -r ../requirements.txt

# Step 1: Create dataset (downloads from HuggingFace, may take time)
python 00_create_dataset.py

# Step 2: Baseline model
python 01_baseline.py

# Step 3: Uncertainty sampling
python 02_uncertainty_sampling.py

# Step 4: HITL labeling (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python 03_hitl_llm.py
```

### On HPC (Step 04 - Fine-tuning)

```bash
sbatch slurm_finetune_a2.sh
```

## Dataset

- **Source:** `AI-Growth-Lab/patents_claims_1.5m_traim_test`
- **Processed dataset:** https://huggingface.co/datasets/Peter512/patents-50k-green
- **Model:** `AI-Growth-Lab/PatentSBERTa`
- **Silver labels:** CPC Y02* codes (is_green_silver)
- **Splits:** train_silver (35k), eval_silver (5k), pool_unlabeled (10k)

## Two Models — Key Distinction

- **GPT-4o-mini:** Used as the HITL labeler (inference only, no fine-tuning)
- **PatentSBERTa:** The open-source model that gets fine-tuned on the gold labels

## LLM -> Human HITL Results

The GPT-4o-mini HITL step labeled all 100 claims with suggested label, confidence, and rationale.

**Note on human review simulation:** The "human step" in `03_hitl_llm.py` is a scripted
simulation: the code automatically overrides the LLM's label for any claim rated low-confidence.
This demonstrates the HITL workflow structure (LLM suggests, human decides) without requiring a
live annotator. In a real deployment, low-confidence rows would be surfaced via a review UI
(e.g. Argilla or Label Studio) where a domain expert reads the claim text and sets the final
label. The simulation flips low-confidence labels to show how disagreement between
the LLM suggestion and human judgment would be recorded.

The simulation overrode 4 claims (those rated low-confidence by GPT-4o-mini) and accepted
96 LLM suggestions without change.

**Override examples (simulated -- all 4 were low-confidence LLM=0 flipped to human=1):**

1. doc_id=9647788: A system with memory and processor for executing component instructions.
   The LLM rated it not-green with low confidence. The simulation flipped to green (label=1).

2. doc_id=8650629: An apparatus comprising a system-on-a-chip (SoC). The LLM found no green
   technology indicators and rated it not-green with low confidence. Flipped to green (label=1).

3. doc_id=9529419: A switching element with interfaces, processing pipeline, and packet buffer.
   The LLM rated it not-green with low confidence, citing data handling rather than green tech.
   Flipped to green (label=1).

Note: because the simulation blindly flips all low-confidence labels, these overrides may
introduce noise rather than corrections. In a real deployment, a domain expert would read
each claim and make an informed decision.

**Agreement rate:** 96% (96/100 claims accepted without override)

## Results

Exact metrics from post-hoc evaluation on eval_silver (5,000 claims), loading the published
HuggingFace models and running inference with the base PatentSBERTa tokenizer (log: eval_all_286597.log).

| Model | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Baseline (Frozen LR) | 0.7696 | 0.7845 | 0.7553 | 0.7742 |
| Fine-tuned PatentSBERTa (A2) | 0.8099 | 0.8207 | 0.7994 | 0.8126 |
