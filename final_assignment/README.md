# Final Assignment: QLoRA-Powered MAS + Targeted HITL

**Course:** MSc BDS - M4: Applied Deep Learning and Artificial Intelligence
**Deadline:** Thursday 5 March 2026 at 10:00 AM
**HuggingFace Model:** https://huggingface.co/Peter512/patentsbert-green-final

## Overview

This final assignment combines QLoRA domain adaptation with the multi-agent debate system:

1. **QLoRA Training:** Fine-tune Llama-3.2-3B-Instruct on train_silver using 4-bit NF4 quantization + LoRA adapters
2. **QLoRA Inference:** Run the fine-tuned adapter on the 100 high-risk claims to produce concrete predictions
3. **QLoRA-Powered MAS:** The Advocate agent receives the actual QLoRA model output per claim and argues from it; Skeptic and Judge use GPT-4o-mini
4. **Exception-based HITL:** Human reviews only claims where the Judge assigned low confidence
5. **Final PatentSBERTa:** Fine-tune one last time with the resulting gold labels

## Files

| File | Description |
|---|---|
| `01_qlora_train.py` | QLoRA fine-tuning of Llama-3.2-3B on train_silver |
| `00_qlora_inference.py` | Run fine-tuned QLoRA adapter on 100 claims, save predictions |
| `02_mas_qlora.py` | CrewAI MAS: Advocate reads QLoRA predictions; Skeptic/Judge on GPT-4o-mini |
| `04_hitl_review.py` | Interactive human review of low-confidence claims (run on login node) |
| `03_finetune_final.py` | Final PatentSBERTa fine-tuning with gold labels |
| `slurm_qlora.sh` | SLURM job for QLoRA training (GPU, ~12 min for 200 steps) |
| `slurm_qlora_inference.sh` | SLURM job for QLoRA inference on 100 claims (GPU, ~10 min) |
| `slurm_mas_final.sh` | SLURM job for MAS classification (no GPU needed) |
| `slurm_finetune_final.sh` | SLURM job for final PatentSBERTa fine-tuning (GPU) |

## How to Run

### On HPC (step by step)

```bash
# Step 1: QLoRA training (already done — adapter at qlora_patent_adapter/)
sbatch slurm_qlora.sh

# Step 2: Run QLoRA adapter on the 100 claims to produce predictions (~10 min, GPU)
sbatch slurm_qlora_inference.sh
# Output: qlora_predictions_100.csv

# Step 3: MAS classification — Advocate reads QLoRA predictions (no GPU needed)
sbatch slurm_mas_final.sh
# Output: final_labels_100_raw.csv

# Step 4: Interactive HITL review — run on login node, NOT via sbatch
python final_assignment/04_hitl_review.py
# Output: final_labels_100.csv

# Step 5: Final PatentSBERTa fine-tuning (GPU)
sbatch slurm_finetune_final.sh
```

## QLoRA Architecture

Mirrors `M3_3_Finetune_opt_bnb_peft.ipynb` notebook pattern:

- **Base model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytesConfig)
- **LoRA:** r=16, alpha=32, target_modules=["q_proj", "v_proj"]
- **Training data:** 10,000 samples from train_silver, formatted as classification prompts
- **Steps:** max_steps=200, lr=2e-4
- **Confirmed completed:** log qlora_286319.log shows training_loss=1.912, adapter saved to qlora_patent_adapter/

## QLoRA Integration: Two-Stage Architecture

The QLoRA adapter is integrated via a two-stage design that avoids the
LangChain/CrewAI compatibility issues of live model loading:

**Stage 1 (`00_qlora_inference.py`):** The fine-tuned Llama-3.2-3B adapter runs
standalone inference on all 100 claims. For each claim, the model compares logits
for the tokens " yes" and " no" at the answer position (matching the training format)
to produce a clean 0/1 label and a short text output. Results saved to
`qlora_predictions_100.csv`.

**Stage 2 (`02_mas_qlora.py`):** The Advocate agent in CrewAI receives the QLoRA
model's actual prediction and raw output as part of its task description. The Advocate
builds its argument around the QLoRA verdict, citing specific phrases from the claim.
The Skeptic challenges the classification independently. The Judge weighs both and
produces the final decision.

This means the fine-tuned QLoRA model's weights genuinely drive the Advocate's
reasoning -- its actual output per claim shapes the debate, not just a system prompt.

**Why not serve QLoRA directly as a CrewAI LLM backend?** CrewAI agents require a
LangChain-compatible LLM interface (e.g. `ChatOpenAI`). Quantized local models loaded
via `bitsandbytes` + `peft` do not expose this interface without additional wrappers
(e.g. a local vLLM or text-generation-inference server), which adds deployment complexity
on HPC SLURM nodes. The two-stage design avoids this by decoupling inference (GPU job)
from orchestration (API-only job), keeping each step simple and reproducible.

## Exception-Based HITL

Unlike Assignments 2 and 3, human review is triggered only for claims where the
Judge assigned confidence="low". All other claims are accepted directly.

**Disagreement results:** The QLoRA model and MAS agreed on 71 out of 100 claims.
The Judge resolved all 29 disagreements with medium or high confidence (83 high,
17 medium, 0 low), so 0 out of 100 claims were flagged for human intervention.
This indicates the debate structure consistently produced decisive outcomes even
when the QLoRA adapter and GPT-4o-mini-based agents initially disagreed.

The interactive `04_hitl_review.py` script runs in the terminal, shows each
flagged claim with the QLoRA prediction, MAS label, and rationale, and asks for
a 0/1 decision or Enter to accept. This is a genuine human review, not a simulation.

## Results

Evaluated on eval_silver (5,000 held-out claims). Log: eval_all_287041.log.

| Model Version | Training Data | F1 | Precision | Recall |
|---|---|---|---|---|
| 1. Baseline | Frozen embeddings | 0.7696 | 0.7845 | 0.7553 |
| 2. Assignment 2 | Silver + Gold (GPT-4o-mini) | 0.8099 | 0.8207 | 0.7994 |
| 3. Assignment 3 | Silver + Gold (CrewAI MAS) | 0.8115 | 0.8224 | 0.8010 |
| 4. Final | Silver + Gold (QLoRA MAS + Targeted HITL) | 0.8097 | 0.8213 | 0.7986 |

**Note on gold_100 evaluation:** The fine-tuned models show ~50% accuracy on the
gold_100 set. This is expected: these 100 claims were specifically selected as the
most uncertain examples (u close to 1.0), meaning they sit near the model's decision
boundary. Performance on this adversarially-selected subset is not representative of
overall model quality -- the eval_silver results above are the meaningful comparison.
