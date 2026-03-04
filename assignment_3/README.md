# Assignment 3: Green Patent Detection — Advanced Architecture (Multi-Agent System)

**Course:** MSc BDS - M4: Applied Deep Learning and Artificial Intelligence
**Deadline:** Monday 23 February 2026
**HuggingFace Model:** https://huggingface.co/Peter512/patentsbert-green-a3

## Overview

This assignment replaces the simple GPT-4o-mini HITL from Assignment 2 with a structured
3-agent CrewAI debate system:

- **Agent 1 (Advocate):** Argues FOR green classification (Y02)
- **Agent 2 (Skeptic):** Argues AGAINST (looks for greenwashing)
- **Agent 3 (Judge):** Weighs both arguments and produces a final JSON decision

The same 100 high-risk claims from Assignment 2 are relabeled using this MAS,
and PatentSBERTa is fine-tuned again with the resulting gold labels.

**Path chosen:** Multi-Agent System (CrewAI) — mirrors the M3_CrewAI_Tutorial_v3 notebook.
QLoRA is reserved for the Final Assignment where it is mandatory, creating a clean progression.

## Files

| File | Description |
|---|---|
| `03_mas_crewai.py` | 3-agent CrewAI debate system for 100 claims |
| `04_finetune_v2.py` | Fine-tune PatentSBERTa with MAS gold labels |
| `slurm_mas_a3.sh` | SLURM job for MAS (API calls, no GPU) |
| `slurm_finetune_a3.sh` | SLURM job for fine-tuning (GPU required) |

## How to Run

```bash
pip install -r ../requirements.txt

# Run MAS classification (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python 03_mas_crewai.py

# Run fine-tuning (GPU recommended, or submit to HPC)
python 04_finetune_v2.py
# OR: sbatch slurm_finetune_a3.sh
```

## CrewAI Agent Design

The system mirrors the M3_CrewAI_Tutorial_v3 notebook structure exactly:

```python
PatentAgents class -> advocate_agent(), skeptic_agent(), judge_agent()
PatentTasks class  -> advocate_task(), skeptic_task(), judge_task()
Crew(agents=[...], tasks=[...], verbose=False)
```

Each claim triggers a sequential debate:
1. Advocate identifies Y02-relevant phrases
2. Skeptic challenges the classification
3. Judge outputs: {"label": 0/1, "confidence": "low/medium/high", "rationale": "..."}

## Agreement Analysis

**Note on human review simulation:** As in Assignment 2, the human review step is
simulated in code: `04_finetune_v2.py` accepts the MAS output as the gold label unless
the Judge confidence is "low" (in which case the code would flip or flag the label for
human inspection). This structure mirrors a real HITL workflow where a domain expert
only reviews the cases the system is uncertain about. In production, flagged claims
would be routed to an annotator via a review interface.

**Assignment 2 (GPT-4o-mini HITL):** 96% agreement (4 human overrides out of 100)

**Assignment 3 (CrewAI MAS):** 100% agreement (0 human overrides out of 100)

The CrewAI MAS produced no low-confidence decisions across all 100 claims: 88 were
rated "medium" and 12 were rated "high" confidence. Because the simulated human override
only triggers on low-confidence cases, no overrides occurred. This is a consequence of
the structured 3-agent debate consistently resolving ambiguous cases, rather than
returning an uncertain single-LLM prediction. All 100 gold labels therefore equal the
Judge's MAS output directly (is_green_human_a3 == mas_label for all rows).

## Comparison Table

Exact metrics from post-hoc evaluation on eval_silver (5,000 claims), log: eval_all_286597.log.

| Model Version | Training Data | F1 | Precision | Recall |
|---|---|---|---|---|
| 1. Baseline | Frozen embeddings (no fine-tuning) | 0.7696 | 0.7845 | 0.7553 |
| 2. Assignment 2 | Silver + Gold (GPT-4o-mini HITL) | 0.8099 | 0.8207 | 0.7994 |
| 3. Assignment 3 | Silver + Gold (CrewAI MAS) | 0.8115 | 0.8224 | 0.8010 |

## Reflection

The CrewAI MAS produced a marginal improvement over the simple GPT-4o-mini approach
(0.8115 vs. 0.8099 F1), suggesting the structured debate adds modest label quality for
borderline cases. However, the improvement is small, and the MAS is substantially more
expensive in API cost (~$0.30 vs. ~$0.02) and slower due to three sequential agent calls
per claim. For a 100-claim HITL step, the debate format is justifiable as a quality
control mechanism, but the downstream PatentSBERTa gain does not clearly offset the
added engineering complexity at this scale.
