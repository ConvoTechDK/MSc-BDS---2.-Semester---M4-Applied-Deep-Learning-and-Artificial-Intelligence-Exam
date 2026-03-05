---
title: "Green Patent Detection: From Active Learning to QLoRA-Powered Multi-Agent Systems"
author: "MSc BDS — M4: Applied Deep Learning and Artificial Intelligence"
date: "March 5, 2026"
geometry: margin=2cm
fontsize: 11pt
---

## Introduction

Patent green technology classification is a high-stakes NLP task: mislabeling a patent as green (greenwashing) or missing a legitimate Y02 contribution has real policy consequences. This portfolio implements an iterative pipeline starting from a frozen embedding baseline and progressively improving label quality through active learning, human-in-the-loop (HITL) validation, and advanced agentic architectures. The central artifact is a fine-tuned PatentSBERTa binary classifier evaluated at four stages of increasing label sophistication.

## Model Comparison

All models are evaluated on the same `eval_silver` held-out set (5,000 claims). Exact F1/Precision/Recall computed post-hoc by loading each model from HuggingFace and running inference (log: eval_all_287041.log).

| Model Version | Training Data Source | F1 | Precision | Recall |
|---|---|---|---|---|
| 1. Baseline | Frozen PatentSBERTa + Logistic Regression | 0.7696 | 0.7845 | 0.7553 |
| 2. Assignment 2 | Silver + Gold (GPT-4o-mini HITL, 100 claims) | 0.8099 | 0.8207 | 0.7994 |
| 3. Assignment 3 | Silver + Gold (CrewAI MAS, 100 claims) | 0.8115 | 0.8224 | 0.8010 |
| 4. Final | Silver + Gold (QLoRA MAS + Targeted HITL, 100 claims) | 0.8097 | 0.8213 | 0.7986 |

Each fine-tuning stage adds 100 high-quality gold labels for the most uncertain pool examples on top of 35,000 silver training labels. The improvement from Baseline to Assignment 2 (+4.0 pp F1) is the largest gain, reflecting that correcting silver labels on the 100 most uncertain examples has the greatest leverage. Assignments 3 and Final show smaller incremental changes, as the 100-claim gold set is a small fraction of the training data.

## Engineering Challenges: Integrating QLoRA into an Agentic Workflow

The primary engineering challenge of the Final Assignment was connecting a locally fine-tuned QLoRA adapter to the CrewAI framework. Three problems arose:

**Adapter loading and memory.** Loading Llama-3.2-3B-Instruct in 4-bit NF4 quantization via BitsAndBytesConfig, then applying the LoRA adapter with PeftModel.from_pretrained(), consumed approximately 8 GB of GPU VRAM. Sequencing the quantized local model (for the Advocate) and OpenAI API calls (for Skeptic and Judge) required careful memory management to avoid conflicts.

**LangChain compatibility.** CrewAI agents require a LangChain-compatible LLM object. Quantized models loaded via bitsandbytes + peft do not expose this interface without additional wrappers (e.g. a local vLLM or text-generation-inference server), which adds deployment complexity on HPC SLURM nodes. The deployed system therefore uses a two-stage design: the QLoRA adapter runs standalone inference on all 100 claims (GPU SLURM job producing `qlora_predictions_100.csv`), then the Advocate agent receives each claim's QLoRA prediction and raw output as task input and argues from it (API-only SLURM job). This decouples GPU inference from agent orchestration, keeping each step simple and reproducible.

**JSON reliability.** Models trained on classification prompts occasionally produce free-text instead of structured JSON. The Judge role was therefore kept exclusively on GPT-4o-mini with response_format={"type": "json_object"}, which reliably produces parseable output.

## Agent Disagreement and HITL Analysis

Across the 100 high-risk claims processed by the Final Assignment's QLoRA-powered MAS, the QLoRA adapter and MAS agreed on 71 out of 100 claims. The Judge resolved all 29 disagreements decisively: **0 claims triggered exception-based HITL** (all received medium or high confidence). Confidence breakdown: 83 high, 17 medium, 0 low. In Assignment 2, the human simulation overrode 4 of 100 LLM suggestions (those rated low confidence by GPT-4o-mini). In Assignment 3, the CrewAI MAS produced 0 low-confidence decisions across all 100 claims -- the structured debate consistently resolved ambiguous cases without requiring human intervention.

## Did the Advanced Architecture Improve the Final Model?

The results show a non-monotonic progression: Assignment 3 (MAS) marginally outperforms Assignment 2 (single LLM) at 0.8115 vs. 0.8099 F1, while the Final model (QLoRA MAS) scores 0.8097, slightly below Assignment 3. These differences are small relative to the complexity added. The main driver of performance gain across all stages is uncertainty-based active learning: by targeting gold labels at the 100 most uncertain examples, each fine-tuning stage improves calibration on the hardest cases. The advanced architectures do not substantially outperform the simpler GPT-4o-mini HITL at this scale, but provide richer rationales and more consistent structured outputs that would likely compound over multiple active learning rounds.

## HuggingFace Links

- Dataset: https://huggingface.co/datasets/Peter512/patents-50k-green
- Assignment 2 model: https://huggingface.co/Peter512/patentsbert-green-a2
- Assignment 3 model: https://huggingface.co/Peter512/patentsbert-green-a3
- Final model: https://huggingface.co/Peter512/patentsbert-green-final

## GitHub Repository

- https://github.com/ConvoTechDK/MSc-BDS---2.-Semester---M4-Applied-Deep-Learning-and-Artificial-Intelligence-Exam

## Video Demo

- Video URL: https://aaudk-my.sharepoint.com/:v:/g/personal/uo02pm_student_aau_dk/IQBKX_oWIFDaQ7D6pkBTrOzyAbOkEcWXWvf-iPgF3nO59So?e=al88kl&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D

OR:

- https://drive.google.com/file/d/1fiGB6DV7pdDACXYH3dsz1rwMGsliNIt8/view?usp=sharing
