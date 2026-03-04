# Assignment 1: SGD Mechanics and Attention Contextualization

**Course:** MSc BDS - M4: Applied Deep Learning and Artificial Intelligence
**Deadline:** Monday 10 February 2026

## Overview

This assignment covers two foundational concepts in deep learning:

- **Part A:** Manual SGD — step-by-step gradient descent computation on the insurance dataset
- **Part B:** Self-Attention — implementing scaled dot-product self-attention from scratch to demonstrate how context shifts a homonym's representation

## Files

| File | Description |
|---|---|
| `assignment_1.ipynb` | Main notebook with all code and explanations |
| `attention_shift.png` | Generated visualization of embedding shifts |

## How to Run

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
jupyter notebook assignment_1.ipynb
```

Run all cells in order. The notebook is self-contained and requires no GPU or external API calls.

## Part A: Manual SGD

**Dataset:** Seaborn insurance dataset (age as x, charges as t)
**Hyperparameters:** alpha=0.0001, w_0=0.5

The notebook manually computes forward pass, loss, gradient, and weight update for the first 3 samples, presenting results in a table and verifying with numpy scalar operations.

| Step | Formula |
|---|---|
| Forward | y_hat = x * w |
| Loss | L = (t - y_hat)^2 |
| Gradient | dL/dw = 2x(y_hat - t) |
| Update | w_new = w_old - alpha * dL/dw |

## Part B: Self-Attention for Homonym "bank"

- Sentence 1: "The bank approved the loan." (financial meaning)
- Sentence 2: "She sat by the bank of the river." (geographic meaning)

Self-attention with Q=K=V=E (scaled dot-product) produces different contextualized representations for "bank" in each sentence, demonstrating that the same token gets different embeddings based on surrounding context. This is the mechanism behind BERT's contextual embeddings.
