#!/bin/bash
#SBATCH --job-name=a2_hitl
#SBATCH --output=logs/hitl_%j.log
#SBATCH --error=logs/hitl_%j.log
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

mkdir -p ~/exam_m4/assignment_2/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate exam

cd ~/exam_m4
export $(cat .env | xargs)

echo "=== Step 2: Uncertainty sampling ==="
python assignment_2/02_uncertainty_sampling.py

echo "=== Step 3: HITL LLM labeling ==="
python assignment_2/03_hitl_llm.py
echo "=== HITL done ==="
