#!/bin/bash
#SBATCH --job-name=finetune_a3
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/finetune_a3_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/finetune_a3_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

echo "Job started: $(date)"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)

echo "=== Starting A3 fine-tuning ==="
python assignment_3/04_finetune_v2.py
echo "=== Fine-tuning done ==="
