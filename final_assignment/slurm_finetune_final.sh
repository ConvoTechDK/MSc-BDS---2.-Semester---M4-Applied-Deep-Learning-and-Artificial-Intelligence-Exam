#!/bin/bash
#SBATCH --job-name=finetune_final
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/finetune_final_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/finetune_final_%j.log
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

echo "=== Starting Final fine-tuning ==="
python final_assignment/03_finetune_final.py
echo "=== Final fine-tuning done ==="
