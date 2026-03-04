#!/bin/bash
#SBATCH --job-name=qlora_patent
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/qlora_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/qlora_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

echo "Job started: $(date)"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)

echo "=== Starting QLoRA training ==="
python final_assignment/01_qlora_train.py
echo "=== QLoRA done ==="
