#!/bin/bash
#SBATCH --job-name=a2_baseline
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/baseline_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/baseline_%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)
echo "=== Starting baseline ==="
python assignment_2/01_baseline.py
echo "=== Baseline done ==="
