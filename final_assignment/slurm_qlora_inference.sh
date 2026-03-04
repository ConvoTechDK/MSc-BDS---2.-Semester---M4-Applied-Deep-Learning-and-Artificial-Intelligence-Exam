#!/bin/bash
#SBATCH --job-name=qlora_infer
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/qlora_infer_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/qlora_infer_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

echo "Job started: $(date)"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)

echo "=== Starting QLoRA inference on 100 claims ==="
python final_assignment/00_qlora_inference.py
echo "=== QLoRA inference done ==="
echo "Next step: sbatch final_assignment/slurm_mas_final.sh"
