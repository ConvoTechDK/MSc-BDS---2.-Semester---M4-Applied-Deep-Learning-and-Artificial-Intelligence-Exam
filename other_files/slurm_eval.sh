#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/eval_all_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/eval_all_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

echo "=== Eval job started: $(date) ==="
source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam

cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)

python eval_all_models.py
echo "=== Eval done ==="
