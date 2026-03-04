#!/bin/bash
#SBATCH --job-name=mas_final
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/mas_final_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/mas_final_%j.log
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

echo "Job started: $(date)"
source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam
cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)

echo "=== Starting Final MAS ==="
python final_assignment/02_mas_qlora.py
echo "=== Final MAS done ==="
