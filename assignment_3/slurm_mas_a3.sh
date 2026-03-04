#!/bin/bash
#SBATCH --job-name=mas_a3
#SBATCH --output=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/mas_a3_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/uo02pm/exam_m4/logs/mas_a3_%j.log
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00

echo "Job started: $(date)"
source /ceph/home/student.aau.dk/uo02pm/miniconda3/etc/profile.d/conda.sh
conda activate exam
cd /ceph/home/student.aau.dk/uo02pm/exam_m4
export $(cat .env | xargs)

echo "=== Starting A3 MAS ==="
python assignment_3/03_mas_crewai.py
echo "=== MAS done ==="
