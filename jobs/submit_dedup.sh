#!/bin/bash
#PBS -N sangraha_dedup
#PBS -l ncpus=1
#PBS -q cpu
#PBS -o logs/sangraha_dedup.log
#PBS -e logs/sangraha_dedup.err
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m ae
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate tesseract_env
cd /home/shubham.agarwal_phd24/projects/BERTaav
echo "=== Job ${PBS_JOBID} | $(date) ==="
python scripts/dedup_sangraha.py
echo "=== Finished at $(date) ==="
