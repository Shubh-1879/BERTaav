#!/bin/bash

#PBS -N test_download
#PBS -l ncpus=4
#PBS -o test_download_out.log
#PBS -e test_download_err.log
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m ae
#PBS -q cpu

module load compiler/anaconda3

# Activate your environment
source activate tesseract_env

# Navigate to your project directory
cd /home/shubham.agarwal_phd24/BERTaav

python3 scripts/test_download.py

echo "Job finished at: $(date)"
