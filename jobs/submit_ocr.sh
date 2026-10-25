#!/bin/bash
#PBS -N run_ocr
#PBS -J 0-49                       
#PBS -l ncpus=4
#PBS -q cpu                        
#PBS -o logs/run_ocr_^array_index^.log
#PBS -e logs/run_ocr_^array_index^.err
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m ae

source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate tesseract_env
export PATH="/home/shubham.agarwal_phd24/.conda/envs/tesseract_env/bin:$PATH"
conda activate tesseract_env

cd /home/shubham.agarwal_phd24/projects/BERTaav

echo "=== Job ${PBS_JOBID} | Array index ${PBS_ARRAY_INDEX} | $(date) ==="

python scripts/run_ocr.py \
    --chunk-id ${PBS_ARRAY_INDEX} \
    --num-chunks 50

echo "=== Finished at $(date) ==="
