#!/bin/bash
#PBS -N run_ocr
#PBS -J 0-49                        # 50 parallel array jobs
#PBS -l ncpus=4
#PBS -q cpu                         # adjust to your enabled queue
#PBS -o logs/run_ocr_^array_index^.log
#PBS -e logs/run_ocr_^array_index^.err
#PBS -M you@example.com
#PBS -m ae

module load compiler/anaconda3
source activate tesseract_env

cd /home/USER/projects/BERTaav

echo "=== Job ${PBS_JOBID} | Array index ${PBS_ARRAY_INDEX} | $(date) ==="

python scripts/run_ocr.py \
    --chunk-id ${PBS_ARRAY_INDEX} \
    --num-chunks 50

echo "=== Finished at $(date) ==="
