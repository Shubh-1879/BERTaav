#!/bin/bash
#PBS -N pdf_to_images
#PBS -J 0-49                        # 50 parallel array jobs (chunks 0-49)
#PBS -l ncpus=4
#PBS -q cpu                         # adjust to your enabled queue
#PBS -o logs/pdf_to_images_^array_index^.log
#PBS -e logs/pdf_to_images_^array_index^.err
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m ae

module load compiler/anaconda3
source activate tesseract_env

cd /home/shubham.agarwal_phd24/projects/BERTaav

echo "=== Job ${PBS_JOBID} | Array index ${PBS_ARRAY_INDEX} | $(date) ==="

# First array job (index 0) also syncs the manifest to pick up any new files
# that arrived since last run. Others just process their chunk.
if [ "${PBS_ARRAY_INDEX}" = "0" ]; then
    python scripts/pdf_to_images.py \
        --sync-first \
        --chunk-id ${PBS_ARRAY_INDEX} \
        --num-chunks 50
else
    python scripts/pdf_to_images.py \
        --chunk-id ${PBS_ARRAY_INDEX} \
        --num-chunks 50
fi

echo "=== Finished at $(date) ==="
