#!/bin/bash
#PBS -N pdf_to_images
#PBS -J 0-49
#PBS -l ncpus=4,walltime=04:00:00
#PBS -q cpu
#PBS -o logs/pdf_to_images_^array_index^.log
#PBS -e logs/pdf_to_images_^array_index^.err
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m ae

source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate tesseract_env
export PATH="/home/shubham.agarwal_phd24/.conda/envs/tesseract_env/bin:$PATH"

cd /home/shubham.agarwal_phd24/projects/BERTaav
echo "=== Job ${PBS_JOBID} | Array index ${PBS_ARRAY_INDEX} | $(date) ==="
echo "Python: $(which python)"
echo "pdfinfo: $(which pdfinfo 2>/dev/null || echo NOT FOUND)"

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
