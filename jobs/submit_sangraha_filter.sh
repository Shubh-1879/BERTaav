#!/bin/bash
#PBS -N sangraha_filter
#PBS -J 0-49
#PBS -l ncpus=1
#PBS -q cpu
#PBS -o logs/sangraha_filter_^array_index^.log
#PBS -e logs/sangraha_filter_^array_index^.err
#PBS -M shubham.agarwal_phd24@ashoka.edu.in
#PBS -m ae
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate tesseract_env
cd /home/shubham.agarwal_phd24/projects/BERTaav
echo "=== Job ${PBS_JOBID} | Array index ${PBS_ARRAY_INDEX} | $(date) ==="

# EDIT THESE after running calibrate + sample:
MIN_DISTINCT=2
MIN_DENSITY=4.0

python scripts/sangraha_filter.py run \
    --chunk-id ${PBS_ARRAY_INDEX} \
    --num-chunks 50 \
    --min-distinct ${MIN_DISTINCT} \
    --min-density ${MIN_DENSITY}
echo "=== Finished at $(date) ==="

