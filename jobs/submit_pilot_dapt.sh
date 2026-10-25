#!/bin/bash
#PBS -N pilot_dapt
#PBS -q gpu
#PBS -l select=1:ncpus=8:naccelerators=1
#PBS -l walltime=03:00:00
#PBS -o logs/pilot_dapt.log
#PBS -e logs/pilot_dapt.err
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCPUS=8
echo "=== $(date) ==="; nvidia-smi || echo "NO GPU"
S=${SOURCES:-lok+sang}
python scripts/pilot_dapt.py run --steps ${STEPS:-600} --sources "${S//+/,}" --tag ${TAG:-mix600}
echo "=== done $(date) ==="
