#!/bin/bash
#PBS -N pibdedup
#PBS -q cpu
#PBS -l select=1:ncpus=4
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export PYTHONHASHSEED=0
python scripts/neardup.py pib --in-dir /storage/shubham.agarwal_phd24/pib_raw --out-dir /storage/shubham.agarwal_phd24/pib_clean
