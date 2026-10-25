#!/bin/bash
#PBS -N am_pool
#PBS -q cpu
#PBS -l select=1:ncpus=2
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
python scripts/am_sample.py pool-sang
