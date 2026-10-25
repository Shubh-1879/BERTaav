#!/bin/bash
#PBS -N pretok_gen
#PBS -q cpu
#PBS -J 0-19
#PBS -l select=1:ncpus=4
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/pretokenize_generic.py --task $PBS_ARRAY_INDEX --ntasks 20 --limit-shards 120
