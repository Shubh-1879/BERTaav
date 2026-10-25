#!/bin/bash
#PBS -N eval_models
#PBS -q gpu
#PBS -l select=1:ncpus=8:naccelerators=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCPUS=8
nvidia-smi
python scripts/eval_models.py --out /storage/shubham.agarwal_phd24/dapt_full/eval \
  base=ai4bharat/IndicBERTv2-MLM-only dapt=/storage/shubham.agarwal_phd24/models/bertaav_dapt_300M
