#!/bin/bash
#PBS -N leak
#PBS -q cpu
#PBS -l select=1:ncpus=4
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
L=/storage/shubham.agarwal_phd24/leak
P=/storage/shubham.agarwal_phd24/pilot_dapt
python scripts/neardup.py leak --val $P/lok_val.jsonl --train-jsonl $L/lok_train_full.jsonl --out $L/leak_lok.json
python scripts/neardup.py leak --val $P/sang_val.jsonl --train-sang --out $L/leak_sang.json
