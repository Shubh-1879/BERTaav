#!/bin/bash
#PBS -N gpucheck
#PBS -q gpu
#PBS -l select=1:ncpus=8:naccelerators=1
#PBS -l walltime=00:05:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCPUS=8
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
