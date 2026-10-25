#!/bin/bash
#PBS -N dapt_full
#PBS -q gpu
#PBS -l select=1:ncpus=8:naccelerators=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/
source /apps/compilers/anaconda3-23.07.0/etc/profile.d/conda.sh
conda activate /storage/shubham.agarwal_phd24/envs/dapt
cd /home/shubham.agarwal_phd24/projects/BERTaav
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCPUS=8
nvidia-smi
EXTRA=""
[ -n "$BENCH" ] && EXTRA="--bench $BENCH"
[ -n "$MAXSTEPS" ] && EXTRA="$EXTRA --max-steps $MAXSTEPS --eval-every 10 --ckpt-every 10"
python scripts/dapt_full.py --tag ${TAG:-full} --total-tokens ${TOKENS:-300000000} --max-hours ${MAXH:-11} $EXTRA --lr ${LR:-1e-4}
