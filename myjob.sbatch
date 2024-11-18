#!/bin/bash
#SBATCH -A changsheng
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 10
cd $SLURM_SUBMIT_DIR
nohup python -u run.py -acc True -epoch 200
