#!/bin/bash
#SBATCH -J test
#SBATCH -p gpu
#SBATCH -o bash.log
#SBATCH -w gpu002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

eval "$(conda shell.bash hook)"
conda activate batch
python  w.py  > run.log 2>error.log
