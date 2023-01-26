#!/bin/bash

#SBATCH --mem-per-cpu=9000
#SBATCH --time=02:00:00
#SBATCH --job-name=expes_post
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0-2

python3.10 4_0_mine_datasets.py --dataset=${SLURM_ARRAY_TASK_ID}