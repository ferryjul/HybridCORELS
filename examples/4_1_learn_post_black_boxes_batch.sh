#!/bin/bash

#SBATCH -n 15
#SBATCH --mem-per-cpu=9000
#SBATCH --time=24:00:00
#SBATCH --job-name=expes_post
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=1-2

srun -W 86400 -n 15 python3.10 4_1_learn_post_black_boxes.py --dataset=${SLURM_ARRAY_TASK_ID}