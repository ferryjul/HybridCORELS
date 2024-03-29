#!/bin/bash

#SBATCH -n 180
#SBATCH --mem-per-cpu=9000
#SBATCH --time=46:00:00
#SBATCH --job-name=expes_pre_last
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=2

srun -W 165600 -n 180 python3.10 4_2_learn_pre_black_boxes.py --dataset=${SLURM_ARRAY_TASK_ID}