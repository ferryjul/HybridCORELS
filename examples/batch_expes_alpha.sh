#!/bin/bash

#SBATCH -n 450
#SBATCH --mem-per-cpu=6500
#SBATCH --time=1:30:00
#SBATCH --job-name=expes_hycopre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0,1

srun -W 4800 -n 450 python3.10 3_1_learn_best_prefixes.py --dataset=${SLURM_ARRAY_TASK_ID}