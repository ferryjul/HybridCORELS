#!/bin/bash

#SBATCH -n 275
#SBATCH --mem-per-cpu=4000
#SBATCH --time=1:00:00
#SBATCH --job-name=expes_hycopre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0,1

srun -W 3600 -n 275 python3.10 3_2_learn_BB_part.py --dataset=${SLURM_ARRAY_TASK_ID}