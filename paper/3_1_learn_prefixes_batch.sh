#!/bin/bash

#SBATCH -n 675
#SBATCH --mem-per-cpu=9000
#SBATCH --time=02:00:00
#SBATCH --job-name=expes_hycopre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0,1,2

srun -W 7200 -n 675 python3.10 3_1_learn_best_prefixes.py --dataset=${SLURM_ARRAY_TASK_ID}