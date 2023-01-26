#!/bin/bash

#SBATCH -n 135
#SBATCH --mem-per-cpu=5000
#SBATCH --time=00:30:00
#SBATCH --job-name=expes_pre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0-2

srun -W 1800 -n 135 python3.10 4_1_learn_pre_prefixes_remeasure_missing.py --expe_id=${SLURM_ARRAY_TASK_ID}