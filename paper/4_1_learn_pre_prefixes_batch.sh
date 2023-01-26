#!/bin/bash

#SBATCH -n 135
#SBATCH --mem-per-cpu=9000
#SBATCH --time=02:00:00
#SBATCH --job-name=expes_pre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0-35

srun -W 7200 -n 135 python3.10 4_1_learn_pre_prefixes.py --expe_id=${SLURM_ARRAY_TASK_ID}