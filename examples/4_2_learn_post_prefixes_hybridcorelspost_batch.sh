#!/bin/bash

#SBATCH -n 324
#SBATCH --mem-per-cpu=9000
#SBATCH --time=02:00:00
#SBATCH --job-name=expes_post_2
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0-44

srun -W 7200 -n 324 python3.10 4_2_learn_post_prefixes_hybridcorelspost.py --expe_id=${SLURM_ARRAY_TASK_ID}


