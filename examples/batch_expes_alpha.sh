#!/bin/bash

#SBATCH -n 240
#SBATCH --mem-per-cpu=8000
#SBATCH --time=01:00:00
#SBATCH --job-name=alpha_expes
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray

srun -W 3600 -n 240 python3.10 experiments_alpha_pre.py --dataset=adult