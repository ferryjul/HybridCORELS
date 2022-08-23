#!/bin/bash

#SBATCH -n 264
#SBATCH --mem-per-cpu=4500
#SBATCH --time=01:00:00
#SBATCH --job-name=alpha_expes
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray

srun -W 3600 -n 264 python3 experiments_alpha_pre.py --dataset=compas