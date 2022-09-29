#!/bin/bash

#SBATCH -n 180
#SBATCH --mem-per-cpu=10000
#SBATCH --time=15:00:00
#SBATCH --job-name=expes_hycopre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0,1

srun -W 54000 -n 180 python3.10 expes_hybridcorelspre.py --dataset=${SLURM_ARRAY_TASK_ID}