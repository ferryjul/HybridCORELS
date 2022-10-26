#!/bin/bash

#SBATCH --mem-per-cpu=6500
#SBATCH --time=1:30:00
#SBATCH --job-name=pre_expes_hycopre
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray
#SBATCH --array=0,1,2

python3.10 example_HybridCORELS.py --dataset=${SLURM_ARRAY_TASK_ID} > ./results/pre_tests/example_test_dataset${SLURM_ARRAY_TASK_ID}.txt