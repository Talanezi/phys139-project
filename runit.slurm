#!/bin/bash
#SBATCH --job-name=B
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=j.rose@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100gb
#SBATCH --time=5-00:00:00
#SBATCH --output=results/logs/Training_B_%j.log
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

module purge
module load pytorch/1.10

date

echo "Training on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS tasks, each with $SLURM_CPUS_PER_TASK cores."

srun python -u train.py

date

