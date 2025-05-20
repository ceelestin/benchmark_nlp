#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=lsd@h100
#SBATCH --cpus-per-gpu 6
#SBATCH --partition=gpu_p6
#SBATCH --job-name=ceve_finetuning
#SBATCH --output=slurm_output/ft_%a.out
#SBATCH --error=slurm_output/ft_%a.out

module purge
module load arch/h100 pytorch-gpu

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs2.txt)

python nlpfinetuning_full.py $params
