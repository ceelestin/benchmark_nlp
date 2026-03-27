#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=ozx@h100
#SBATCH --cpus-per-gpu 10
#SBATCH --job-name=ceve_finetuning
#SBATCH --output=slurm_output/ft_%a.out
#SBATCH --error=slurm_output/ft_%a.out

module purge
module load arch/h100 pytorch-gpu
# module load pytorch-gpu

# Force offline mode before importing transformers/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs/test_qwen.txt)

python nlpfinetuning_full_adapted.py $params
