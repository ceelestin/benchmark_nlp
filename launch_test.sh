#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=sth@h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --cpus-per-gpu 10
#SBATCH --array=1
#SBATCH --job-name=ceve_qwen_test
#SBATCH --output=slurm_output/test_qwen_%a.out
#SBATCH --error=slurm_output/test_qwen_%a.out

module purge
module load arch/h100 pytorch-gpu
# module load pytorch-gpu

# If a venv layer was needed (Step 2, Branch B), activate it here:
# source $WORK/venvs/qwen_layer/bin/activate

# Point HF at the primed offline cache on $WORK (compute nodes have no internet)
export HF_HOME=$WORK/hf_cache

# Force offline mode before importing transformers/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs/test_qwen_light.txt)

python nlpfinetuning_full_adapted.py $params
