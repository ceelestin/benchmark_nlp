#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --account=lsd@v100
#SBATCH --cpus-per-gpu 10
#SBATCH --partition=gpu_p13
#SBATCH --job-name=ceve_finetuning
#SBATCH --output=slurm_output/ft_%a.out
#SBATCH --error=slurm_output/ft_%a.out

module purge
# module load arch/h100 pytorch-gpu
module load pytorch-gpu

# Force offline mode before importing transformers/datasets
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs_cv_100seeds.txt)

python nlpfinetuning_full_adapted.py $params
