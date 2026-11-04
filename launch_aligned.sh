#!/bin/bash
# SLURM array launcher of nlpfinetuning_aligned.py (the BERT and XLM-RoBERTa runs of the paper): n_tr = 1,000 and
# 3,000, seeds 0-49, K = 20 splits, the full 600,000-review benchmarking set scored after every split, the outer
# chunks after the first. One seed per array task; line i of $CONFIG holds the arguments of task i. After the
# prepare step (README.md), submit from the repository root:
#   mkdir -p slurm_output
#   sbatch --array=1-100 --export=ALL,CONFIG=configs/aligned_roberta.txt launch_aligned.sh
#   sbatch --array=1-100 --export=ALL,CONFIG=configs/aligned_bert.txt launch_aligned.sh
# DATA_DIR (default $SCRATCH/yelp_aligned) must be the --data-dir of the prepare step. Written for the H100
# partition of Jean-Zay (IDRIS): set your account below, and adapt the #SBATCH lines and the module line.
#SBATCH --time=02:30:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=<project>@h100
#SBATCH --cpus-per-gpu 10
#SBATCH --job-name=yelp_aligned
#SBATCH --output=slurm_output/aligned_%A_%a.out
module purge
module load arch/h100 pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
DATA_DIR=${DATA_DIR:-$SCRATCH/yelp_aligned}
params=$(awk -v i="${SLURM_ARRAY_TASK_ID}" 'NR==i' "${CONFIG}")
python nlpfinetuning_aligned.py run $params --n-splits 20 --data-dir "$DATA_DIR" --out-dir results_aligned
