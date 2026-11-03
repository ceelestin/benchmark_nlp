#!/bin/bash
# Production runs of nlpfinetuning_aligned.py (CE's go, 2026-11-03): BERT and XLM-RoBERTa, n_tr = 1,000 and 3,000,
# seeds 0-49, K = 20 splits, the full 600,000-review benchmark scored after every split, the outer chunks after the
# first. One seed per array task; the line of $CONFIG given by the task index holds its arguments.
#   sbatch --array=1-100 --export=ALL,CONFIG=configs/aligned_roberta.txt launch_aligned.sh
#SBATCH --time=02:30:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=sth@h100
#SBATCH --cpus-per-gpu 10
#SBATCH --job-name=ceve_yelp_aligned
#SBATCH --output=slurm_output/aligned_%A_%a.out
module purge
module load arch/h100 pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $WORK/benchmark_nlp
params=$(awk -v i="${SLURM_ARRAY_TASK_ID}" 'NR==i' "${CONFIG}")
python nlpfinetuning_aligned.py run $params --n-splits 20 --data-dir $SCRATCH/yelp_aligned --out-dir results_aligned
