#!/bin/bash
# Smoke test of nlpfinetuning_aligned.py (CE's go, 2026-11-03): one seed (999, outside the production seeds),
# K = 4 splits, n_tr = 1,000, XLM-RoBERTa; the full 600,000-review benchmark scored after every split, the outer
# chunks after the first, and a padding check on 20,000 benchmark reviews (timing and identical predictions).
#SBATCH --time=02:00:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=sth@h100
#SBATCH --cpus-per-gpu 10
#SBATCH --job-name=ceve_yelp_smoke
#SBATCH --output=slurm_output/aligned_smoke_%j.out
#SBATCH --error=slurm_output/aligned_smoke_%j.out
module purge
module load arch/h100 pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $WORK/benchmark_nlp
python nlpfinetuning_aligned.py run --model-choices roberta --train-sizes 1000 --n-splits 4 --seeds 999 1000 \
    --data-dir $SCRATCH/yelp_aligned --out-dir results_aligned_smoke --check-padding 20000
