#!/bin/bash
# Smoke test of nlpfinetuning_aligned.py, one job: one seed (999, outside the paper's seeds), K = 4 splits,
# n_tr = 1,000, XLM-RoBERTa; the full 600,000-review benchmark scored after every split, the outer chunks after the
# first, and a padding check on 20,000 benchmark reviews (time, and predictions identical to padding each block of
# 1,000 reviews to its longest one). After the prepare step, from the repository root:
#   mkdir -p slurm_output && sbatch launch_aligned_smoke.sh
#SBATCH --time=02:00:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=<project>@h100
#SBATCH --cpus-per-gpu 10
#SBATCH --job-name=yelp_smoke
#SBATCH --output=slurm_output/aligned_smoke_%j.out
#SBATCH --error=slurm_output/aligned_smoke_%j.out
module purge
module load arch/h100 pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
DATA_DIR=${DATA_DIR:-$SCRATCH/yelp_aligned}
python nlpfinetuning_aligned.py run --model-choices roberta --train-sizes 1000 --n-splits 4 --seeds 999 1000 \
    --data-dir "$DATA_DIR" --out-dir results_aligned_smoke --check-padding 20000
