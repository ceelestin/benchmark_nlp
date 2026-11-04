#!/bin/bash
# SLURM array launcher of nlpfinetuning_full_adapted.py (the Qwen3-0.6B runs of the paper): array task i runs the
# arguments on line i of $CONFIG. Submit from the repository root, for example
#   mkdir -p slurm_output
#   sbatch --array=1-50 --export=ALL,CONFIG=configs/qwen_0.6b_1000.txt launch.sh        # K = 20 splits
#   sbatch --array=1-50 --export=ALL,CONFIG=configs/qwen_0.6b_1000_nocv.txt launch.sh   # single split
# The array size is the number of lines of the file (wc -l counts one less when the last line has no newline).
# Written for the H100 partition of Jean-Zay (IDRIS): set your account below, and adapt the #SBATCH lines and the
# module line to your cluster.
#SBATCH --time=12:00:00
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --account=<project>@h100
#SBATCH --cpus-per-gpu 10
#SBATCH --job-name=yelp_qwen3
#SBATCH --output=slurm_output/qwen3_%A_%a.out
#SBATCH --error=slurm_output/qwen3_%A_%a.out

if [ -z "${CONFIG}" ]; then
    echo "Set CONFIG to a configuration file, e.g. sbatch --export=ALL,CONFIG=configs/qwen_0.6b_1000.txt ..." >&2
    exit 1
fi

module purge
module load arch/h100 pytorch-gpu/py3/2.11.0   # the module of the Qwen3-0.6B runs (see requirements-qwen3.txt)

# The compute nodes have no internet access: $HF_HOME must already hold the dataset and the checkpoint
# (load_data_and_model.py, run beforehand on a node that has it). $WORK is the work space of Jean-Zay.
export HF_HOME=$WORK/hf_cache

# Offline mode, set before transformers and datasets are imported
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

params=$(awk -v idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' "${CONFIG}")

python nlpfinetuning_full_adapted.py $params
