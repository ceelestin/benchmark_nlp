# Yelp Review Full runs of "Crossing the Validation Crisis"

Fine-tuning loops and run configurations of the Yelp Review Full experiments of

> Célestin Eve, Gaël Varoquaux and Thomas Moreau (2026). *Crossing the Validation Crisis: Cross-Validation
> Reduces Benchmarking Variance Surprisingly Well.*

The entry point to the code of the paper is
[ceelestin/benchmark_regression](https://github.com/ceelestin/benchmark_regression): it holds the simulated
benchmark and the scripts that draw every figure and table of the paper from the run outputs, the Yelp and PCam
ones included. This repository fine-tunes the language models and writes those outputs; its companion
[ceelestin/benchmark_pcam](https://github.com/ceelestin/benchmark_pcam) does the same for PatchCamelyon.

| Path | Contents |
|---|---|
| `nlpfinetuning_aligned.py` | **BERT and XLM-RoBERTa runs of the paper**, on the protocol of the PCam runs; two steps, `prepare` and `run` |
| `nlpfinetuning_full_adapted.py` | **Qwen3-0.6B runs of the paper**, on an earlier protocol (below) |
| `load_data_and_model.py` | fills the Hugging Face cache with the dataset and the checkpoints, for offline compute nodes |
| `configs/*.txt` | the run sets of the paper: one line = the arguments of one job (table below) |
| `configs/legacy/`, `legacy/` | earlier run sets and the 2025 fine-tuning loop, not used in the paper |
| `launch_aligned.sh`, `launch_aligned_smoke.sh` | SLURM array launcher of `nlpfinetuning_aligned.py`, and a one-job smoke test |
| `launch.sh` | SLURM array launcher of `nlpfinetuning_full_adapted.py` |
| `requirements.txt`, `requirements-qwen3.txt` | pinned versions of the two environments |

## What the scripts do

Appendix F.3 of the paper describes both protocols. Every model is fully fine-tuned with a 5-class head
(`AutoModelForSequenceClassification`) by the Hugging Face `Trainer`: 3 epochs, AdamW with the default learning
rate of 5e-5, batches of 32, mixed precision, reviews truncated to 512 tokens; the epoch with the best validation
accuracy is kept. In each of the K ShuffleSplit splits of the study set (test fraction 0.2), the model is trained on
60% of the study set and validated on 20%.

**`nlpfinetuning_aligned.py`** (BERT `google-bert/bert-base-cased`, XLM-RoBERTa `FacebookAI/xlm-roberta-base`):

* a fixed stratified split (random_state 42) keeps 600,000 of the 650,000 training reviews as the
  **benchmarking set** and leaves a **pool** of 50,000;
* each seed draws a stratified **study set** of 1.25 n_tr reviews from the pool (`--train-sizes` gives n_tr, the
  training and validation subsets together); the rest of the pool is the **outer set**;
* after every split, the kept model is scored (accuracy, negative log-likelihood, Brier score) on the validation
  subset, the test fold and the whole benchmarking set; the model of the first split also scores the outer set, in
  chunks of the test-fold size (at most 200);
* the study-only redundancy statistics of the PCam script are computed from the per-sample test losses;
* `prepare` tokenizes the 650,000 reviews once per model and fixes the split (CPU); `run` trains and scores (GPU).
  Each run writes one Parquet file per (model, n_tr) to `--out-dir`.

**`nlpfinetuning_full_adapted.py`** (Qwen3-0.6B `Qwen/Qwen3-0.6B`; the BERT and XLM-RoBERTa runs of January 2026
also used it, but the paper no longer reports them):

* each seed shuffles the 650,000 training reviews: the first 600,000 form the benchmarking set (not stratified,
  and different for every seed), the next ones the study set, the remaining ones the outer set;
* `--study-sizes` is the size of the study set (1,250 for n_tr = 1,000);
* the K-split runs score the validation subset, the test fold and the benchmarking set; the single-split runs
  (`--disable-cross-validation`), with the same seeds, study sets and first split, train their own model and also
  score the outer set in chunks of the test-fold size;
* reviews are padded to the longest of each block of 1,000. Each call writes one Parquet file to
  `results_output/parquet/`.

## Install

Python 3.12 and one GPU per job (the paper's runs used one NVIDIA H100 each). The two scripts ran in two
environments, the Jean-Zay modules `pytorch-gpu/py3/2.8.0` and `pytorch-gpu/py3/2.11.0`:

```bash
pip install -r requirements.txt          # nlpfinetuning_aligned.py: torch 2.8.0, transformers 4.55.4, datasets 4.0.0
pip install -r requirements-qwen3.txt    # nlpfinetuning_full_adapted.py: torch 2.11.0, transformers 5.7.0, datasets 4.8.5
```

Use one virtual environment per file, and install the torch wheel that matches your CUDA. On Jean-Zay,
`module load arch/h100 pytorch-gpu/py3/2.8.0` (or `.../2.11.0`) is enough; the launchers do it.

## Data

The scripts read the Hugging Face dataset
[`yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) (650,000 training reviews, 5 classes;
the paper's runs used revision `c1f9ee9`) and the three checkpoints (revisions `cd5ef92`, `e73636d` and
`c1899de`). As cluster compute nodes are often offline, fill the cache once on a node with internet access,
then tokenize for the aligned runs (CPU, 8 processes):

```bash
export HF_HOME=$WORK/hf_cache            # the cache the launchers use; any persistent directory
python load_data_and_model.py
python nlpfinetuning_aligned.py prepare --model-choices bert roberta --data-dir $SCRATCH/yelp_aligned
```

**Terms of use.** Yelp Review Full was built by Zhang, Zhao and LeCun (Character-level Convolutional Networks for
Text Classification, *NeurIPS*, 2015) from the Yelp Dataset Challenge. Its Hugging Face card refers to the
[Yelp Dataset Terms of Use](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf),
which allow non-commercial use only; the data, and the run outputs keyed by review, are not redistributed here.
The checkpoints are under Apache-2.0 (BERT, Qwen3-0.6B) and MIT (XLM-RoBERTa).

## Run one job

From the repository root, after the data step:

```bash
python nlpfinetuning_aligned.py run --model-choices roberta --train-sizes 1000 --n-splits 20 --seeds 0 1 \
    --data-dir $SCRATCH/yelp_aligned --out-dir results_aligned
python nlpfinetuning_full_adapted.py --n-splits 20 --seeds 0 1 --study-sizes 1250 --model-choices qwen3-0.6b
```

The first is one line of `configs/aligned_roberta.txt` (XLM-RoBERTa, seed 0, n_tr = 1,000, K = 20) and writes
`results_aligned/yelp_results_CV_shuffle_split_unstrat_roberta_nsplits_20_seeds_0-0_sizes_1000_<date>_<time>.parquet`;
in its output, `study_size` is n_tr and `study_pool_size` the study set. The second is one line of
`configs/qwen_0.6b_1000.txt`; in its output, `study_set_size` is the study set. `launch_aligned_smoke.sh` runs a
short test of the first script (one seed, K = 4).

## From a configuration file to a SLURM array

Each line of a file of `configs/` holds the arguments of one job (one seed, one model, one size). The launchers run
line `$SLURM_ARRAY_TASK_ID` of the file given in `$CONFIG`:

```bash
mkdir -p slurm_output                       # SLURM writes the logs there and does not create it
sbatch --array=1-100 --export=ALL,CONFIG=configs/aligned_bert.txt launch_aligned.sh
sbatch --array=1-50 --export=ALL,CONFIG=configs/qwen_0.6b_1000.txt launch.sh
```

The array size is the number of lines (`grep -c . FILE`). Before submitting, set `#SBATCH --account=<project>@h100`
in the launchers, and adapt the partition, the time limit and the module line to your cluster. The launchers set
`HF_HOME=$WORK/hf_cache` (`$WORK` is the work space of Jean-Zay) and the offline mode of the Hugging Face libraries;
`launch_aligned.sh` reads the prepared data from `$DATA_DIR` (default `$SCRATCH/yelp_aligned`).

## Run sets behind the paper

Figure and table numbers are those of the paper. n_tr counts the training and validation subsets together; the
study set, in parentheses, has 1.25 n_tr reviews.

| Paper | Script and launcher | Configuration file (lines) | Models | n_tr (study set) | Seeds | Splits | Commit of the runs |
|---|---|---|---|---|---|---|---|
| Figures 5 (right), 6 (Yelp points), 7 (n_tr = 3,000) and D.1; Tables E.2, M.1 (Yelp rows) and F.2 | `nlpfinetuning_aligned.py`, `launch_aligned.sh` | `aligned_bert.txt` (100), `aligned_roberta.txt` (100) | BERT, XLM-RoBERTa | 1,000 and 3,000 (1,250 and 3,750) | 0-49 | 20; the first is the single split | `8151be5` (script `ed00146`), November 2026 |
| Figure 5 (right, Qwen3-0.6B), Figure D.1, Tables E.2 and F.2 | `nlpfinetuning_full_adapted.py`, `launch.sh` | `qwen_0.6b_1000.txt` (50): K-split runs; `qwen_0.6b_1000_nocv.txt` (50): single-split runs | Qwen3-0.6B | 1,000 (1,250) | 0-49 | 20; 1 | script `6a405c4` (2026-03-28), configurations `8605250` and `8abcea4`; runs of June and July 2026 |

Both scripts are unchanged since these runs, so the files of `configs/` reproduce them with the current code and the
pinned environments (GPU training is not bitwise deterministic, so scores match in distribution, not to the last
digit). The run outputs are not part of this repository.

## License

BSD-3-Clause (see `LICENSE`). The dataset and the checkpoints keep their own terms (see Data).
