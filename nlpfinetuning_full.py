import gc
import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- Training Constants ---
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 32

OUTPUT_DIR = Path("results_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run NLP full fine-tuning benchmarks for specified models."
    )
    parser.add_argument(
        "--gpu-id", type=str, default="0", help="GPU ID to use."
    )
    parser.add_argument(
        "--n-splits", type=int, nargs='+', default=[1],
        help="List of n_splits values for ShuffleSplit."
    )
    parser.add_argument(
        "--seeds", type=int, nargs=2, default=[0, 1],
        metavar=('START_SEED', 'END_SEED_EXCLUSIVE'), help="Range of seeds."
    )
    parser.add_argument(
        "--study-sizes", type=int, nargs='+', default=[100, 500],
        help="List of study set sizes."
    )
    parser.add_argument(
        "--model-choices",
        type=str,
        nargs='+',
        default=["bert"],
        choices=["bert", "roberta", "t5", "modernbert"],
        help="Choose one or more models to run full fine-tuning on."
    )
    args = parser.parse_args()
    return args


args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

MODEL_HUGGINGFACE_IDENTIFIERS = {
    "bert": "google-bert/bert-base-cased",
    "roberta": "FacebookAI/xlm-roberta-base",
    "t5": "google-t5/t5-base",
    "modernbert": "answerdotai/ModernBERT-base"
}

print("Beginning the script with the following configurations:")
print(f"  GPU ID (CUDA_VISIBLE_DEVICES): {args.gpu_id}")
print(f"  n_splits values: {args.n_splits}")
print(f"  Seed range: from {args.seeds[0]} to {args.seeds[1]-1}")
print(f"  Study set sizes: {args.study_sizes}")
print(f"  Models selected for full fine-tuning: {args.model_choices}")

n_splits_values_to_test = args.n_splits
seeds_range = range(args.seeds[0], args.seeds[1])
study_set_sizes_to_test = args.study_sizes

benchmarking_set_size = 600_000
cv_test_size = 0.2

if "CUDA_VISIBLE_DEVICES" in os.environ:
    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"CUDA_VISIBLE_DEVICES explicitly set to '{cvd}'.")
else:
    print("CUDA_VISIBLE_DEVICES not set.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(
        f"PyTorch is using device: {device}. "
        "Sees {torch.cuda.device_count()} CUDA device(s)."
    )
else:
    device = torch.device("cpu")
    print(f"CUDA not available. Using CPU: {device}")

raw_dataset_full = load_dataset("yelp_review_full")
num_labels_for_classification = raw_dataset_full["train"].features["label"].num_classes

all_results_data = []
metric_accuracy = evaluate.load("accuracy")

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_accuracy.compute(predictions=predictions, references=labels)

# --- Main Experiment Loop ---
for model_arg_choice_iter in args.model_choices:
    CURRENT_MODEL_HF_NAME = MODEL_HUGGINGFACE_IDENTIFIERS[model_arg_choice_iter]
    tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL_HF_NAME)
    print(f"\n\n#################### Processing Model: {model_arg_choice_iter} ({CURRENT_MODEL_HF_NAME}) ####################")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="longest", truncation=True, max_length=512)

    for current_study_set_size in study_set_sizes_to_test:
        print(f"\n================ Model {model_arg_choice_iter} - Study Set Size: {current_study_set_size} ================")
        for current_n_splits_value in n_splits_values_to_test:
            print(f"\n============= Model {model_arg_choice_iter}, Size {current_study_set_size} - n_splits: {current_n_splits_value} =============")
            for current_seed in seeds_range:
                print(f"\n========== Model {model_arg_choice_iter}, Size {current_study_set_size}, n_splits {current_n_splits_value} - Seed: {current_seed} ==========")
                np.random.seed(current_seed)
                random.seed(current_seed)
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)

                original_train_set = raw_dataset_full["train"].shuffle(seed=current_seed)
                if len(original_train_set) < benchmarking_set_size + current_study_set_size:
                    print(f"WARNING: Original train set too small for model {model_arg_choice_iter}, size {current_study_set_size}, seed {current_seed}. Skipping.")
                    continue

                benchmarking_set_raw = original_train_set.select(range(benchmarking_set_size))
                study_set_raw = original_train_set.select(range(benchmarking_set_size, benchmarking_set_size + current_study_set_size))

                benchmarking_set_tokenized = benchmarking_set_raw.map(tokenize_function, batched=True, remove_columns=['text'], load_from_cache_file=True)
                benchmarking_set = benchmarking_set_tokenized.rename_column("label", "labels") if "label" in benchmarking_set_tokenized.features else benchmarking_set_tokenized

                study_set_tokenized = study_set_raw.map(tokenize_function, batched=True, remove_columns=['text'], load_from_cache_file=True)
                study_set_for_cv = study_set_tokenized.rename_column("label", "labels") if "label" in study_set_tokenized.features else study_set_tokenized

                min_samples_for_split = max(1, math.ceil(1/cv_test_size), math.ceil(1/(1-cv_test_size))) if 0 < cv_test_size < 1 else 2
                if len(study_set_for_cv) < min_samples_for_split and current_n_splits_value > 0:
                    print(f"WARNING: Study set for CV too small ({len(study_set_for_cv)} samples) for model {model_arg_choice_iter}, size {current_study_set_size}, seed {current_seed}. Skipping seed.")
                    continue

                shuffle_splitter = ShuffleSplit(n_splits=current_n_splits_value, test_size=cv_test_size, random_state=current_seed)
                fold_iteration_successful = False

                for fold_idx, (train_indices, val_indices) in enumerate(shuffle_splitter.split(study_set_for_cv)):
                    print(f"\n------- Model {model_arg_choice_iter}, Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed} - Fold {fold_idx + 1}/{current_n_splits_value} -------")
                    train_fold_dataset = study_set_for_cv.select(train_indices)
                    eval_fold_dataset = study_set_for_cv.select(val_indices)

                    if not train_fold_dataset or not eval_fold_dataset:
                        print(f"WARNING: Fold {fold_idx+1} has empty train or eval set. Skipping fold.")
                        continue
                    print(f"Train fold size: {len(train_fold_dataset)}, Eval fold size: {len(eval_fold_dataset)}")
                    fold_iteration_successful = True

                    # Only one configuration: full fine-tuning
                    training_config_name = "full_finetune"
                    print(f"\n===== Model {model_arg_choice_iter}, Fold {fold_idx+1}, Config: {training_config_name} =====")

                    model_instance = AutoModelForSequenceClassification.from_pretrained(CURRENT_MODEL_HF_NAME, num_labels=num_labels_for_classification)
                    model_instance.to(device)

                    num_gpus_runtime = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
                    logging_steps_val = max(1, len(train_fold_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE * num_gpus_runtime * 4))

                    output_dir_path = (
                        OUTPUT_DIR / f"model_{model_arg_choice_iter}" /
                        f"size_{current_study_set_size}" /
                        f"nsplits_{current_n_splits_value}" /
                        f"seed_{current_seed}" / f"fold_{fold_idx+1}" /
                        f"{training_config_name}"
                    )

                    training_args_obj = TrainingArguments(
                        output_dir=output_dir_path,
                        eval_strategy="epoch",
                        num_train_epochs=NUM_TRAIN_EPOCHS,
                        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                        per_device_eval_batch_size=128,
                        logging_steps=logging_steps_val,
                        save_strategy="best",
                        save_total_limit=1,  # Limit the number of checkpoints
                        save_only_model=True,  # Limit the checkpoints' size
                        load_best_model_at_end=True,
                        dataloader_num_workers=4,
                        metric_for_best_model="eval_accuracy",
                        push_to_hub=False,
                        label_names=["labels"],
                        report_to="none",
                        fp16=torch.cuda.is_available(),
                        seed=current_seed
                    )

                    trainer_obj = Trainer(
                        model=model_instance,
                        args=training_args_obj,
                        train_dataset=train_fold_dataset,
                        eval_dataset=eval_fold_dataset,
                        compute_metrics=compute_metrics_fn,
                        tokenizer=tokenizer
                    )

                    current_train_runtime_val = None
                    try:
                        train_output_obj = trainer_obj.train()
                        if train_output_obj and hasattr(train_output_obj, 'metrics') and isinstance(train_output_obj.metrics, dict):
                            current_train_runtime_val = train_output_obj.metrics.get('train_runtime')
                        print(f"Training completed. Runtime: {current_train_runtime_val if current_train_runtime_val is not None else 'N/A'}s")

                        eval_results_fold_val_set = trainer_obj.evaluate(eval_dataset=eval_fold_dataset)
                        eval_results_benchmark_set = trainer_obj.evaluate(eval_dataset=benchmarking_set)

                        result_entry = {
                            "model_arg_choice": model_arg_choice_iter,
                            "model_hf_name": CURRENT_MODEL_HF_NAME,
                            "study_set_size": current_study_set_size, "n_splits_cv": current_n_splits_value,
                            "seed": current_seed, "fold_index": fold_idx + 1, "training_config": training_config_name,
                            "train_runtime": current_train_runtime_val,
                            "fold_val_eval_loss": eval_results_fold_val_set.get('eval_loss'),
                            "fold_val_eval_accuracy": eval_results_fold_val_set.get('eval_accuracy'),
                            "fold_val_eval_runtime": eval_results_fold_val_set.get('eval_runtime'),
                            "benchmark_eval_loss": eval_results_benchmark_set.get('eval_loss'),
                            "benchmark_eval_accuracy": eval_results_benchmark_set.get('eval_accuracy'),
                            "benchmark_eval_runtime": eval_results_benchmark_set.get('eval_runtime'),
                            "epoch_at_best_fold_val": eval_results_fold_val_set.get('epoch'),
                        }
                        all_results_data.append(result_entry)
                    except Exception as e:
                        print(f"ERROR during training/evaluation for model {model_arg_choice_iter}, config {training_config_name}: {e}")
                        error_entry = {
                            "model_arg_choice": model_arg_choice_iter,
                            "model_hf_name": CURRENT_MODEL_HF_NAME,
                            "study_set_size": current_study_set_size, "n_splits_cv": current_n_splits_value,
                            "seed": current_seed, "fold_index": fold_idx + 1, "training_config": training_config_name,
                            "train_runtime": current_train_runtime_val, "error": str(e)
                        }
                        all_results_data.append(error_entry)
                    finally:
                        del model_instance, trainer_obj
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                if not fold_iteration_successful:
                    print(f"WARNING: No valid folds processed for model {model_arg_choice_iter}, size {current_study_set_size}, n_splits {current_n_splits_value}, seed {current_seed}.")

print("\n--- All Benchmarking Runs Finished ---")
results_df = pd.DataFrame(all_results_data)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
models_str = "_".join(args.model_choices)
n_splits_str = "_".join(map(str, args.n_splits))
seeds_str = f"{args.seeds[0]}-{args.seeds[1]-1}"
study_sizes_str = "_".join(map(str, args.study_sizes))
args_desc = f"models_{models_str}_nsplits_{n_splits_str}_seeds_{seeds_str}_sizes_{study_sizes_str}"
parquet_file_path = OUTPUT_DIR / "parquet" / f"benchmark_results_cv_{args_desc}_{timestamp}.parquet"

if not results_df.empty:
    parquet_file_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(parquet_file_path)
    print(f"\nResults saved to {parquet_file_path}")
    print("\n--- Results (from DataFrame) ---")
    sort_order = ['model_arg_choice', 'study_set_size', 'n_splits_cv', 'seed', 'fold_index', 'training_config']
    results_df_sorted = results_df.sort_values(by=[col for col in sort_order if col in results_df.columns])

    for model_val, model_group_df in results_df_sorted.groupby('model_arg_choice', sort=False):
        print(f"\n#################### Model Choice: {model_val} ({MODEL_HUGGINGFACE_IDENTIFIERS[model_val]}) ####################")
        for size_val, size_group_df in model_group_df.groupby('study_set_size', sort=False):
            print(f"\n==================== Study Set Size: {size_val} ====================")
            for n_splits_val, nsplits_group_df in size_group_df.groupby('n_splits_cv', sort=False):
                print(f"\n  ------------------ n_splits_cv: {n_splits_val} ------------------")
                for seed_val, seed_group_df in nsplits_group_df.groupby('seed', sort=False):
                    print(f"\n    ========== Seed: {seed_val} ==========")
                    for fold_val, fold_group_df in seed_group_df.groupby('fold_index', sort=False):
                        print(f"\n      ~~~~~~~~ Fold: {fold_val} ~~~~~~~~")
                        for _, row_data in fold_group_df.iterrows():
                            train_conf = row_data.get('training_config', 'N/A')
                            train_rt = row_data.get('train_runtime')
                            print(f"\n        --- Training Config: {train_conf} ---")
                            if 'error' in row_data and pd.notna(row_data['error']):
                                print(f"          ERROR: {row_data['error']}")
                                if pd.notna(train_rt): print(f"          Training Runtime (before error): {train_rt:.2f}s")
                            else:
                                if pd.notna(train_rt): print(f"          Training Runtime: {train_rt:.2f}s")
                                for metric_prefix in ["fold_val", "benchmark"]:
                                    acc = row_data.get(f'{metric_prefix}_eval_accuracy')
                                    loss = row_data.get(f'{metric_prefix}_eval_loss')
                                    rt = row_data.get(f'{metric_prefix}_eval_runtime')
                                    acc_s = f"{acc:.4f}" if isinstance(acc, float) else "N/A"
                                    loss_s = f"{loss:.4f}" if isinstance(loss, float) else "N/A"
                                    rt_s = f"{rt:.2f}" if isinstance(rt, float) else "N/A"
                                    set_name = "Fold Validation Set" if metric_prefix == "fold_val" else "Benchmarking Set"
                                    print(f"          {set_name}: Accuracy = {acc_s}, Loss = {loss_s}, Runtime = {rt_s}s")
                                print(f"          Epoch for Best Fold Validation Model: {row_data.get('epoch_at_best_fold_val', 'N/A')}")
else:
    print("No results collected. Parquet file not saved.")
print("\nScript completed.")
