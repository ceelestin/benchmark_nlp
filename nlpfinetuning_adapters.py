import os
# GPU ID will be set after parsing arguments, before most torch imports.
# TOKENIZERS_PARALLELISM is set early.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse # For command-line arguments
import math # For math.ceil
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import ShuffleSplit
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.trainer_utils import TrainOutput # To help with type hinting if needed
import torch
import pandas as pd
from peft import (
    LoraConfig,
    IA3Config,
    AdaLoraConfig,
    LoHaConfig,
    LoKrConfig,
    VeraConfig,
    get_peft_model,
    TaskType
)
import random
from collections import namedtuple # For understanding TrainOutput structure

# --- Training Constants (ensure these match TrainingArguments) ---
NUM_TRAIN_EPOCHS_FOR_ADAPTERS = 3
PER_DEVICE_TRAIN_BATCH_SIZE_FOR_ADAPTERS = 32

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run NLP fine-tuning benchmarks with various configurations.")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default="0",
        help="GPU ID to use (e.g., '0', '0,1'). Sets CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        nargs='+',
        default=[2, 3],
        help="List of n_splits values for ShuffleSplit cross-validation (e.g., --n-splits 2 5 10)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=2,
        default=[0, 1], # Default: seed 0
        metavar=('START_SEED', 'END_SEED_EXCLUSIVE'),
        help="Range of seeds to study (e.g., --seeds 0 5 for seeds 0, 1, 2, 3, 4)"
    )
    parser.add_argument(
        "--study-sizes",
        type=int,
        nargs='+',
        default=[100, 500],
        help="List of study set sizes to test (e.g., --study-sizes 100 500 1000)"
    )
    args = parser.parse_args()
    return args

args = parse_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

print("Beginning the script with the following configurations:")
print(f"  GPU ID (CUDA_VISIBLE_DEVICES): {args.gpu_id}")
print(f"  n_splits values: {args.n_splits}")
print(f"  Seed range: from {args.seeds[0]} to {args.seeds[1]-1}")
print(f"  Study set sizes: {args.study_sizes}")

n_splits_values_to_test = args.n_splits
seeds_range = range(args.seeds[0], args.seeds[1])
study_set_sizes_to_test = args.study_sizes

benchmarking_set_size = 600_000
cv_test_size = 0.2

if "CUDA_VISIBLE_DEVICES" in os.environ:
    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"CUDA_VISIBLE_DEVICES explicitly set to '{cvd}'. Physical GPU(s) {cvd} should now be visible to PyTorch.")
else:
    print("CUDA_VISIBLE_DEVICES not set. PyTorch will use its default GPU selection logic.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is using device: {device} (corresponds to physical GPU(s) {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}).")
    print(f"PyTorch sees {torch.cuda.device_count()} CUDA device(s).")
else:
    device = torch.device("cpu")
    print(f"CUDA not available. Using CPU: {device}")

raw_dataset_full = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
MODEL_NAME = "google-bert/bert-base-cased"

def tokenize(examples):
    return tokenizer(examples["text"], padding="longest", truncation=True, max_length=512)

# --- PEFT Configurations ---
common_attn_target_modules = ["query", "key", "value"]
common_ffn_target_modules = ["intermediate.dense", "output.dense"]

lora_config_default = LoraConfig(
    r=16, lora_alpha=32, target_modules=common_attn_target_modules,
    lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS
)
ia3_config_default = IA3Config(
    target_modules=common_attn_target_modules + common_ffn_target_modules,
    feedforward_modules=common_ffn_target_modules,
    task_type=TaskType.SEQ_CLS
)
loha_config_default = LoHaConfig(
    task_type=TaskType.SEQ_CLS, r=16, alpha=32,
    target_modules=common_attn_target_modules,
    rank_dropout=0.0, module_dropout=0.0,
)
lokr_config_default = LoKrConfig(
    task_type=TaskType.SEQ_CLS, r=16, alpha=32,
    target_modules=common_attn_target_modules,
    rank_dropout=0.0, module_dropout=0.0,
)
vera_config_default = VeraConfig(
    task_type=TaskType.SEQ_CLS, r=16,
    target_modules=common_attn_target_modules,
    vera_dropout=0.05,
)

model_configurations = {
    "no_adapter": None,
    "lora_default": lora_config_default,
    "ia3_default": ia3_config_default,
    "adalora": "CREATE_ADALORA_DYNAMICALLY",
    # "loha_default": loha_config_default,
    "lokr_default": lokr_config_default,
    # "vera_default": vera_config_default,
}

all_results_data = []
metric_accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_accuracy.compute(predictions=predictions, references=labels)

# --- Main Experiment Loop ---
for current_study_set_size in study_set_sizes_to_test:
    print(f"\n\n================ Running for Study Set Size: {current_study_set_size} ================")
    for current_n_splits_value in n_splits_values_to_test:
        print(f"\n\n============= Study Size {current_study_set_size} - Running for n_splits: {current_n_splits_value} =============")
        for current_seed in seeds_range:
            print(f"\n\n========== Study Size {current_study_set_size}, n_splits {current_n_splits_value} - Running for Seed: {current_seed} ==========")
            np.random.seed(current_seed)
            random.seed(current_seed)
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)

            original_train_set = raw_dataset_full["train"].shuffle(seed=current_seed)
            if len(original_train_set) < benchmarking_set_size + current_study_set_size:
                print(f"WARNING: Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed}: Original train set too small. Skipping.")
                continue

            benchmarking_set_raw = original_train_set.select(range(benchmarking_set_size))
            study_set_raw = original_train_set.select(
                range(benchmarking_set_size, benchmarking_set_size + current_study_set_size)
            )

            benchmarking_set_tokenized = benchmarking_set_raw.map(tokenize, batched=True, remove_columns=['text'], load_from_cache_file=False)
            benchmarking_set = benchmarking_set_tokenized.rename_column("label", "labels") if "label" in benchmarking_set_tokenized.features and "labels" not in benchmarking_set_tokenized.features else benchmarking_set_tokenized

            study_set_tokenized = study_set_raw.map(tokenize, batched=True, remove_columns=['text'], load_from_cache_file=False)
            study_set_for_cv = study_set_tokenized.rename_column("label", "labels") if "label" in study_set_tokenized.features and "labels" not in study_set_tokenized.features else study_set_tokenized

            min_samples_for_split_calc = 1
            if current_n_splits_value > 0:
                 min_samples_for_split_calc = max(1, math.ceil(1/cv_test_size), math.ceil(1/(1-cv_test_size))) if cv_test_size > 0 and cv_test_size < 1 else 2

            if len(study_set_for_cv) < min_samples_for_split_calc and current_n_splits_value > 0 :
                 print(f"WARNING: Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed}: Study set for CV has {len(study_set_for_cv)} samples. "
                       f"Requires at least {min_samples_for_split_calc} for {current_n_splits_value} splits with test_size {cv_test_size}. Skipping seed.")
                 continue

            shuffle_splitter = ShuffleSplit(n_splits=current_n_splits_value, test_size=cv_test_size, random_state=current_seed)
            fold_iteration_successful = False
            for fold_idx, (train_indices, val_indices) in enumerate(shuffle_splitter.split(study_set_for_cv)):
                print(f"\n------- Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed} - Fold {fold_idx + 1}/{current_n_splits_value} -------")
                train_fold_dataset = study_set_for_cv.select(train_indices)
                eval_fold_dataset = study_set_for_cv.select(val_indices)

                if len(train_fold_dataset) == 0 or len(eval_fold_dataset) == 0:
                    print(f"WARNING: Fold {fold_idx+1} has empty train ({len(train_fold_dataset)}) or eval ({len(eval_fold_dataset)}) set. Skipping fold.")
                    continue

                print(f"Train fold size: {len(train_fold_dataset)}, Eval fold size: {len(eval_fold_dataset)}")
                fold_iteration_successful = True

                for config_name, peft_config_template in model_configurations.items():
                    print(f"\n===== Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed}, Fold {fold_idx+1} - Model: {config_name} =====")

                    actual_peft_config = None
                    if config_name == "adalora" and peft_config_template == "CREATE_ADALORA_DYNAMICALLY":
                        _num_gpus_for_calc = torch.cuda.device_count() if torch.cuda.is_available() else 0
                        _actual_num_gpus_for_calc = max(1, _num_gpus_for_calc)

                        _steps_per_epoch = math.ceil(len(train_fold_dataset) / (PER_DEVICE_TRAIN_BATCH_SIZE_FOR_ADAPTERS * _actual_num_gpus_for_calc))
                        _total_training_steps = int(_steps_per_epoch * NUM_TRAIN_EPOCHS_FOR_ADAPTERS)

                        if _total_training_steps <= 0:
                            print(f"WARNING: AdaLoRA total_step calculation for {config_name} resulted in {_total_training_steps} "
                                  f"for train set size {len(train_fold_dataset)}. Skipping this config for this fold.")
                            continue
                        actual_peft_config = AdaLoraConfig(
                            task_type=TaskType.SEQ_CLS, r=16, target_r=8, init_r=12, lora_alpha=32,
                            target_modules=common_attn_target_modules + ["output.dense"],
                            lora_dropout=0.05,
                            total_step=_total_training_steps
                        )
                    elif isinstance(peft_config_template, (LoraConfig, IA3Config, AdaLoraConfig, LoHaConfig, LoKrConfig, VeraConfig)):
                        actual_peft_config = peft_config_template
                    elif peft_config_template is None and config_name == "no_adapter":
                        actual_peft_config = None
                    else:
                        print(f"ERROR: Unrecognized peft_config_template for {config_name}: {peft_config_template}. Skipping.")
                        continue

                    print(f"Initializing base model for {config_name}...")
                    model = AutoModelForSequenceClassification.from_pretrained(
                        MODEL_NAME,
                        num_labels=raw_dataset_full["train"].features["label"].num_classes
                    )

                    if actual_peft_config:
                        print(f"Applying PEFT adapter: {config_name}")
                        model = get_peft_model(model, actual_peft_config)
                        model.print_trainable_parameters()

                    model.to(device)

                    num_gpus_runtime = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    logging_steps_value = max(1, len(train_fold_dataset) // (PER_DEVICE_TRAIN_BATCH_SIZE_FOR_ADAPTERS * max(1, num_gpus_runtime) * 4))

                    training_args = TrainingArguments(
                        output_dir=f"results_output/size_{current_study_set_size}/nsplits_{current_n_splits_value}/seed_{current_seed}/fold_{fold_idx+1}/{config_name}",
                        eval_strategy="epoch",
                        num_train_epochs=NUM_TRAIN_EPOCHS_FOR_ADAPTERS,
                        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE_FOR_ADAPTERS,
                        per_device_eval_batch_size=128,
                        logging_steps=logging_steps_value,
                        save_strategy="epoch",
                        dataloader_num_workers=10,
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_accuracy",
                        push_to_hub=False,
                        label_names=["labels"],
                        report_to="none",
                        fp16=torch.cuda.is_available(),
                        seed=current_seed
                    )

                    trainer = Trainer(
                        model=model, args=training_args, train_dataset=train_fold_dataset,
                        eval_dataset=eval_fold_dataset, compute_metrics=compute_metrics, tokenizer=tokenizer
                    )

                    print(f"Starting training for {config_name}...")
                    train_output = None
                    current_train_runtime = None
                    try:
                        train_output = trainer.train()
                        if train_output and hasattr(train_output, 'metrics') and isinstance(train_output.metrics, dict):
                            current_train_runtime = train_output.metrics.get('train_runtime')
                        print(f"Training completed for {config_name}. Runtime: {current_train_runtime if current_train_runtime is not None else 'N/A'}s")

                        eval_results_fold_val = trainer.evaluate(eval_dataset=eval_fold_dataset)
                        print(f"{config_name} fold validation set results: {eval_results_fold_val}")
                        eval_results_benchmark = trainer.evaluate(eval_dataset=benchmarking_set)
                        print(f"{config_name} benchmarking set results: {eval_results_benchmark}")

                        config_data = {
                            "study_set_size": current_study_set_size, "n_splits_cv": current_n_splits_value,
                            "seed": current_seed, "fold_index": fold_idx + 1, "model_config": config_name,
                            "train_runtime": current_train_runtime,
                            "fold_val_eval_loss": eval_results_fold_val.get('eval_loss'),
                            "fold_val_eval_accuracy": eval_results_fold_val.get('eval_accuracy'),
                            "fold_val_eval_runtime": eval_results_fold_val.get('eval_runtime'),
                            "benchmark_eval_loss": eval_results_benchmark.get('eval_loss'),
                            "benchmark_eval_accuracy": eval_results_benchmark.get('eval_accuracy'),
                            "benchmark_eval_runtime": eval_results_benchmark.get('eval_runtime'),
                            "epoch_at_best_fold_val": eval_results_fold_val.get('epoch'),
                        }
                        all_results_data.append(config_data)
                    except Exception as e:
                        print(f"ERROR during training/evaluation for Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed}, Fold {fold_idx+1}, Config {config_name}: {e}")
                        error_train_runtime = None
                        if train_output and hasattr(train_output, 'metrics') and isinstance(train_output.metrics, dict):
                            error_train_runtime = train_output.metrics.get('train_runtime')
                        error_data = {
                            "study_set_size": current_study_set_size, "n_splits_cv": current_n_splits_value,
                            "seed": current_seed, "fold_index": fold_idx + 1, "model_config": config_name,
                            "train_runtime": error_train_runtime, "error": str(e)
                        }
                        all_results_data.append(error_data)
                    finally:
                        del model, trainer
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

            if not fold_iteration_successful:
                print(f"WARNING: No valid folds processed for Size {current_study_set_size}, n_splits {current_n_splits_value}, Seed {current_seed}.")

print("\n--- All Benchmarking Runs Finished ---")
results_df = pd.DataFrame(all_results_data)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
n_splits_str = "_".join(map(str, args.n_splits))
seeds_str = f"{args.seeds[0]}-{args.seeds[1]-1}"
study_sizes_str = "_".join(map(str, args.study_sizes))
args_desc = f"nsplits_{n_splits_str}_seeds_{seeds_str}_sizes_{study_sizes_str}"
parquet_file_path = f"benchmark_results_cv_{args_desc}_{timestamp}.parquet"
results_df.to_parquet(parquet_file_path)
print(f"\nResults saved to {parquet_file_path}")

if not results_df.empty:
    print("\n--- Results (from DataFrame) ---")
    results_df_sorted = results_df.sort_values(by=['study_set_size', 'n_splits_cv', 'seed', 'fold_index', 'model_config'])
    for size_val in sorted(results_df_sorted['study_set_size'].unique()):
        print(f"\n==================== Study Set Size: {size_val} ====================")
        size_df = results_df_sorted[results_df_sorted['study_set_size'] == size_val]
        for n_splits_val in sorted(size_df['n_splits_cv'].unique()):
            print(f"\n  ------------------ n_splits_cv: {n_splits_val} ------------------")
            nsplits_df = size_df[size_df['n_splits_cv'] == n_splits_val]
            for seed_val in sorted(nsplits_df['seed'].unique()):
                print(f"\n    ========== Seed: {seed_val} ==========")
                seed_df_subset = nsplits_df[nsplits_df['seed'] == seed_val]
                for fold_val in sorted(seed_df_subset['fold_index'].unique()):
                    print(f"\n      ~~~~~~~~ Fold: {fold_val} ~~~~~~~~")
                    fold_df_subset = seed_df_subset[seed_df_subset['fold_index'] == fold_val]
                    for index, row in fold_df_subset.iterrows():
                        train_rt_val = row.get('train_runtime')
                        if 'error' in row and pd.notna(row['error']):
                            print(f"\n        --- Configuration: {row['model_config']} --- ERROR: {row['error']}")
                            if pd.notna(train_rt_val): print(f"          Training Runtime (before error): {train_rt_val:.2f}s")
                        else:
                            print(f"\n        --- Configuration: {row['model_config']} ---")
                            if pd.notna(train_rt_val): print(f"          Training Runtime: {train_rt_val:.2f}s")

                            val_acc = row.get('fold_val_eval_accuracy')
                            val_loss = row.get('fold_val_eval_loss')
                            val_rt = row.get('fold_val_eval_runtime')

                            val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else "N/A"
                            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else "N/A"
                            val_rt_str = f"{val_rt:.2f}" if isinstance(val_rt, float) else "N/A"

                            print(f"          Fold Validation Set: Accuracy = {val_acc_str}, Loss = {val_loss_str}, Runtime = {val_rt_str}s")

                            bench_acc = row.get('benchmark_eval_accuracy')
                            bench_loss = row.get('benchmark_eval_loss')
                            bench_rt = row.get('benchmark_eval_runtime')

                            bench_acc_str = f"{bench_acc:.4f}" if isinstance(bench_acc, float) else "N/A"
                            bench_loss_str = f"{bench_loss:.4f}" if isinstance(bench_loss, float) else "N/A"
                            bench_rt_str = f"{bench_rt:.2f}" if isinstance(bench_rt, float) else "N/A"

                            print(f"          Benchmarking Set: Accuracy = {bench_acc_str}, Loss = {bench_loss_str}, Runtime = {bench_rt_str}s")
                            print(f"          Epoch for Best Fold Validation Model: {row.get('epoch_at_best_fold_val', 'N/A')}")
else:
    print("No results collected to summarize or save.")
print("\nScript completed.")