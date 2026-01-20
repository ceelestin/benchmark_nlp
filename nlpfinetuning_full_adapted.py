import gc
import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    parser.add_argument(
        "--disable-cross-validation",
        action="store_true",
        help="If set, disables cross-validation (overriding --n-splits to 1) and evaluates on partitioned hidden test sets."
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="If set, skips tokenization and evaluation on the fixed benchmarking set."
    )
    args = parser.parse_args()
    return args


args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
print(f"Cross-Validation Status: {'Disabled' if args.disable_cross_validation else 'Enabled'}")

MODEL_HUGGINGFACE_IDENTIFIERS = {
    "bert": "google-bert/bert-base-cased",
    "roberta": "FacebookAI/xlm-roberta-base",
    "t5": "google-t5/t5-base",
    "modernbert": "answerdotai/ModernBERT-base"
}

print("Beginning the script with the following configurations:")
print(f"  GPU ID (CUDA_VISIBLE_DEVICES): {args.gpu_id}")
print(f"  n_splits values: {args.n_splits if not args.disable_cross_validation else '[1 (Fixed)]'}")
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
        f"Sees {torch.cuda.device_count()} CUDA device(s)."
    )
else:
    device = torch.device("cpu")
    print(f"CUDA not available. Using CPU: {device}")

print("Loading dataset...", end="", flush=True)
raw_dataset_full = load_dataset(
    "yelp_review_full", download_mode="reuse_cache_if_exists"
)
num_labels_for_classification = raw_dataset_full["train"].features["label"].num_classes
print("done")

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# --- Main Experiment Loop ---
all_results_data = []
for model_arg_choice_iter in args.model_choices:
    CURRENT_MODEL_HF_NAME = MODEL_HUGGINGFACE_IDENTIFIERS[model_arg_choice_iter]
    tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL_HF_NAME)
    print(f"\n\n#################### Processing Model: {model_arg_choice_iter} ({CURRENT_MODEL_HF_NAME}) ####################")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="longest", truncation=True, max_length=512)

    for current_study_set_size in study_set_sizes_to_test:
        print(f"\n================ Model {model_arg_choice_iter} - Study Set Size: {current_study_set_size} ================")

        # If CV is disabled, we only run one split. Otherwise, we iterate over the requested n_splits.
        current_n_splits_list = [1] if args.disable_cross_validation else n_splits_values_to_test

        for current_n_splits_value in current_n_splits_list:
            if not args.disable_cross_validation:
                print(f"\n============= Model {model_arg_choice_iter}, Size {current_study_set_size} - n_splits: {current_n_splits_value} =============")
            else:
                print(f"\n============= Model {model_arg_choice_iter}, Size {current_study_set_size} - CV Disabled (Hidden Set Evaluation Mode) =============")

            for current_seed in seeds_range:
                print(f"\n========== Seed: {current_seed} ==========")
                np.random.seed(current_seed)
                random.seed(current_seed)
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)

                original_train_set = raw_dataset_full["train"].shuffle(seed=current_seed)

                # --- Dataset Partitioning ---
                # 1. Benchmarking Set (Fixed size)
                # 2. Study Set (Variable size)
                # 3. Hidden Test Set (Remaining samples, only used if CV is disabled)

                if len(original_train_set) < benchmarking_set_size + current_study_set_size:
                    print(f"WARNING: Original train set too small. Skipping.")
                    continue

                benchmarking_set_raw = original_train_set.select(range(benchmarking_set_size))
                study_set_raw = original_train_set.select(range(benchmarking_set_size, benchmarking_set_size + current_study_set_size))

                # Tokenize generic sets
                benchmarking_set = None
                if not args.skip_benchmark:
                    benchmarking_set_tokenized = benchmarking_set_raw.map(tokenize_function, batched=True, remove_columns=['text'], load_from_cache_file=True)
                    benchmarking_set = benchmarking_set_tokenized.rename_column("label", "labels") if "label" in benchmarking_set_tokenized.features else benchmarking_set_tokenized

                study_set_tokenized = study_set_raw.map(tokenize_function, batched=True, remove_columns=['text'], load_from_cache_file=True)
                study_set_for_cv = study_set_tokenized.rename_column("label", "labels") if "label" in study_set_tokenized.features else study_set_tokenized

                # Logic for Hidden Test Set (if NO CV)
                hidden_test_set_tokenized = None
                if args.disable_cross_validation:
                    start_hidden_index = benchmarking_set_size + current_study_set_size
                    # Select everything remaining
                    if start_hidden_index < len(original_train_set):
                        hidden_test_set_raw = original_train_set.select(range(start_hidden_index, len(original_train_set)))
                        print(f"Preparing Hidden Test Set ({len(hidden_test_set_raw)} samples)...")
                        hidden_test_set_tokenized = hidden_test_set_raw.map(tokenize_function, batched=True, remove_columns=['text'], load_from_cache_file=True)
                        hidden_test_set_tokenized = hidden_test_set_tokenized.rename_column("label", "labels") if "label" in hidden_test_set_tokenized.features else hidden_test_set_tokenized
                    else:
                         print("WARNING: No data left for Hidden Test Set.")

                min_samples_for_split = max(1, math.ceil(1/cv_test_size), math.ceil(1/(1-cv_test_size))) if 0 < cv_test_size < 1 else 2
                if len(study_set_for_cv) < min_samples_for_split and current_n_splits_value > 0:
                    print(f"WARNING: Study set too small for splits. Skipping seed.")
                    continue

                shuffle_splitter = ShuffleSplit(n_splits=current_n_splits_value, test_size=cv_test_size, random_state=current_seed)
                fold_iteration_successful = False

                for fold_idx, (train_indices_temp, test_indices) in enumerate(shuffle_splitter.split(study_set_for_cv)):
                    # 1. ShuffleSplit splits Study Set into:
                    #    - 80% (train_indices_temp) -> Candidate for Training
                    #    - 20% (test_indices)       -> Study Test Set (Unseen data for final scoring)

                    print(f"\n------- Fold {fold_idx + 1}/{current_n_splits_value} -------")

                    # 2. Split the 80% Candidate Training set further:
                    #    - 75% of it becomes the actual Training Set.
                    #    - 25% of it becomes the Validation Set (used for early stopping/best model).
                    #    Ratios of Total:
                    #       Training: 0.8 * 0.75 = 0.60 (60%)
                    #       Validation: 0.8 * 0.25 = 0.20 (20%)
                    #       Test: 0.20 (20%)

                    if len(train_indices_temp) > 1:
                        train_indices, val_indices = train_test_split(
                            train_indices_temp,
                            test_size=0.25,
                            random_state=current_seed + fold_idx # Deterministic split based on seed
                        )
                    else:
                        train_indices = train_indices_temp
                        val_indices = []

                    train_fold_dataset = study_set_for_cv.select(train_indices)
                    validation_fold_dataset = study_set_for_cv.select(val_indices)
                    test_fold_dataset = study_set_for_cv.select(test_indices)

                    if len(train_fold_dataset) == 0:
                        continue

                    # Handle empty validation for tiny sets
                    if len(validation_fold_dataset) == 0 and len(test_fold_dataset) > 0:
                        validation_fold_dataset = test_fold_dataset # Fallback

                    print(f"Train size: {len(train_fold_dataset)}, Validation size: {len(validation_fold_dataset)}, Test (Study) size: {len(test_fold_dataset)}")
                    fold_iteration_successful = True

                    training_config_name = "full_finetune"

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
                        save_total_limit=1,
                        save_only_model=True,
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
                        eval_dataset=validation_fold_dataset, # Use Validation set for training loop evaluation
                        compute_metrics=compute_metrics_fn,
                        tokenizer=tokenizer
                    )

                    current_train_runtime_val = None
                    try:
                        train_output_obj = trainer_obj.train()
                        if train_output_obj and hasattr(train_output_obj, 'metrics') and isinstance(train_output_obj.metrics, dict):
                            current_train_runtime_val = train_output_obj.metrics.get('train_runtime')
                        print(f"Training completed. Runtime: {current_train_runtime_val if current_train_runtime_val is not None else 'N/A'}s")

                        # Evaluate on all sets
                        eval_results_validation = trainer_obj.evaluate(eval_dataset=validation_fold_dataset)
                        eval_results_study_test = trainer_obj.evaluate(eval_dataset=test_fold_dataset)
                        eval_results_benchmark_set = {}
                        if not args.skip_benchmark and benchmarking_set is not None:
                            eval_results_benchmark_set = trainer_obj.evaluate(eval_dataset=benchmarking_set)

                        # --- Hidden Test Set Evaluation (If applicable) ---
                        hidden_test_accuracies = []
                        hidden_test_losses = []

                        if args.disable_cross_validation and hidden_test_set_tokenized is not None:
                            study_test_size = len(test_fold_dataset) # Use Test set size as chunk size
                            if study_test_size > 0:
                                print(f"Evaluating on hidden test subsets of size {study_test_size}...")
                                # We use trainer.predict for speed on larger hidden sets
                                # But we iterate in chunks to mimic the 'subset' logic/evaluation

                                # To be efficient, we get all predictions then slice locally
                                # Note: Predicting on huge dataset might OOM if we keep all logits.
                                # With 50k samples approx, typical system handles it.
                                # If it's huge, we should iterate dataset.

                                hidden_len = len(hidden_test_set_tokenized)
                                # Iterate through the hidden set in chunks of size `study_test_size`
                                for start_idx in range(0, hidden_len, study_test_size):
                                    end_idx = start_idx + study_test_size
                                    if end_idx > hidden_len:
                                        # "each being the same size as the test set" -> Discard incomplete final chunk
                                        break

                                    subset_dataset = hidden_test_set_tokenized.select(range(start_idx, end_idx))
                                    # result includes: test_loss, test_accuracy, test_runtime, etc.
                                    subset_result = trainer_obj.evaluate(eval_dataset=subset_dataset)

                                    hidden_test_accuracies.append(subset_result.get('eval_accuracy'))
                                    hidden_test_losses.append(subset_result.get('eval_loss'))

                                print(f"Evaluated on {len(hidden_test_accuracies)} hidden subsets.")

                        result_entry = {
                            "model_arg_choice": model_arg_choice_iter,
                            "model_hf_name": CURRENT_MODEL_HF_NAME,
                            "study_set_size": current_study_set_size,
                            "n_splits_cv": current_n_splits_value if not args.disable_cross_validation else 1,
                            "seed": current_seed,
                            "fold_index": fold_idx + 1,
                            "training_config": training_config_name,
                            "cross_validation": not args.disable_cross_validation,
                            "skipped_benchmark": args.skip_benchmark,
                            "train_runtime": current_train_runtime_val,

                            # Validation Set Metrics (Used for model selection)
                            "validation_eval_loss": eval_results_validation.get('eval_loss'),
                            "validation_eval_accuracy": eval_results_validation.get('eval_accuracy'),
                            "validation_eval_runtime": eval_results_validation.get('eval_runtime'),
                            "epoch_at_best_validation": eval_results_validation.get('epoch'),

                            # Study Test Set Metrics (Unseen data)
                            "study_test_eval_loss": eval_results_study_test.get('eval_loss'),
                            "study_test_eval_accuracy": eval_results_study_test.get('eval_accuracy'),
                            "study_test_eval_runtime": eval_results_study_test.get('eval_runtime'),

                            # Benchmarking Set Metrics (Fixed set), None if skipped
                            "benchmark_eval_loss": eval_results_benchmark_set.get('eval_loss'),
                            "benchmark_eval_accuracy": eval_results_benchmark_set.get('eval_accuracy'),
                            "benchmark_eval_runtime": eval_results_benchmark_set.get('eval_runtime'),

                            # Store hidden results as lists. They will be None or empty if CV is enabled.
                            "hidden_test_accuracies": hidden_test_accuracies if hidden_test_accuracies else None,
                            "hidden_test_losses": hidden_test_losses if hidden_test_losses else None
                        }
                        all_results_data.append(result_entry)

                    except Exception as e:
                        print(f"ERROR: {e}")
                        error_entry = {
                            "model_arg_choice": model_arg_choice_iter,
                            "model_hf_name": CURRENT_MODEL_HF_NAME,
                            "study_set_size": current_study_set_size,
                            "seed": current_seed, "fold_index": fold_idx + 1,
                            "training_config": training_config_name,
                            "train_runtime": current_train_runtime_val, "error": str(e)
                        }
                        all_results_data.append(error_entry)
                    finally:
                        del model_instance, trainer_obj
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                if not fold_iteration_successful:
                     print("WARNING: No valid folds.")

print("\n--- All Benchmarking Runs Finished ---")
results_df = pd.DataFrame(all_results_data)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
models_str = "_".join(args.model_choices)
n_splits_str = "_".join(map(str, args.n_splits if not args.disable_cross_validation else [1]))
seeds_str = f"{args.seeds[0]}-{args.seeds[1]-1}"
study_sizes_str = "_".join(map(str, args.study_sizes))
cv_status_str = "no_cv" if args.disable_cross_validation else "cv"
args_desc = f"{cv_status_str}_models_{models_str}_nsplits_{n_splits_str}_seeds_{seeds_str}_sizes_{study_sizes_str}"
parquet_file_path = OUTPUT_DIR / "parquet" / f"benchmark_results_{args_desc}_{timestamp}.parquet"

if not results_df.empty:
    parquet_file_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(parquet_file_path)
    print(f"\nResults saved to {parquet_file_path}")

    # Optional: Basic print of results
    if args.disable_cross_validation:
         print("\n--- Hidden Test Set stats for first successful run ---")
         # Just verify the first valid entry with hidden scores
         valid_entry = next((item for item in all_results_data if item.get("hidden_test_accuracies")), None)
         if valid_entry:
             accs = valid_entry["hidden_test_accuracies"]
             print(f"Number of hidden subsets evaluated: {len(accs)}")
             print(f"Mean hidden subset accuracy: {np.mean(accs):.4f}")
else:
    print("No results collected.")
print("\nScript completed.")
