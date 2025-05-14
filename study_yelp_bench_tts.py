import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Set this BEFORE any torch-related imports

from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split # Changed from RepeatedKFold
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd # For saving to Parquet
from peft import LoraConfig, IA3Config, get_peft_model, TaskType # For PEFT adapters

print("Beginning the script...")
# Corrected print statement to match CUDA_VISIBLE_DEVICES setting
if "CUDA_VISIBLE_DEVICES" in os.environ:
    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"CUDA_VISIBLE_DEVICES set to '{cvd}'. Physical GPU {cvd} should now be visible as 'cuda:0' to PyTorch.")
else:
    print("CUDA_VISIBLE_DEVICES not set. PyTorch will use its default GPU selection logic.")


# Determine the device PyTorch will use
if torch.cuda.is_available():
    device = torch.device("cuda") # This will be cuda:0, which corresponds to physical GPU set by CUDA_VISIBLE_DEVICES
    print(f"PyTorch is using device: {device} (corresponds to physical GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}).")
    print(f"PyTorch sees {torch.cuda.device_count()} CUDA device(s).")
else:
    device = torch.device("cpu")
    print(f"CUDA not available. Using CPU: {device}")


raw_dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
MODEL_NAME = "google-bert/bert-base-cased" # Define model name for reuse

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Prepare datasets
original_train_set = raw_dataset["train"].shuffle(seed=42)

benchmarking_set_size = 600000
study_set_for_train_val_size = 100 # Renamed from study_set_for_cv_size

if len(original_train_set) < benchmarking_set_size + study_set_for_train_val_size:
    raise ValueError(
        f"Original train set (size {len(original_train_set)}) is too small "
        f"for the requested benchmarking_set ({benchmarking_set_size}) "
        f"and study_set_for_train_val ({study_set_for_train_val_size})."
    )

benchmarking_set_raw = original_train_set.select(range(benchmarking_set_size))
study_set_raw = original_train_set.select(
    range(benchmarking_set_size, benchmarking_set_size + study_set_for_train_val_size)
)

print(f"Benchmarking set created with {len(benchmarking_set_raw)} raw samples.")
print(f"Study set for Train/Val created with {len(study_set_raw)} raw samples.")

print("Tokenizing benchmarking set...")
benchmarking_set_tokenized = benchmarking_set_raw.map(tokenize, batched=True, remove_columns=['text'])
if "label" in benchmarking_set_tokenized.features and "labels" not in benchmarking_set_tokenized.features:
    benchmarking_set = benchmarking_set_tokenized.rename_column("label", "labels")
else:
    benchmarking_set = benchmarking_set_tokenized

print("Tokenizing study set for Train/Val...")
study_set_tokenized = study_set_raw.map(tokenize, batched=True, remove_columns=['text'])
if "label" in study_set_tokenized.features and "labels" not in study_set_tokenized.features:
    study_set_for_split = study_set_tokenized.rename_column("label", "labels")
else:
    study_set_for_split = study_set_tokenized

print("Datasets prepared and tokenized.")
print(f"Benchmarking set features: {benchmarking_set.features}")
print(f"Study set for split features: {study_set_for_split.features}")

# --- Split study_set into train and validation ---
study_indices = np.arange(len(study_set_for_split))
# Stratify if labels are available and it's a classification task
stratify_column = study_set_for_split["labels"] if "labels" in study_set_for_split.features else None
train_indices, val_indices = train_test_split(
    study_indices,
    test_size=0.2, # 20% for validation
    random_state=42,
    stratify=stratify_column
)

train_dataset = study_set_for_split.select(train_indices)
eval_dataset = study_set_for_split.select(val_indices)

print(f"Study set split into: {len(train_dataset)} training samples, {len(eval_dataset)} validation samples.")


metric_accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_accuracy.compute(predictions=predictions, references=labels)

# --- PEFT Configurations ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

ia3_config = IA3Config(
    target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
    feedforward_modules=["intermediate.dense", "output.dense"],
    task_type=TaskType.SEQ_CLS
)

model_configurations = {
    "no_adapter": None,
    "lora": lora_config,
    "ia3": ia3_config
}

all_results_data = [] # To store data for Parquet file

# --- Main Loop for Model Configurations ---
for config_name, peft_config_obj in model_configurations.items():
    print(f"\n\n===== Starting Benchmark for Model Configuration: {config_name} =====")

    print(f"Initializing base model for {config_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=raw_dataset["train"].features["label"].num_classes
    )

    if peft_config_obj:
        print(f"Applying PEFT adapter: {config_name}")
        model = get_peft_model(model, peft_config_obj)
        model.print_trainable_parameters()

    model.to(device)
    print(f"Model for {config_name} moved to {device}.")

    training_args = TrainingArguments(
        output_dir=f"results_output/{config_name}", # Removed fold_idx
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        push_to_hub=False,
        label_names=["labels"],
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # Use the split train_dataset
        eval_dataset=eval_dataset,   # Use the split eval_dataset
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print(f"Starting training for {config_name}...")
    trainer.train()
    print(f"Training completed for {config_name}.")

    print(f"Evaluating {config_name} on its validation set...")
    eval_results_val = trainer.evaluate(eval_dataset=eval_dataset)
    print(f"{config_name} validation set results: {eval_results_val}")

    print(f"Evaluating {config_name}'s best model on the benchmarking set...")
    eval_results_benchmark = trainer.evaluate(eval_dataset=benchmarking_set)
    print(f"{config_name} benchmarking set results: {eval_results_benchmark}")

    # Store results
    config_data = {
        "model_config": config_name,
        "val_eval_loss": eval_results_val.get('eval_loss'),
        "val_eval_accuracy": eval_results_val.get('eval_accuracy'),
        "val_eval_runtime": eval_results_val.get('eval_runtime'),
        "val_eval_samples_per_second": eval_results_val.get('eval_samples_per_second'),
        "benchmark_eval_loss": eval_results_benchmark.get('eval_loss'),
        "benchmark_eval_accuracy": eval_results_benchmark.get('eval_accuracy'),
        "benchmark_eval_runtime": eval_results_benchmark.get('eval_runtime'),
        "benchmark_eval_samples_per_second": eval_results_benchmark.get('eval_samples_per_second'),
        "epoch_at_best_val": eval_results_val.get('epoch'),
        # "epoch_at_best_benchmark" might be misleading as benchmark eval uses model saved based on val set
    }
    all_results_data.append(config_data)

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n--- All Benchmarking Finished ---")

# Convert results to DataFrame and save to Parquet
results_df = pd.DataFrame(all_results_data)
parquet_file_path = "benchmark_results_train_test_split.parquet" # Changed filename
results_df.to_parquet(parquet_file_path)
print(f"\nResults saved to {parquet_file_path}")

# --- Optional: Print Summary Statistics from DataFrame ---
if not results_df.empty:
    print("\n--- Results (from DataFrame) ---")
    for index, row in results_df.iterrows():
        print(f"\n--- Configuration: {row['model_config']} ---")
        print(f"  Validation Set: Accuracy = {row['val_eval_accuracy']:.4f}, Loss = {row['val_eval_loss']:.4f}, Runtime = {row['val_eval_runtime']:.2f}s")
        print(f"  Benchmarking Set: Accuracy = {row['benchmark_eval_accuracy']:.4f}, Loss = {row['benchmark_eval_loss']:.4f}, Runtime = {row['benchmark_eval_runtime']:.2f}s")
        print(f"  Epoch for Best Validation Model: {row['epoch_at_best_val']}")
else:
    print("No results collected to summarize or save.")

print("\nScript completed.")