"""Fill the Hugging Face cache with Yelp Review Full and the checkpoints, so that offline compute nodes find them.

Run once on a node with internet access, with the HF_HOME of the launchers (for example
HF_HOME=$WORK/hf_cache python load_data_and_model.py). The paper uses bert, roberta and qwen3-0.6b; the other
checkpoints of the list (t5, modernbert, qwen3-1.7b) served only in earlier runs.
"""
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import logging as hf_logging


MODEL_HUGGINGFACE_IDENTIFIERS = {
    "t5":         "google-t5/t5-base",
    "bert":       "google-bert/bert-base-cased",
    "roberta":    "FacebookAI/xlm-roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

DECODER_MODELS = {"qwen3-0.6b", "qwen3-1.7b"}

raw_dataset_full = load_dataset(
    "yelp_review_full", "yelp_review_full",
    download_mode="reuse_cache_if_exists"
)
num_labels_for_classification = (
    raw_dataset_full["train"].features["label"].num_classes
)

for model_key, model_name in MODEL_HUGGINGFACE_IDENTIFIERS.items():
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_key in DECODER_MODELS:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        hf_logging.set_verbosity_error()
    model_instance = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels_for_classification
    )
    if model_key in DECODER_MODELS:
        hf_logging.set_verbosity_warning()
