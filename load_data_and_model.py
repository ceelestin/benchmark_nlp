import evaluate
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
