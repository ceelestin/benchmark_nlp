import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


MODEL_HUGGINGFACE_IDENTIFIERS = {
    "t5": "google-t5/t5-base",
    "bert": "google-bert/bert-base-cased",
    "roberta": "FacebookAI/xlm-roberta-base",
    "modernbert": "answerdotai/ModernBERT-base"
}

raw_dataset_full = load_dataset(
    "yelp_review_full", download_mode="reuse_cache_if_exists"
)
num_labels_for_classification = (
    raw_dataset_full["train"].features["label"].num_classes
)

for model_name in MODEL_HUGGINGFACE_IDENTIFIERS.values():
    print(f"Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_instance = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels_for_classification
    )
