"""Fine-tune Romanian BERT (dumitrescustefan/bert-base-romanian-cased-v1) for spam classification.

Runs locally or on Colab through notebooks/bert_colab.ipynb. The model reads the
cleaned, cased text (placeholders and comma-below diacritics, which the model
card asks for); it is recomputed from the raw email here so that Colab only needs
the split files and not the Stanza preprocessing.
"""

import logging
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.split import load_splits
from src.models.evaluate import evaluate, plot_confusion_matrices
from src.preprocessing.cleaning import clean_text
from src.utils.config import load_config

MODEL_DIR_NAME = "bert-spam-ro"
ID2LABEL = {0: "ham", 1: "spam"}

logger = logging.getLogger(__name__)


def to_dataset(texts: list[str], labels: list[int], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    metrics = evaluate(labels, np.argmax(logits, axis=-1))
    return {key: metrics[key] for key in ("spam_precision", "spam_recall", "spam_f1")}


def train_bert(config: dict) -> pd.DataFrame:
    params = config["bert"]
    paths = config["paths"]
    splits = load_splits(config)
    texts = {name: [clean_text(t, config["preprocessing"]) for t in df["text"]] for name, df in splits.items()}

    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    lengths = [len(ids) for ids in tokenizer(texts["train"])["input_ids"]]
    logger.info("train emails over %d tokens: %d of %d", params["max_length"],
                sum(n > params["max_length"] for n in lengths), len(lengths))

    datasets = {
        name: to_dataset(texts[name], splits[name]["label"].tolist(), tokenizer, params["max_length"])
        for name in splits
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        params["model_name"], num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    )

    checkpoint_dir = paths["models"] / "bert_checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"] * 2,
        weight_decay=params["weight_decay"],
        # In transformers 5 a float below 1 here is the warmup ratio (warmup_ratio was removed).
        warmup_steps=params["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spam_f1",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=config["data"]["random_state"],
        logging_steps=10,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model_dir = paths["models"] / MODEL_DIR_NAME
    trainer.save_model(str(model_dir))
    # Each checkpoint holds the optimizer state too (~1.3 GB); only the best model is kept.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    rows = []
    for split in ("val", "test"):
        predictions = np.argmax(trainer.predict(datasets[split]).predictions, axis=-1)
        rows.append({"model": "bert", "split": split, **evaluate(splits[split]["label"], predictions)})
        if split == "test":
            matrix = confusion_matrix(splits[split]["label"], predictions, labels=[0, 1])
            plot_confusion_matrices({"Romanian BERT": matrix}, paths["figures"] / "bert_confusion_matrix.png")
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval(), device


def predict_spam_proba(texts: list[str], config: dict, batch_size: int = 32) -> np.ndarray:
    """Spam probability for raw email texts, using the fine-tuned model saved by train_bert."""
    tokenizer, model, device = load_model(config["paths"]["models"] / MODEL_DIR_NAME)
    cleaned = [clean_text(t, config["preprocessing"]) for t in texts]
    probabilities = []
    for start in range(0, len(cleaned), batch_size):
        inputs = tokenizer(cleaned[start : start + batch_size], truncation=True, padding=True,
                           max_length=config["bert"]["max_length"], return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        probabilities.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    results = train_bert(config)
    out_path = config["paths"]["reports"] / "bert_results.csv"
    results.round(4).to_csv(out_path, index=False)
    columns = ["model", "split", "spam_precision", "spam_recall", "spam_f1", "false_positives", "false_negatives"]
    print(results[columns].round(3).to_string(index=False))
    logger.info("saved %s", out_path)
