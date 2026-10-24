"""Score every saved model on the test split: one comparison table plus per-email predictions."""

import logging

import joblib
import numpy as np
import pandas as pd

from src.data.split import load_splits
from src.features.embeddings import document_vectors, load_table
from src.models.bert import predict_spam_proba
from src.models.evaluate import evaluate, plot_dot_comparison
from src.utils.config import load_config

TFIDF_MODELS = {
    "TF-IDF + Naive Bayes": "baseline_naive_bayes",
    "TF-IDF + Logistic Regression": "baseline_logistic_regression",
    "TF-IDF + linear SVM": "baseline_linear_svm",
}

logger = logging.getLogger(__name__)


def spam_scores(config: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Model name -> spam score per email (probability, or the SVM's decision value)."""
    models_dir = config["paths"]["models"]
    scores = {}
    for name, file in TFIDF_MODELS.items():
        pipeline = joblib.load(models_dir / f"{file}.joblib")
        if hasattr(pipeline, "predict_proba"):
            scores[name] = pipeline.predict_proba(df["text_lemma"])[:, 1]
        else:
            scores[name] = pipeline.decision_function(df["text_lemma"])

    table = load_table(config["paths"]["interim"] / "fasttext_ro_table.npz")
    vectors, _ = document_vectors(df["text_clean"].tolist(), table)
    fasttext = joblib.load(models_dir / "fasttext_logistic_regression.joblib")
    scores["fastText + Logistic Regression"] = fasttext.predict_proba(vectors)[:, 1]

    scores["Romanian BERT"] = predict_spam_proba(df["text"].tolist(), config)
    return scores


def decision(name: str, score: np.ndarray) -> np.ndarray:
    # The SVM score is a signed distance, the others are probabilities.
    threshold = 0.0 if "SVM" in name else 0.5
    return (score > threshold).astype(int)


def run_comparison(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = load_splits(config)["test"]
    scores = spam_scores(config, test)

    rows = []
    predictions = test[["file", "category", "label", "text"]].copy()
    for name, score in scores.items():
        predicted = decision(name, score)
        rows.append({"model": name, **evaluate(test["label"], predicted)})
        predictions[f"{name} | score"] = score
        predictions[f"{name} | pred"] = predicted
    return pd.DataFrame(rows), predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    paths = config["paths"]
    table, predictions = run_comparison(config)

    table.round(4).to_csv(paths["reports"] / "comparison.csv", index=False)
    predictions.round(4).to_csv(paths["reports"] / "test_predictions.csv", index=False)

    long = table.melt(id_vars="model", value_vars=["spam_precision", "spam_recall", "spam_f1"],
                      var_name="metric", value_name="score")
    long["metric"] = long["metric"].str.replace("spam_", "")
    plot_dot_comparison(long, row="model", value="score", hue="metric", path=paths["figures"] / "comparison.png",
                        title="Spam precision, recall and F1 on the test set (147 emails)")

    columns = ["model", "spam_precision", "spam_recall", "spam_f1", "accuracy", "false_positives", "false_negatives"]
    print(table[columns].round(3).to_string(index=False))

    pred_columns = [c for c in predictions.columns if c.endswith("| pred")]
    wrong = predictions[(predictions[pred_columns].ne(predictions["label"], axis=0)).any(axis=1)]
    print(f"\n{len(wrong)} test emails misclassified by at least one model")
