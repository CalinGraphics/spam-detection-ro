"""LIME explanations for TF-IDF + Logistic Regression and Romanian BERT on a few test emails.

Picks the emails that at least two models get wrong, plus the spam that logistic
regression is most confident about as a reference. Needs reports/test_predictions.csv
from src.models.compare.
"""

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.data.split import load_splits
from src.models.bert import predict_spam_proba
from src.models.evaluate import INK, INK_SECONDARY, style_axes
from src.utils.config import load_config

# Diverging pair from the validated palette: red pushes towards spam, blue towards ham.
SPAM_COLOR, HAM_COLOR = "#e34948", "#2a78d6"
N_FEATURES = 8

logger = logging.getLogger(__name__)


def pick_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    pred_columns = [c for c in predictions.columns if c.endswith("| pred")]
    n_wrong = predictions[pred_columns].ne(predictions["label"], axis=0).sum(axis=1)
    hard = predictions[n_wrong >= 2]
    caught = predictions[(predictions["label"] == 1) & (n_wrong == 0)]
    reference = caught.nlargest(1, "TF-IDF + Logistic Regression | score")
    return pd.concat([hard, reference])


def explain(texts: list[str], predict_fn, num_samples: int, random_state: int) -> list[list[tuple[str, float]]]:
    explainer = LimeTextExplainer(class_names=["ham", "spam"], random_state=random_state)
    return [
        explainer.explain_instance(text, predict_fn, num_features=N_FEATURES, num_samples=num_samples).as_list()
        for text in texts
    ]


def plot_explanation(panels: dict[str, list[tuple[str, float]]], title: str, path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6))
    for ax, (name, weights) in zip(axes, panels.items()):
        weights = sorted(weights, key=lambda item: abs(item[1]))
        words = [word for word, _ in weights]
        values = [value for _, value in weights]
        ax.barh(words, values, color=[SPAM_COLOR if v > 0 else HAM_COLOR for v in values], height=0.6)
        ax.axvline(0, color="#c3c2b7", linewidth=1)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("weight (> 0 pushes towards spam)", color=INK_SECONDARY, fontsize=9)
        style_axes(ax)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def run_explanations(config: dict) -> pd.DataFrame:
    paths = config["paths"]
    random_state = config["data"]["random_state"]
    predictions = pd.read_csv(paths["reports"] / "test_predictions.csv")
    examples = pick_examples(predictions)
    # The lemmatized text the TF-IDF model reads is only in the split file.
    test = load_splits(config)["test"].set_index("text")
    examples = examples.assign(text_lemma=test.loc[examples["text"], "text_lemma"].to_numpy())

    logreg = joblib.load(paths["models"] / "baseline_logistic_regression.joblib")

    def bert_proba(texts: list[str]) -> np.ndarray:
        spam = predict_spam_proba(texts, config)
        return np.column_stack([1 - spam, spam])

    lr_weights = explain(examples["text_lemma"].tolist(), logreg.predict_proba, 2000, random_state)
    bert_weights = explain(examples["text"].tolist(), bert_proba, 1000, random_state)

    rows = []
    records = examples.to_dict("records")
    for i, (row, lr, bert) in enumerate(zip(records, lr_weights, bert_weights), start=1):
        subject = row["text"].split("\n", 1)[0][:60]
        title = (
            f"{subject}  (true: {'spam' if row['label'] else 'ham'}; P(spam) "
            f"LR {row['TF-IDF + Logistic Regression | score']:.2f}, BERT {row['Romanian BERT | score']:.2f})"
        )
        plot_explanation({"TF-IDF + Logistic Regression": lr, "Romanian BERT": bert}, title,
                         paths["figures"] / f"lime_example_{i}.png")
        for model, weights in (("logistic_regression", lr), ("bert", bert)):
            rows += [{"example": i, "file": row["file"], "label": row["label"], "model": model, "word": w, "weight": v}
                     for w, v in weights]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    weights = run_explanations(config)
    weights.round(4).to_csv(config["paths"]["reports"] / "lime_weights.csv", index=False)
    print(weights.groupby(["example", "model"]).head(3).to_string(index=False))
