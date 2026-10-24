"""Logistic Regression on averaged fastText cc.ro.300 document vectors."""

import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.embeddings import build_table, document_vectors, download_vectors, load_table, tokenize
from src.models.baselines import load_splits
from src.models.evaluate import category_folds, evaluate, plot_confusion_matrices, select_param
from src.utils.config import load_config

TEXT_COLUMN = "text_clean"

logger = logging.getLogger(__name__)


def make_model(c: float, random_state: int) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=2000, random_state=random_state))]
    )


def run_fasttext(config: dict) -> pd.DataFrame:
    params = config["fasttext"]
    paths = config["paths"]
    splits = load_splits(config)
    random_state = config["data"]["random_state"]

    table_path = paths["interim"] / "fasttext_ro_table.npz"
    if not table_path.exists():
        vec_path = paths["raw"] / "fasttext" / "cc.ro.300.vec.gz"
        download_vectors(params["vectors_url"], vec_path)
        # Test words are included on purpose: this only decides which pretrained vectors
        # to keep (no labels involved), the same as using the full 2M-word table.
        vocabulary = {w for df in splits.values() for text in df[TEXT_COLUMN] for w in tokenize(text)}
        build_table(vec_path, table_path, vocabulary, params["top_n_words"])
    table = load_table(table_path)

    features = {}
    for name, df in splits.items():
        features[name], coverage = document_vectors(df[TEXT_COLUMN].tolist(), table)
        logger.info("%s: %.1f%% of words have a vector", name, 100 * coverage)

    train = splits["train"]
    c = select_param(
        lambda value: make_model(value, random_state), params["logreg_c"], features["train"], train["label"],
        category_folds(train, random_state), name="fasttext C",
    )
    model = make_model(c, random_state).fit(features["train"], train["label"])
    joblib.dump(model, paths["models"] / "fasttext_logistic_regression.joblib")

    rows = []
    for split in ("val", "test"):
        predictions = model.predict(features[split])
        rows.append({"model": "fasttext_logreg", "param": c, "split": split,
                     **evaluate(splits[split]["label"], predictions)})
        if split == "test":
            matrix = confusion_matrix(splits[split]["label"], predictions, labels=[0, 1])
            plot_confusion_matrices({"fastText + logistic regression": matrix},
                                    paths["figures"] / "fasttext_confusion_matrix.png")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    results = run_fasttext(config)
    out_path = config["paths"]["reports"] / "fasttext_results.csv"
    results.round(4).to_csv(out_path, index=False)
    columns = ["model", "param", "split", "spam_precision", "spam_recall", "spam_f1", "false_positives",
               "false_negatives"]
    print(results[columns].round(3).to_string(index=False))
    logger.info("saved %s", out_path)
