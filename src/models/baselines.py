"""TF-IDF baselines: Multinomial Naive Bayes, Logistic Regression and a linear SVM.

Every model is trained on the train split and scored on val and test, once on
lemmatized and once on stemmed text, so the two normalizations can be compared.
"""

import logging

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.features.tfidf import build_tfidf
from src.models.evaluate import evaluate, plot_confusion_matrices, plot_dot_comparison
from src.utils.config import load_config

TEXT_COLUMNS = {"lemma": "text_lemma", "stem": "text_stem"}

logger = logging.getLogger(__name__)


def make_classifiers(random_state: int) -> dict[str, ClassifierMixin]:
    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "linear_svm": LinearSVC(random_state=random_state),
    }


def make_pipeline(classifier: ClassifierMixin, tfidf_params: dict) -> Pipeline:
    return Pipeline([("tfidf", build_tfidf(tfidf_params)), ("clf", classifier)])


def load_splits(config: dict) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(config["paths"]["processed"] / f"{name}.parquet") for name in ("train", "val", "test")}


def run_baselines(config: dict) -> pd.DataFrame:
    splits = load_splits(config)
    paths = config["paths"]
    rows, test_matrices = [], {}

    for normalization, column in TEXT_COLUMNS.items():
        for name, classifier in make_classifiers(config["data"]["random_state"]).items():
            pipeline = make_pipeline(classifier, config["tfidf"])
            pipeline.fit(splits["train"][column], splits["train"]["label"])

            for split in ("val", "test"):
                df = splits[split]
                predictions = pipeline.predict(df[column])
                rows.append({"model": name, "text": normalization, "split": split, **evaluate(df["label"], predictions)})
                if split == "test" and normalization == "lemma":
                    test_matrices[name] = confusion_matrix(df["label"], predictions, labels=[0, 1])

            if normalization == "lemma":
                joblib.dump(pipeline, paths["models"] / f"baseline_{name}.joblib")

    results = pd.DataFrame(rows)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(test_matrices, paths["figures"] / "baseline_confusion_matrices.png")
    test = results[results["split"] == "test"]
    for metric in ("spam_precision", "spam_f1"):
        plot_dot_comparison(
            test, row="model", value=metric, hue="text", path=paths["figures"] / f"baseline_{metric}.png",
            title=f"{metric.replace('_', ' ')} on the test set, lemmas vs stems",
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    results = run_baselines(config)
    out_path = config["paths"]["reports"] / "baseline_results.csv"
    results.round(4).to_csv(out_path, index=False)
    columns = ["model", "text", "split", "spam_precision", "spam_recall", "spam_f1", "false_positives", "false_negatives"]
    print(results[columns].round(3).to_string(index=False))
    logger.info("saved %s", out_path)
