"""Ablation: TF-IDF alone, then TF-IDF plus each hand-crafted feature group.

Differences between these models on the 147-email test set come down to one or
two emails, so the comparison uses 5-fold cross-validation on train+val (mean
and std over folds); the test score of a model refit on train+val is reported
alongside. The regularization strength is picked once, on the TF-IDF-only model,
and kept fixed so that only the features change between rows.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.features.handcrafted import FEATURE_COLUMNS, FEATURE_GROUPS, extract_features
from src.features.tfidf import build_tfidf
from src.models.baselines import load_splits
from src.models.evaluate import evaluate, plot_dot_comparison
from src.utils.config import load_config

TEXT_COLUMN = "text_lemma"
SCORING = {"spam_precision": "precision", "spam_recall": "recall", "spam_f1": "f1"}

logger = logging.getLogger(__name__)


def feature_sets() -> dict[str, tuple[bool, list[str]]]:
    """Name -> (use TF-IDF, hand-crafted columns)."""
    sets = {"tfidf": (True, [])}
    sets.update({f"tfidf + {group}": (True, columns) for group, columns in FEATURE_GROUPS.items()})
    sets["tfidf + all"] = (True, FEATURE_COLUMNS)
    sets["hand-crafted only"] = (False, FEATURE_COLUMNS)
    return sets


def make_model(use_tfidf: bool, columns: list[str], c: float, config: dict) -> Pipeline:
    transformers = []
    if use_tfidf:
        transformers.append(("tfidf", build_tfidf(config["tfidf"]), TEXT_COLUMN))
    if columns:
        # Min-max keeps the extra features in [0, 1] like the TF-IDF weights. With
        # standardization they reached values of 5-10, the regularized model leaned
        # on them instead of the text, and every group looked harmful.
        transformers.append(("handcrafted", MinMaxScaler(), columns))
    classifier = LogisticRegression(C=c, max_iter=2000, random_state=config["data"]["random_state"])
    return Pipeline([("features", ColumnTransformer(transformers)), ("clf", classifier)])


def pick_c(train_val: pd.DataFrame, cv: list, config: dict) -> float:
    scores = {
        c: cross_val_score(make_model(True, [], c, config), train_val, train_val["label"], cv=cv, scoring="f1").mean()
        for c in config["ablation"]["logreg_c_grid"]
    }
    logger.info("C grid on tfidf only: %s", {c: round(s, 4) for c, s in scores.items()})
    return max(scores, key=scores.get)


def run_ablation(config: dict) -> pd.DataFrame:
    splits = load_splits(config)
    for name, df in splits.items():
        logger.info("extracting hand-crafted features for %s", name)
        splits[name] = pd.concat([df, extract_features(df["text"].tolist())], axis=1)

    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test = splits["test"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["data"]["random_state"])
    cv = list(folds.split(train_val, train_val["category"]))
    c = pick_c(train_val, cv, config)

    rows = []
    for name, (use_tfidf, columns) in feature_sets().items():
        model = make_model(use_tfidf, columns, c, config)
        scores = cross_validate(model, train_val, train_val["label"], cv=cv, scoring=SCORING)
        row = {"features": name, "C": c}
        for metric in SCORING:
            row[f"cv_{metric}"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()

        model.fit(train_val, train_val["label"])
        test_metrics = evaluate(test["label"], model.predict(test))
        row.update({f"test_{metric}": test_metrics[metric] for metric in SCORING})
        row["test_false_positives"] = test_metrics["false_positives"]
        rows.append(row)
        logger.info("%-20s cv F1 %.3f", name, row["cv_spam_f1"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    results = run_ablation(config)
    paths = config["paths"]
    results.round(4).to_csv(paths["reports"] / "ablation_results.csv", index=False)

    for metric in ("spam_f1", "spam_precision"):
        plot_dot_comparison(
            results, row="features", value=f"cv_{metric}", error=f"cv_{metric}_std",
            path=paths["figures"] / f"ablation_{metric}.png",
            title=f"{metric.replace('_', ' ')}, 5-fold CV on train+val (mean ± std)",
        )
    columns = ["features", "cv_spam_precision", "cv_spam_recall", "cv_spam_f1", "cv_spam_f1_std",
               "test_spam_precision", "test_spam_f1", "test_false_positives"]
    print(results[columns].round(3).to_string(index=False))
