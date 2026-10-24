"""Metrics, hyperparameter selection and plots shared by all models."""

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

# Validated light-mode palette (blue / orange categorical slots, blue sequential ramp).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f9f9f7", "#9ec5f4", "#2a78d6", "#104281"])
INK, INK_SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LABELS = ["ham", "spam"]


def evaluate(y_true, y_pred) -> dict:
    """Per-class precision/recall/F1 plus the confusion counts (spam is the positive class)."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "spam_precision": precision[1],
        "spam_recall": recall[1],
        "spam_f1": f1[1],
        "ham_precision": precision[0],
        "ham_recall": recall[0],
        "ham_f1": f1[0],
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def category_folds(df: pd.DataFrame, random_state: int, n_splits: int = 5) -> list:
    """Stratified folds on the email category, which also balances spam/ham."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(folds.split(df, df["category"]))


def select_param(build: Callable[[float], BaseEstimator], values: list[float], X, y, cv: list, name: str = "") -> float:
    """Return the value whose model has the best mean spam F1 over the given folds."""
    scores = {value: cross_val_score(build(value), X, y, cv=cv, scoring="f1").mean() for value in values}
    logger.info("%s grid: %s", name, {value: round(score, 4) for value, score in scores.items()})
    return max(scores, key=scores.get)


def _style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)


def plot_confusion_matrices(matrices: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.4 * len(matrices), 3.4), squeeze=False)
    for ax, (name, cm) in zip(axes[0], matrices.items()):
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=cm.sum() / 2)
        for (i, j), count in np.ndenumerate(cm):
            dark_cell = count > cm.sum() / 4
            ax.text(j, i, str(count), ha="center", va="center", fontsize=13, color="#fcfcfb" if dark_cell else INK)
        ax.set_xticks([0, 1], LABELS)
        ax.set_yticks([0, 1], LABELS)
        ax.set_xlabel("predicted", color=INK_SECONDARY)
        ax.set_ylabel("true", color=INK_SECONDARY)
        ax.set_title(name, color=INK, fontsize=11)
        ax.tick_params(length=0, labelcolor=INK_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def plot_dot_comparison(
    df: pd.DataFrame, row: str, value: str, path: Path, hue: str | None = None, error: str | None = None, title: str = ""
) -> None:
    """Horizontal dot plot of `value` per `row`, optionally one series per `hue`.

    Dots instead of bars because the scores sit close to 1 and a bar would need
    a truncated axis to show any difference.
    """
    rows = list(dict.fromkeys(df[row]))
    hues = list(dict.fromkeys(df[hue])) if hue else [None]
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.2))
    offsets = np.linspace(-0.15, 0.15, len(hues)) if len(hues) > 1 else [0.0]

    for color, offset, h in zip(SERIES_COLORS, offsets, hues):
        part = df if h is None else df[df[hue] == h]
        y = [rows.index(r) + offset for r in part[row]]
        xerr = part[error] if error else None
        ax.errorbar(part[value], y, xerr=xerr, fmt="o", ms=8, color=color, ecolor=color, elinewidth=2,
                    capsize=0, label=h, markeredgecolor="#fcfcfb", markeredgewidth=2)

    ax.set_yticks(range(len(rows)), rows)
    ax.invert_yaxis()
    ax.set_xlabel(value.replace("_", " "), color=INK_SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if hue:
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
