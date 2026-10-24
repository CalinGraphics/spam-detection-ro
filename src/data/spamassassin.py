"""Load the extracted SpamAssassin corpus into a deduplicated DataFrame."""

import logging
from pathlib import Path

import pandas as pd

from src.preprocessing.email_parsing import parse_email
from src.utils.config import load_config

FOLDER_LABELS = {
    "easy_ham": 0,
    "easy_ham_2": 0,
    "hard_ham": 0,
    "spam": 1,
    "spam_2": 1,
}

logger = logging.getLogger(__name__)


def load_spamassassin(corpus_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with columns text, label, source, file (one row per email)."""
    rows = []
    for folder, label in FOLDER_LABELS.items():
        for path in sorted((corpus_dir / folder).iterdir()):
            # "cmds" is a leftover shell script shipped with each folder, not an email.
            if path.name == "cmds":
                continue
            rows.append(
                {
                    "text": parse_email(path.read_bytes()),
                    "label": label,
                    "source": folder,
                    "file": path.name,
                }
            )
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # The corpus contains resent copies of the same message under different file
    # names, so duplicates are detected on the parsed text, case-insensitively.
    key = df["text"].str.lower().str.split().str.join(" ")
    df = df[key.str.len() > 0]
    key = key[df.index]

    conflicting = df.groupby(key)["label"].nunique() > 1
    if conflicting.any():
        logger.warning("dropping %d texts labelled both spam and ham", conflicting.sum())
        df = df[~key.isin(conflicting[conflicting].index)]
        key = key[df.index]

    return df[~key.duplicated()].reset_index(drop=True)


def describe(df: pd.DataFrame) -> None:
    words = df["text"].str.split().str.len()
    print(f"emails: {len(df)}  spam ratio: {df['label'].mean():.3f}")
    print(df.groupby("source")["label"].agg(count="size", label="first").to_string())
    print("\nwords per email by label:")
    print(words.groupby(df["label"]).describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    raw = load_spamassassin(config["paths"]["raw"] / "spamassassin")
    df = deduplicate(raw)
    logger.info("loaded %d emails, %d left after deduplication", len(raw), len(df))
    describe(df)

    out_path = config["paths"]["interim"] / "spamassassin_en.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("saved %s", out_path)
