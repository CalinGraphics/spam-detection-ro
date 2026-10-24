"""Stratified train/val/test split of the email dataset."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_emails
from src.utils.config import load_config

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def load_splits(config: dict) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(config["paths"]["processed"] / f"{name}.parquet") for name in SPLITS}


def split_dataset(
    df: pd.DataFrame, test_size: float, val_size: float, random_state: int
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test; val_size and test_size are fractions of the whole dataset."""
    # Stratifying on category (each one belongs to a single label) keeps the
    # spam/ham balance and makes sure small categories appear in every split.
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["category"], random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["category"],
        random_state=random_state,
    )
    return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    params = config["data"]
    df = load_emails(config["paths"]["raw"] / "generated")

    splits = split_dataset(df, params["test_size"], params["val_size"], params["random_state"])
    for name, part in splits.items():
        path = config["paths"]["processed"] / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(path, index=False)
        logger.info("%-5s %4d emails, spam ratio %.3f -> %s", name, len(part), part["label"].mean(), path)
