"""Load the Romanian email dataset from the CSV batches in data/raw/generated/.

Each CSV has the columns `text`, `label` (1 = spam, 0 = ham) and `category`
(the email type requested at generation time, kept for error analysis).
"""

from pathlib import Path

import pandas as pd

from src.utils.config import load_config

REQUIRED_COLUMNS = {"text", "label", "category"}


def load_emails(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {folder}")

    frames = []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8")
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frames.append(df[["text", "label", "category"]].assign(file=path.name))

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    bad_labels = ~df["label"].isin([0, 1])
    if bad_labels.any():
        raise ValueError(f"labels must be 0 or 1, found: {sorted(df.loc[bad_labels, 'label'].unique())}")

    df["label"] = df["label"].astype(int)
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def describe(df: pd.DataFrame) -> None:
    words = df["text"].str.split().str.len()
    print(f"emails: {len(df)}  spam ratio: {df['label'].mean():.3f}")
    print(df.groupby(["label", "category"]).size().to_string())
    print("\nwords per email by label:")
    print(words.groupby(df["label"]).describe().round(0).to_string())


if __name__ == "__main__":
    config = load_config()
    describe(load_emails(config["paths"]["raw"] / "generated"))
