"""Load the LLM-generated Romanian emails used only as an extra test set.

Unlike the main corpus these are written directly in Romanian, not machine-translated.
Expected input: one or more CSV files in data/raw/synthetic_ro/ with columns
`text` and `label` (1 = spam, 0 = ham), UTF-8 encoded.
"""

from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def load_synthetic_ro(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {folder}")

    frames = []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8")
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frames.append(df[["text", "label"]].assign(file=path.name))

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    bad_labels = ~df["label"].isin([0, 1])
    if bad_labels.any():
        raise ValueError(f"labels must be 0 or 1, found: {sorted(df.loc[bad_labels, 'label'].unique())}")

    df["label"] = df["label"].astype(int)
    return df.drop_duplicates(subset="text").reset_index(drop=True)


if __name__ == "__main__":
    config = load_config()
    df = load_synthetic_ro(config["paths"]["raw"] / "synthetic_ro")
    print(f"emails: {len(df)}  spam: {df['label'].sum()}  ham: {(df['label'] == 0).sum()}")
