"""Add cleaned, lemmatized and stemmed text columns to the train/val/test splits."""

import logging
from functools import partial

import pandas as pd

from src.data.split import SPLITS
from src.preprocessing.cleaning import clean_text
from src.preprocessing.nlp import cached_apply, lemmatize_texts, stem_texts
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def preprocess_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    params = config["preprocessing"]
    cleaned = [clean_text(text, params) for text in df["text"]]

    # Stanza is the slow step, so its output is cached; the file name carries the
    # options that change the result, so toggling them doesn't reuse stale output.
    cache_name = f"lemma_cache_sw{int(params['remove_stopwords'])}_lc{int(params['lowercase'])}.parquet"
    lemmatize = partial(
        lemmatize_texts,
        remove_stopwords=params["remove_stopwords"],
        lowercase=params["lowercase"],
        use_gpu=params["stanza_use_gpu"],
    )
    lemmas = cached_apply(cleaned, lemmatize, config["paths"]["interim"] / cache_name)
    stems = stem_texts(cleaned, remove_stopwords=params["remove_stopwords"], lowercase=params["lowercase"])

    return df.assign(text_clean=cleaned, text_lemma=lemmas, text_stem=stems)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    processed = config["paths"]["processed"]
    for split in SPLITS:
        path = processed / f"{split}.parquet"
        df = preprocess_frame(pd.read_parquet(path), config)
        df.to_parquet(path, index=False)
        logger.info("%-5s %d emails -> %s", split, len(df), path)
