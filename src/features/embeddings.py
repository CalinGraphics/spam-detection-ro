"""Averaged fastText (cc.ro.300) document vectors.

The full .vec file has 2M words and takes ~2.5 GB in memory, so it is streamed
once and reduced to the corpus vocabulary plus the most frequent words, saved
as a small .npz table.
"""

import gzip
import logging
import urllib.request
from pathlib import Path

import numpy as np

from src.preprocessing.cleaning import EMAIL_TOKEN, MONEY_TOKEN, NUMBER_TOKEN, URL_TOKEN
from src.preprocessing.nlp import STOPWORDS, blank_tokenizer

# Placeholders have no vector; a word with a close meaning keeps their signal in the average.
PLACEHOLDER_WORDS = {URL_TOKEN: "link", EMAIL_TOKEN: "email", MONEY_TOKEN: "bani", NUMBER_TOKEN: "număr"}

# Common Crawl text mixes both spellings, so a comma-below word missing from the
# table is looked up again in its cedilla form.
_COMMA_TO_CEDILLA = str.maketrans({"ș": "ş", "ț": "ţ"})

logger = logging.getLogger(__name__)

EmbeddingTable = tuple[dict[str, int], np.ndarray]


def download_vectors(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, path)


def tokenize(text: str) -> list[str]:
    words = []
    for token in blank_tokenizer()(text):
        word = token.text.strip("-")
        if word in PLACEHOLDER_WORDS:
            words.append(PLACEHOLDER_WORDS[word])
        elif any(ch.isalpha() for ch in word) and word.lower() not in STOPWORDS:
            words.append(word.lower())
    return words


def build_table(vec_path: Path, out_path: Path, needed: set[str], top_n: int) -> None:
    """Keep the vectors of `needed` words (lowercased) plus the top_n most frequent words."""
    lookup = needed | {w.translate(_COMMA_TO_CEDILLA) for w in needed}
    vectors: dict[str, np.ndarray] = {}
    with gzip.open(vec_path, "rt", encoding="utf-8", errors="replace") as f:
        next(f)  # header: "<count> <dim>"
        for rank, line in enumerate(f):
            word, values = line.rstrip().split(" ", 1)
            key = word.lower()
            if rank >= top_n and key not in lookup:
                continue
            # The file is sorted by frequency, so the first spelling seen wins,
            # except that an exact lowercase entry replaces a capitalized one.
            if key not in vectors or word == key:
                vectors[key] = np.array(values.split(" "), dtype=np.float32)
    words = list(vectors)
    # A numpy string array pads every word to the longest one (hundreds of MB here),
    # so the words are stored as a single newline-joined string.
    np.savez_compressed(
        out_path, words="\n".join(words), vectors=np.stack([vectors[w] for w in words]).astype(np.float16)
    )
    covered = sum(1 for w in needed if w in vectors or w.translate(_COMMA_TO_CEDILLA) in vectors)
    logger.info("%d of %d corpus words have a vector; table has %d words", covered, len(needed), len(words))


def load_table(path: Path) -> EmbeddingTable:
    data = np.load(path)
    words = str(data["words"]).split("\n")
    return {word: i for i, word in enumerate(words)}, data["vectors"].astype(np.float32)


def document_vectors(texts: list[str], table: EmbeddingTable) -> tuple[np.ndarray, float]:
    """Mean vector of each text's known words (zeros if none) and the share of words found."""
    index, matrix = table
    out = np.zeros((len(texts), matrix.shape[1]), dtype=np.float32)
    found = total = 0
    for i, text in enumerate(texts):
        words = tokenize(text)
        rows = []
        for word in words:
            row = index.get(word, index.get(word.translate(_COMMA_TO_CEDILLA)))
            if row is not None:
                rows.append(row)
        total += len(words)
        found += len(rows)
        if rows:
            out[i] = matrix[rows].mean(axis=0)
    return out, found / total if total else 0.0
