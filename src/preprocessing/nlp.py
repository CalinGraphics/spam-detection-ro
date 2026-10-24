"""Tokenization, lemmatization (Stanza) and stemming (Snowball) of cleaned Romanian text."""

import hashlib
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.ro.stop_words import STOP_WORDS

from src.preprocessing.cleaning import EMAIL_TOKEN, MONEY_TOKEN, NUMBER_TOKEN, URL_TOKEN, normalize_diacritics

PLACEHOLDERS = {URL_TOKEN, EMAIL_TOKEN, MONEY_TOKEN, NUMBER_TOKEN}

# spaCy's list mixes cedilla and comma-below spellings of the same words.
STOPWORDS = frozenset(normalize_diacritics(word) for word in STOP_WORDS)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_stanza(use_gpu: bool = True) -> spacy.Language:
    import spacy_stanza

    return spacy_stanza.load_pipeline("ro", processors="tokenize,pos,lemma", use_gpu=use_gpu, logging_level="WARN")


@lru_cache(maxsize=1)
def blank_tokenizer() -> spacy.Language:
    # The rule-based Romanian tokenizer handles clitics ("n-am", "într-o") without loading a model.
    return spacy.blank("ro")


def _keep(word: str, remove_stopwords: bool) -> bool:
    if word in PLACEHOLDERS:
        return True
    if not any(ch.isalnum() for ch in word):
        return False
    return not (remove_stopwords and word.lower() in STOPWORDS)


def _finish(words: list[str], lowercase: bool) -> str:
    # Placeholders stay uppercase so they can't collide with a real word like "email".
    return " ".join(w if w in PLACEHOLDERS or not lowercase else w.lower() for w in words)


def token_lemma(token) -> str:
    return token.text if token.text in PLACEHOLDERS else (token.lemma_ or token.text)


def join_lemmas(lemmas: list[str], remove_stopwords: bool = True, lowercase: bool = True) -> str:
    """Turn a token-level lemma sequence into the text the TF-IDF models were trained on."""
    lemmas = [lem.strip("-") for lem in lemmas]
    return _finish([lem for lem in lemmas if _keep(lem, remove_stopwords)], lowercase)


def lemmatize_texts(
    texts: list[str], remove_stopwords: bool = True, lowercase: bool = True, use_gpu: bool = True
) -> list[str]:
    nlp = load_stanza(use_gpu)
    return [
        join_lemmas([token_lemma(tok) for tok in doc], remove_stopwords, lowercase) for doc in nlp.pipe(texts)
    ]


def stem_texts(texts: list[str], remove_stopwords: bool = True, lowercase: bool = True) -> list[str]:
    stemmer = SnowballStemmer("romanian")
    results = []
    for doc in blank_tokenizer().pipe(texts):
        # Clitics come out as separate tokens with a hyphen ("contactați-ne" -> "-ne"),
        # which would otherwise slip past the stopword list.
        words = [tok.text.strip("-") for tok in doc]
        words = [w for w in words if _keep(w, remove_stopwords)]
        results.append(_finish([w if w in PLACEHOLDERS else stemmer.stem(w) for w in words], lowercase))
    return results


def cached_apply(texts: list[str], fn: Callable[[list[str]], list[str]], cache_path: Path) -> list[str]:
    """Run fn only on texts missing from the parquet cache at cache_path.

    The cache is keyed by a hash of the input text, so it must be cleared when
    the function's parameters change.
    """
    cache: dict[str, str] = {}
    if cache_path.exists():
        stored = pd.read_parquet(cache_path)
        cache = dict(zip(stored["key"], stored["value"]))

    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = {k: t for k, t in zip(keys, texts) if k not in cache}
    if missing:
        logger.info("%s: %d cached, %d to process", cache_path.name, len(texts) - len(missing), len(missing))
        cache.update(zip(missing.keys(), fn(list(missing.values()))))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"key": list(cache), "value": list(cache.values())}).to_parquet(cache_path, index=False)
    return [cache[k] for k in keys]
