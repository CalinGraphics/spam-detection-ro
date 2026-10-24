"""Score and explain a single email with any of the saved models (used by the demo app).

Explanations come from LIME for every model, at token level. For the TF-IDF
models the email is lemmatized once and each perturbed sample reuses those
lemmas, since running Stanza on a thousand samples would take minutes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

from src.features.embeddings import document_vectors, load_table
from src.models.bert import MODEL_DIR_NAME, predict_spam_proba
from src.preprocessing.cleaning import clean_text
from src.preprocessing.nlp import blank_tokenizer, join_lemmas, load_stanza, token_lemma

# Linear SVM is left out on purpose: it outputs a margin, not a probability.
MODELS = {
    "Logistic Regression": "baseline_logistic_regression.joblib",
    "Naive Bayes": "baseline_naive_bayes.joblib",
    "fastText": "fasttext_logistic_regression.joblib",
    "BERT": MODEL_DIR_NAME,
}
TFIDF_MODELS = {"Logistic Regression", "Naive Bayes"}


@dataclass
class Token:
    text: str
    lemma: str
    start: int
    end: int


@dataclass
class Explanation:
    text: str  # cleaned text the tokens point into
    spam_probability: float
    tokens: list[Token]
    weights: list[float]  # LIME weight per token, > 0 pushes towards spam


def available_models(config: dict) -> list[str]:
    models_dir: Path = config["paths"]["models"]
    available = [name for name, file in MODELS.items() if (models_dir / file).exists()]
    if "fastText" in available and not (config["paths"]["interim"] / "fasttext_ro_table.npz").exists():
        available.remove("fastText")
    return available


@lru_cache(maxsize=4)
def _load_joblib(path: Path):
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_fasttext_table(path: Path):
    return load_table(path)


def _tokenize(cleaned: str, with_lemmas: bool, use_gpu: bool) -> tuple[str, list[Token]]:
    doc = load_stanza(use_gpu)(cleaned) if with_lemmas else blank_tokenizer()(cleaned)
    tokens = [
        # LIME splits on single spaces, so a token must never contain one.
        Token(tok.text.replace(" ", ""), token_lemma(tok) if with_lemmas else tok.text, tok.idx, tok.idx + len(tok.text))
        for tok in doc
        if tok.text.strip()
    ]
    return doc.text, tokens


def _scorer(model: str, tokens: list[Token], config: dict) -> Callable[[list[list[int]]], np.ndarray]:
    """Return a function mapping lists of kept token indices to spam probabilities."""
    models_dir = config["paths"]["models"]
    params = config["preprocessing"]

    if model in TFIDF_MODELS:
        pipeline = _load_joblib(models_dir / MODELS[model])

        def score(kept: list[list[int]]) -> np.ndarray:
            texts = [join_lemmas([tokens[i].lemma for i in idx], params["remove_stopwords"], params["lowercase"])
                     for idx in kept]
            return pipeline.predict_proba(texts)[:, 1]

    elif model == "fastText":
        classifier = _load_joblib(models_dir / MODELS[model])
        table = _load_fasttext_table(config["paths"]["interim"] / "fasttext_ro_table.npz")

        def score(kept: list[list[int]]) -> np.ndarray:
            vectors, _ = document_vectors([" ".join(tokens[i].text for i in idx) for idx in kept], table)
            return classifier.predict_proba(vectors)[:, 1]

    else:

        def score(kept: list[list[int]]) -> np.ndarray:
            return predict_spam_proba([" ".join(tokens[i].text for i in idx) for idx in kept], config)

    return score


def explain_email(text: str, model: str, config: dict, num_samples: int = 1000) -> Explanation:
    cleaned = clean_text(text, config["preprocessing"])
    display_text, tokens = _tokenize(cleaned, model in TFIDF_MODELS, config["preprocessing"]["stanza_use_gpu"])
    if not tokens:
        raise ValueError("the email has no words left after cleaning")

    score = _scorer(model, tokens, config)
    spam_probability = float(score([list(range(len(tokens)))])[0])

    def lime_predict(samples: list[str]) -> np.ndarray:
        # Removed tokens come back as empty strings, so positions still line up with `tokens`.
        kept = [[i for i, part in enumerate(sample.split(" ")) if part] for sample in samples]
        spam = score(kept)
        return np.column_stack([1 - spam, spam])

    explainer = LimeTextExplainer(
        class_names=["ham", "spam"], split_expression=" ", bow=False, mask_string="",
        random_state=config["data"]["random_state"],
    )
    result = explainer.explain_instance(
        " ".join(t.text for t in tokens), lime_predict, num_features=min(len(tokens), 15), num_samples=num_samples
    )
    weights = [0.0] * len(tokens)
    for index, weight in result.as_map()[1]:
        weights[index] = weight
    return Explanation(display_text, spam_probability, tokens, weights)
