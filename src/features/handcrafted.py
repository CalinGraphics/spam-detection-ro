"""Hand-crafted features: NER counts, phishing phrases, sentiment, surface statistics, out-of-vocabulary ratio.

They are computed on the original text (HTML and headers stripped, diacritics
normalized) rather than on the placeholder version, since entity types, links
and capital letters are exactly what the placeholders and lowercasing remove.
"""

import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher

from src.preprocessing.cleaning import URL_PATTERN, normalize_diacritics, strip_headers, strip_html
from src.preprocessing.nlp import STOPWORDS

FEATURE_GROUPS = {
    "ner": ["ner_money", "ner_org"],
    "phrases": ["phrase_account", "phrase_prize", "phrase_urgency", "phrase_action"],
    "sentiment": ["sentiment_pos", "sentiment_neg"],
    "surface": ["uppercase_ratio", "n_links", "n_exclamations"],
    "oov": ["oov_ratio"],
}
FEATURE_COLUMNS = [column for group in FEATURE_GROUPS.values() for column in group]


def _fold(word: str) -> str:
    # About 5% of the emails are written without diacritics, so each pattern word also matches its folded form.
    return "".join(ch for ch in unicodedata.normalize("NFD", word) if unicodedata.category(ch) != "Mn")


def _tok(*words: str, op: str | None = None) -> dict:
    variants = sorted({w for word in words for w in (word, _fold(word))})
    token = {"LOWER": {"IN": variants}}
    if op:
        token["OP"] = op
    return token


_NUMBER = {"LIKE_NUM": True}

PHRASE_PATTERNS = {
    "phrase_account": [
        # "contul dvs. a fost blocat", "contul tău a fost suspendat temporar"
        [_tok("contul", "cont", "contului"), _tok("dvs.", "dvs", "dumneavoastră", "tău", "tau", op="?"),
         _tok("a", "va"), _tok("fost", "fi"), _tok("blocat", "suspendat", "restricționat", "limitat",
                                                 "dezactivat", "închis", "șters")],
        # "verificați / confirmați / actualizați datele / contul / identitatea"
        [_tok("verificați", "verifică", "confirmați", "confirmă", "actualizați", "actualizează", "reactivați",
              "reactivează", "validați", "validează"),
         _tok("datele", "contul", "identitatea", "cardul", "adresa", "metoda")],
        [_tok("datele"), _tok("cardului", "bancare", "personale")],
        [_tok("datele"), _tok("de"), _tok("autentificare", "card", "plată")],
    ],
    "phrase_prize": [
        [_tok("ați", "ai"), _tok("câștigat")],
        [_tok("ați", "ai", "ești"), _tok("fost"), _tok("selectat", "selectată", "ales", "aleasă", "extras", "desemnat")],
        [_tok("revendică", "revendicați", "revendica"), _tok("premiul", "premiu", "acum")],
        [_tok("felicitări")],
    ],
    "phrase_urgency": [
        [_tok("în", "in"), _NUMBER, _tok("de", op="?"), _tok("ore", "ore.", "zile")],
        [_tok("doar", "numai"), _tok("azi", "astăzi", "acum")],
        [_tok("ultima", "ultimul"), _tok("șansă", "avertisment", "notificare", "zi", "ocazie")],
        [_tok("oferta", "premiul", "linkul", "codul"), _tok("expiră")],
        [_tok("urgent", "imediat")],
    ],
    "phrase_action": [
        [_tok("click", "clic", "apasă", "apăsați", "intră", "intrați", "accesați", "accesează"),
         _tok("aici", "pe", "linkul")],
        [_tok("taxa", "taxă"), _tok("de"), _tok("livrare", "procesare", "transport", "deblocare", "eliberare",
                                                "relivrare", "vamală", "depozitare")],
        [_tok("plătește", "plătiți", "achitați", "achită"), _tok("doar", "acum", "taxa", "online", op="?")],
    ],
}


@lru_cache(maxsize=1)
def load_spacy() -> tuple[spacy.Language, Matcher]:
    nlp = spacy.load("ro_core_news_sm", disable=["parser"])
    matcher = Matcher(nlp.vocab)
    for name, patterns in PHRASE_PATTERNS.items():
        # Optional tokens yield several overlapping matches for one phrase; keep the longest.
        matcher.add(name, patterns, greedy="LONGEST")
    return nlp, matcher


@lru_cache(maxsize=1)
def load_rowordnet() -> tuple[frozenset[str], dict[str, tuple[float, float]]]:
    """Return RoWordNet's literals and the mean SentiWN (positive, negative) score per literal."""
    import rowordnet

    wn = rowordnet.RoWordNet()
    literals: set[str] = set()
    scores: dict[str, list[tuple[float, float]]] = {}
    for synset_id in wn.synsets():
        synset = wn.synset(synset_id)
        words = [literal.lower() for literal in synset.literals]
        literals.update(words)
        # Some synsets carry no SentiWN annotation; they still count for the word list.
        if synset.sentiwn is None:
            continue
        positive, negative, _ = synset.sentiwn
        for word in words:
            scores.setdefault(word, []).append((positive, negative))
    sentiment = {word: tuple(np.mean(values, axis=0)) for word, values in scores.items()}
    return frozenset(literals), sentiment


def prepare_text(text: str) -> str:
    return normalize_diacritics(strip_headers(strip_html(text)))


def _doc_features(doc, matcher: Matcher, literals: frozenset[str], sentiment: dict) -> dict:
    labels = [ent.label_ for ent in doc.ents]
    features = {"ner_money": labels.count("MONEY"), "ner_org": labels.count("ORGANIZATION")}

    for name in PHRASE_PATTERNS:
        features[name] = 0
    for match_id, _, _ in matcher(doc):
        features[doc.vocab.strings[match_id]] += 1

    scores, checked, unknown = [], 0, 0
    for token in doc:
        if token.like_url or token.like_email or token.like_num or token.pos_ == "PROPN":
            continue
        word = token.text.lower()
        lemma = token.lemma_.lower()
        if lemma in sentiment:
            scores.append(sentiment[lemma])
        # Only word-like tokens of 3+ characters: digits inside letters ("c4știg") count as unknown.
        if len(word) >= 3 and any(ch.isalpha() for ch in word):
            checked += 1
            if not ({word, lemma} & literals or word in STOPWORDS or lemma in STOPWORDS):
                unknown += 1

    features["sentiment_pos"], features["sentiment_neg"] = np.mean(scores, axis=0) if scores else (0.0, 0.0)

    text = doc.text
    # URLs are lowercase by convention and would dilute the ratio of a shouted message.
    letters = [ch for ch in URL_PATTERN.sub(" ", text) if ch.isalpha()]
    features["uppercase_ratio"] = sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0
    features["n_links"] = len(URL_PATTERN.findall(text))
    features["n_exclamations"] = text.count("!")
    features["oov_ratio"] = unknown / checked if checked else 0.0
    return features


def extract_features(texts: list[str]) -> pd.DataFrame:
    nlp, matcher = load_spacy()
    literals, sentiment = load_rowordnet()
    docs = nlp.pipe(prepare_text(t) for t in texts)
    return pd.DataFrame([_doc_features(doc, matcher, literals, sentiment) for doc in docs], columns=FEATURE_COLUMNS)
