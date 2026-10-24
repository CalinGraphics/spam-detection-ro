"""TF-IDF features: word 1-2 grams plus character 3-5 grams."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf(params: dict) -> FeatureUnion:
    """Build the vectorizer from the `tfidf` section of config.yaml.

    The input is already lowercased by preprocessing, so lowercase=False keeps the
    uppercase placeholder tokens (URLTOKEN, ...) distinct from ordinary words.
    """
    common = {"max_features": params["max_features"], "min_df": params["min_df"], "sublinear_tf": True, "lowercase": False}
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(params["word_ngram_range"]), **common)),
            # char_wb keeps n-grams inside word boundaries, which is where the
            # morphology and the obfuscations ("c4știg") are.
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(params["char_ngram_range"]), **common)),
        ]
    )
