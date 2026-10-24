import numpy as np

from src.features.embeddings import document_vectors, tokenize
from src.preprocessing.cleaning import URL_TOKEN


def test_tokenize_drops_stopwords_and_maps_placeholders():
    assert tokenize(f"Ați câștigat un premiu pe {URL_TOKEN}!") == ["câștigat", "premiu", "link"]


def test_document_vector_is_mean_of_known_words_with_cedilla_fallback():
    index = {"premiu": 0, "câştigat": 1}
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vectors, coverage = document_vectors(["câștigat premiu necunoscut"], (index, matrix))
    np.testing.assert_allclose(vectors[0], [0.5, 0.5])
    assert coverage == 2 / 3


def test_text_without_known_words_gives_zero_vector():
    vectors, coverage = document_vectors(["xyzabc"], ({}, np.zeros((0, 3), dtype=np.float32)))
    assert not vectors.any() and coverage == 0.0
