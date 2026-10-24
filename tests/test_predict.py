import pytest

from src.models.predict import available_models, explain_email
from src.utils.config import load_config

CONFIG = load_config()

pytestmark = pytest.mark.skipif(
    "fastText" not in available_models(CONFIG), reason="needs the trained fastText model (python -m src.models.fasttext_lr)"
)


def test_tokens_point_back_into_the_cleaned_text():
    result = explain_email("Ați câștigat 500 de lei!\n\nIntrați pe http://premiu.ro acum.", "fastText", CONFIG, num_samples=200)
    assert 0.0 <= result.spam_probability <= 1.0
    assert len(result.weights) == len(result.tokens)
    for token in result.tokens:
        assert result.text[token.start:token.end] == token.text


def test_email_with_nothing_left_after_cleaning_is_rejected():
    with pytest.raises(ValueError):
        explain_email("   \n\n  ", "fastText", CONFIG)
