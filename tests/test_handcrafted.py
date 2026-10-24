import pytest

from src.features.handcrafted import FEATURE_COLUMNS, extract_features


@pytest.fixture(scope="module")
def features():
    texts = [
        "Contul dvs. a fost blocat. Ați câștigat! Click aici în 24 de ore: http://x-plata.com",
        "Contul tau a fost suspendat, ai castigat, confirma datele cardului",
        "Ne vedem mâine la cursul de statistică.",
        "c4știg GRATU1T b4ni rapizi",
        "<p>FELICITĂRI!!! <a href='http://a.com'>aici</a> și www.b.ro</p>",
    ]
    return extract_features(texts)


def test_all_columns_present(features):
    assert list(features.columns) == FEATURE_COLUMNS


def test_phishing_phrases_are_counted(features):
    row = features.iloc[0]
    assert row["phrase_account"] == 1
    assert row["phrase_prize"] == 1
    assert row["phrase_action"] == 1
    assert row["phrase_urgency"] == 1


def test_phrases_match_without_diacritics(features):
    row = features.iloc[1]
    assert row["phrase_account"] == 2
    assert row["phrase_prize"] == 1


def test_ordinary_email_has_no_phrase_matches(features):
    row = features.iloc[2]
    assert row[["phrase_account", "phrase_prize", "phrase_urgency", "phrase_action"]].sum() == 0


def test_obfuscated_words_raise_oov_ratio(features):
    assert features.iloc[3]["oov_ratio"] > 0.5
    assert features.iloc[2]["oov_ratio"] < 0.2


def test_surface_counts_on_html_email(features):
    row = features.iloc[4]
    assert row["n_links"] == 2
    assert row["n_exclamations"] == 3
    assert row["uppercase_ratio"] > 0.5
