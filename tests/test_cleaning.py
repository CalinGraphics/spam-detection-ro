from src.preprocessing.cleaning import (
    EMAIL_TOKEN,
    MONEY_TOKEN,
    NUMBER_TOKEN,
    URL_TOKEN,
    clean_text,
    normalize_diacritics,
    replace_entities,
    strip_headers,
    strip_html,
)
from src.preprocessing.nlp import stem_texts

ALL_ON = {
    "normalize_diacritics": True,
    "replace_urls": True,
    "replace_emails": True,
    "replace_money": True,
    "replace_numbers": True,
}


def test_cedilla_diacritics_become_comma_below():
    assert normalize_diacritics("Aţi câştigat! ŞTIRE ŢARĂ") == "Ați câștigat! ȘTIRE ȚARĂ"


def test_correct_diacritics_are_unchanged():
    assert normalize_diacritics("ăâîșț ĂÂÎȘȚ") == "ăâîșț ĂÂÎȘȚ"


def test_html_is_stripped_and_link_target_kept():
    text = strip_html('<p>Actualizați <b>acum</b>: <a href="http://x-plata.com">aici</a></p><script>x()</script>')
    assert "<" not in text and "x()" not in text
    assert "Actualizați" in text and "acum" in text
    assert "http://x-plata.com" in text


def test_html_entities_are_unescaped_in_plain_text():
    assert strip_html("Preț&nbsp;mic &amp; bun") == "Preț\xa0mic & bun"


def test_leading_raw_header_block_is_removed():
    raw = "From: a@b.ro\nTo: c@d.ro\nSubject: Salut\nContent-Type: text/plain\n\nCorpul mesajului."
    assert strip_headers(raw) == "Corpul mesajului."


def test_forwarded_headers_are_removed():
    text = (
        "---------- Forwarded message ---------\n"
        "From: Secretariat\n"
        "Date: luni, 9 ian.\n"
        "Subject: program sesiune\n"
        "\n"
        "Programul a fost afișat."
    )
    assert strip_headers(text).strip() == "Programul a fost afișat."


def test_single_header_like_line_in_body_is_kept():
    text = "Programarea a fost confirmată:\nData: 18.11, ora 08:15\nLocație: Bd. Iuliu Maniu 7"
    assert "Data: 18.11, ora 08:15" in strip_headers(text)


def test_reply_quote_markers_are_removed():
    assert strip_headers("> Se știe când e întâlnirea?\n>> Marius") == "Se știe când e întâlnirea?\nMarius"


def test_subject_with_colon_is_not_treated_as_header_block():
    text = "URGENT: Ai câștigat\n\nRevendică premiul."
    assert strip_headers(text) == text


def test_urls_and_emails_are_replaced():
    text = replace_entities("Scrie la plata@banca-x.ro sau intră pe https://banca-x.ro/login?id=1.")
    assert text.count(EMAIL_TOKEN) == 1 and text.count(URL_TOKEN) == 1
    assert "@" not in text and "http" not in text
    assert text.rstrip().endswith(".")


def test_bare_domains_and_www_are_replaced():
    text = replace_entities("Comandă: www.slimdrops.com sau banirapizi.xyz acum")
    assert text.count(URL_TOKEN) == 2


def test_money_amounts_are_replaced():
    text = replace_entities("Taxa de 2,99 lei, premiul de 1.000.000 EUR, $1,000 și 250 de euro.")
    assert text.count(MONEY_TOKEN) == 4
    assert NUMBER_TOKEN not in text


def test_plain_numbers_are_replaced():
    text = replace_entities("AWB 4821903377, ora 08:15, în 24 de ore")
    assert text.count(NUMBER_TOKEN) == 3


def test_obfuscated_words_are_not_split_by_number_replacement():
    text = replace_entities("c4știg GRATU1T V1agra")
    assert text == "c4știg GRATU1T V1agra"


def test_replacements_can_be_disabled():
    text = "Plătește 5 lei pe https://x.ro"
    assert replace_entities(text, urls=False, emails=False, money=False, numbers=False) == text


def test_clean_text_keeps_case_and_collapses_whitespace():
    text = clean_text("Câştigaţi   100 LEI\n\n\n\nacum pe www.premiu.ro", ALL_ON)
    assert text == f"Câștigați {MONEY_TOKEN}\n\nacum pe {URL_TOKEN}"


def test_stemming_removes_stopwords_and_keeps_placeholders():
    [stemmed] = stem_texts([f"Ați câștigat premiile noastre pe {URL_TOKEN}"])
    assert stemmed.split() == ["câștig", "prem", URL_TOKEN]


def test_stemming_drops_hyphenated_clitics_that_are_stopwords():
    [stemmed] = stem_texts(["contactați-ne urgent"])
    assert "-ne" not in stemmed.split() and "ne" not in stemmed.split()


def test_stemming_keeps_stopwords_when_disabled():
    [stemmed] = stem_texts(["Ați câștigat"], remove_stopwords=False)
    assert stemmed.split()[0] == "ați"
