"""Text cleaning applied before any tokenization: HTML, email headers, entity placeholders, diacritics."""

import html
import re

from bs4 import BeautifulSoup

URL_TOKEN = "URLTOKEN"
EMAIL_TOKEN = "EMAILTOKEN"
MONEY_TOKEN = "MONEYTOKEN"
NUMBER_TOKEN = "NUMTOKEN"

_HTML_TAG = re.compile(r"<\s*/?\s*(html|body|head|p|div|br|a|table|tr|td|span|b|i|u|ul|li|h\d|small|font|img)\b", re.I)

# A raw email pasted with its headers: "Key: value" lines (plus folded continuation lines) up to the first blank line.
_LEADING_HEADER_BLOCK = re.compile(r"\A(?:[A-Za-z][\w-]*:[^\n]*\n(?:[ \t]+[^\n]*\n)*)+\n")
_RFC822_KEYS = re.compile(r"^(From|Received|Return-Path|Message-ID|To):", re.I | re.M)

_FORWARD_SEPARATOR = re.compile(
    r"^-{2,}\s*(Forwarded message|Original Message|Mesaj redirecționat|Mesaj original)\s*-{2,}\s*$", re.I | re.M
)
_HEADER_LINE = re.compile(
    r"^(From|To|Cc|Bcc|Date|Sent|Subject|Reply-To|De la|Către|Data|Trimis|Subiect):\s", re.I
)
_QUOTE_MARKER = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.M)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']*[^\s<>\"'.,;:!?)]"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|ro|net|org|info|biz|eu|io|co|xyz|top|shop|online|vip|app|ru)\b(?:/[^\s<>\"']*[^\s<>\"'.,;:!?)])?",
    re.I,
)
_AMOUNT = r"\d{1,3}(?:[.,]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
_MONEY = re.compile(
    rf"[$€£]\s?(?:{_AMOUNT})"
    rf"|(?:{_AMOUNT})\s?(?:de\s)?(?:[$€£]|(?:lei|ron|eur|euro|usd|usdt|dolari|lire sterline|lire)\b)",
    re.I,
)
# Standalone numbers only, so obfuscated words like "c4știg" or "GRATU1T" stay intact.
_NUMBER = re.compile(r"\b\d+(?:[.,:/]\d+)*\b")

_CEDILLA_TO_COMMA = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})


def strip_html(text: str) -> str:
    text = html.unescape(text)
    if not _HTML_TAG.search(text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for link in soup.find_all("a", href=True):
        # Keep the link target, it is the main spam signal of an HTML anchor.
        link.append(f" {link['href']}")
    return soup.get_text(separator="\n")


def strip_headers(text: str) -> str:
    """Remove a leading raw header block, forwarded-message headers and reply quote markers."""
    match = _LEADING_HEADER_BLOCK.match(text)
    if match and _RFC822_KEYS.search(match.group()):
        text = text[match.end():]

    text = _FORWARD_SEPARATOR.sub("", text)
    text = _QUOTE_MARKER.sub("", text)

    # Header lines only count inside a run of two or more, so a body line such as
    # "Data: 18.11, ora 08:15" in an appointment email is kept.
    lines = text.split("\n")
    is_header = [bool(_HEADER_LINE.match(line)) for line in lines]
    kept = [
        line
        for i, line in enumerate(lines)
        if not is_header[i]
        or not ((i > 0 and is_header[i - 1]) or (i + 1 < len(lines) and is_header[i + 1]))
    ]
    return "\n".join(kept)


def replace_entities(
    text: str, urls: bool = True, emails: bool = True, money: bool = True, numbers: bool = True
) -> str:
    # Order matters: an email address contains a domain, and URLs/amounts contain digits.
    if emails:
        text = _EMAIL.sub(f" {EMAIL_TOKEN} ", text)
    if urls:
        text = URL_PATTERN.sub(f" {URL_TOKEN} ", text)
    if money:
        text = _MONEY.sub(f" {MONEY_TOKEN} ", text)
    if numbers:
        text = _NUMBER.sub(f" {NUMBER_TOKEN} ", text)
    return text


def normalize_diacritics(text: str) -> str:
    """Map the legacy cedilla forms ş/ţ to the correct comma-below ș/ț."""
    return text.translate(_CEDILLA_TO_COMMA)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(text: str, config: dict) -> str:
    """Apply the cleaning steps enabled in the `preprocessing` section of config.yaml.

    Case is preserved: lowercasing happens after lemmatization/stemming, and the
    cased text is what the BERT model and the hand-crafted features use.
    """
    text = strip_headers(strip_html(text))
    if config["normalize_diacritics"]:
        text = normalize_diacritics(text)
    text = replace_entities(
        text,
        urls=config["replace_urls"],
        emails=config["replace_emails"],
        money=config["replace_money"],
        numbers=config["replace_numbers"],
    )
    return normalize_whitespace(text)
