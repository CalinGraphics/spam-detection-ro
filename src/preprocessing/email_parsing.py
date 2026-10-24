"""Turn a raw RFC 822 email into plain text: subject plus body, without headers or HTML."""

import email
import re
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from email.message import Message

from bs4 import BeautifulSoup

_BLANK_LINES = re.compile(r"\n{3,}")
_INLINE_SPACES = re.compile(r"[ \t\xa0]+")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def normalize_whitespace(text: str) -> str:
    lines = [_INLINE_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _decode_payload(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "latin-1"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        # Old spam often declares charsets Python doesn't know (e.g. "x-unknown").
        return payload.decode("latin-1", errors="replace")


def _decode_subject(message: Message) -> str:
    subject = message.get("Subject", "")
    try:
        return str(make_header(decode_header(subject)))
    except (UnicodeError, LookupError, HeaderParseError):
        return str(subject)


def extract_body(message: Message) -> str:
    """Return the text body, preferring text/plain parts and falling back to HTML."""
    plain_parts, html_parts = [], []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_decode_payload(part))
        elif content_type == "text/html":
            html_parts.append(_decode_payload(part))

    if plain_parts:
        body = "\n".join(plain_parts)
        # Some senders put HTML in a part labelled text/plain.
        if re.search(r"<(html|body|p|div|br|table)\b", body, re.IGNORECASE):
            body = html_to_text(body)
    else:
        body = "\n".join(html_to_text(html) for html in html_parts)
    return normalize_whitespace(body)


def parse_email(raw: bytes) -> str:
    """Parse raw email bytes into "subject\\n\\nbody"."""
    message = email.message_from_bytes(raw)
    subject = normalize_whitespace(_decode_subject(message))
    body = extract_body(message)
    return f"{subject}\n\n{body}".strip()
