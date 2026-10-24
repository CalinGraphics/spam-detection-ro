from src.preprocessing.email_parsing import parse_email

PLAIN = b"""From: someone@example.com
To: list@example.com
Subject: Meeting tomorrow
Content-Type: text/plain; charset="us-ascii"

Hi all,

   the meeting   is at 10.



See you.
"""

HTML_ONLY = b"""From: promo@example.com
Subject: =?iso-8859-1?q?Special_offer_f=FCr_you?=
Content-Type: text/html; charset="iso-8859-1"

<html><head><style>p {color: red}</style></head>
<body><p>Click <b>here</b> now!</p><script>alert(1)</script></body></html>
"""

MULTIPART = b"""From: a@example.com
Subject: Both parts
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="XYZ"

--XYZ
Content-Type: text/plain; charset="utf-8"

Plain version.
--XYZ
Content-Type: text/html; charset="utf-8"

<p>HTML version.</p>
--XYZ--
"""

UNKNOWN_CHARSET = b"""Subject: Weird
Content-Type: text/plain; charset="x-unknown"

Caf\xe9 offer.
"""


def test_plain_email_drops_headers_and_extra_whitespace():
    assert parse_email(PLAIN) == "Meeting tomorrow\n\nHi all,\n\nthe meeting is at 10.\n\nSee you."


def test_html_only_email_is_converted_to_text():
    text = parse_email(HTML_ONLY)
    assert text.startswith("Special offer für you")
    assert "Click" in text and "here" in text and "now!" in text
    assert "<" not in text and "alert" not in text and "color" not in text


def test_multipart_alternative_prefers_plain_part():
    text = parse_email(MULTIPART)
    assert "Plain version." in text
    assert "HTML version." not in text


def test_unknown_charset_falls_back_to_latin1():
    assert "Café offer." in parse_email(UNKNOWN_CHARSET)
