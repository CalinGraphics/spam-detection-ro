"""Demo: paste an email, pick a model, see the verdict and the words behind it.

Run from the project root: streamlit run app/app.py
"""

import html
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts app/ on the import path, not the project root that holds src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import Explanation, available_models, explain_email  # noqa: E402
from src.preprocessing.cleaning import EMAIL_TOKEN, MONEY_TOKEN, NUMBER_TOKEN, URL_TOKEN  # noqa: E402
from src.utils.config import load_config  # noqa: E402

PLACEHOLDER_LABELS = {URL_TOKEN: "link", EMAIL_TOKEN: "email address", MONEY_TOKEN: "amount", NUMBER_TOKEN: "number"}

MODEL_NOTES = {
    "Logistic Regression": "TF-IDF on lemmas. Best precision on the test set, and fast.",
    "Naive Bayes": "TF-IDF on lemmas. The simplest baseline, misses more short scams.",
    "fastText": "Averaged cc.ro.300 word vectors. Weakest model, kept for comparison.",
    "BERT": "Fine-tuned Romanian BERT. Same test score as logistic regression, slower.",
}

EXAMPLES = {
    "Phishing": (
        "Plata coletului dvs. este în așteptare\n\n"
        "Bună ziua,\n\n"
        "Coletul dumneavoastră nu poate fi livrat până la achitarea taxei de procesare de 2,49 lei. "
        "Plătiți în următoarele 24 de ore pentru a evita returnarea:\n"
        "https://livrare-colet-confirmare.com\n\n"
        "Serviciul Clienți"
    ),
    "Legitimate": (
        "Programare consultație\n\n"
        "Bună ziua,\n\n"
        "Vă confirmăm programarea de joi, 14 martie, la ora 10:30, la cabinetul dr. Ionescu. "
        "Dacă nu puteți ajunge, vă rugăm să ne anunțați cu o zi înainte.\n\n"
        "Cu stimă,\nClinica Medicală Sfântul Luca"
    ),
}

SPAM_RGB, HAM_RGB = "227, 73, 72", "42, 120, 214"

CSS = """
<style>
  .block-container { max-width: 760px; padding-top: 3rem; }
  .eyebrow { font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; color: #898781; margin: 0; }
  .title { font-size: 2.1rem; line-height: 1.1; letter-spacing: -0.02em; font-weight: 700; color: #0b0b0b;
           margin: 0.35rem 0 0.5rem; }
  .lede { color: #52514e; font-size: 1rem; line-height: 1.5; margin: 0 0 1.5rem; }

  .verdict { display: flex; align-items: center; gap: 1.25rem; padding: 1.1rem 1.25rem; border-radius: 12px;
             background: #ffffff; border: 1px solid #e1e0d9; margin: 0.5rem 0 1.5rem; }
  .badge { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; color: #ffffff; padding: 0.35rem 0.7rem;
           border-radius: 999px; white-space: nowrap; }
  .badge.spam { background: #c03030; }
  .badge.ham { background: #006300; }
  .verdict-main { flex: 1; min-width: 0; }
  .probability { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; color: #0b0b0b;
                 font-variant-numeric: tabular-nums; line-height: 1.2; }
  .probability small { font-size: 0.9rem; font-weight: 400; color: #52514e; letter-spacing: 0; }
  .meter { height: 6px; border-radius: 3px; background: #e1e0d9; margin-top: 0.5rem; overflow: hidden; }
  .meter span { display: block; height: 100%; border-radius: 3px; background: #52514e; }

  .section { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
             color: #52514e; margin: 0 0 0.5rem; }
  .email-view { white-space: pre-wrap; line-height: 1.75; font-size: 0.95rem; color: #0b0b0b; background: #ffffff;
                border: 1px solid #e1e0d9; border-radius: 12px; padding: 1rem 1.1rem; overflow-wrap: anywhere; }
  .email-view mark { color: inherit; border-radius: 4px; padding: 0.05em 0.15em; }
  .chip { font-size: 0.78em; color: #52514e; border: 1px solid #c3c2b7; border-radius: 4px; padding: 0 0.3em; }
  .legend { display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.8rem; color: #52514e; margin: 0.6rem 0 0; }
  .swatch { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 3px; vertical-align: -0.1rem;
            margin-right: 0.35rem; }

  ul.signals { list-style: none; padding: 0; margin: 0; }
  .signals li { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 0.15rem 0.75rem;
                padding: 0.45rem 0; border-bottom: 1px solid #ecebe6; font-size: 0.92rem; }
  .signals .word { color: #0b0b0b; overflow-wrap: anywhere; }
  .signals .direction { color: #52514e; font-size: 0.8rem; }
  .signals .bar { grid-column: 1 / -1; height: 4px; border-radius: 2px; }
  .footnote { color: #898781; font-size: 0.8rem; line-height: 1.5; margin-top: 2.5rem; }
</style>
"""


def is_word(text: str) -> bool:
    # Punctuation is part of what LIME perturbs, but a tinted comma reads as noise.
    return text in PLACEHOLDER_LABELS or any(ch.isalnum() for ch in text)


def display_word(text: str) -> str:
    if text in PLACEHOLDER_LABELS:
        return f'<span class="chip">{PLACEHOLDER_LABELS[text]}</span>'
    return html.escape(text)


def highlighted_email(result: Explanation) -> str:
    peak = max((abs(w) for t, w in zip(result.tokens, result.weights) if is_word(t.text)), default=0.0)
    parts, cursor = [], 0
    for token, weight in zip(result.tokens, result.weights):
        parts.append(html.escape(result.text[cursor:token.start]))
        word = display_word(result.text[token.start:token.end])
        # Only the clearly influential words get a tint; the rest stays plain text.
        if peak > 0 and abs(weight) >= 0.25 * peak and is_word(token.text):
            alpha = 0.12 + 0.48 * abs(weight) / peak
            rgb = SPAM_RGB if weight > 0 else HAM_RGB
            direction = "spam" if weight > 0 else "ham"
            word = (f'<mark style="background: rgba({rgb}, {alpha:.2f})" '
                    f'title="pushes towards {direction} ({weight:+.3f})">{word}</mark>')
        parts.append(word)
        cursor = token.end
    parts.append(html.escape(result.text[cursor:]))
    return f'<div class="email-view">{"".join(parts)}</div>'


def signal_list(result: Explanation, limit: int = 6) -> str:
    ranked = sorted(zip(result.tokens, result.weights), key=lambda item: abs(item[1]), reverse=True)
    ranked = [(token, weight) for token, weight in ranked if weight != 0 and is_word(token.text)][:limit]
    if not ranked:
        return '<p class="lede">No single word stands out for this email.</p>'
    peak = abs(ranked[0][1])
    rows = []
    for token, weight in ranked:
        rgb = SPAM_RGB if weight > 0 else HAM_RGB
        direction = "towards spam" if weight > 0 else "towards ham"
        rows.append(
            f'<li><span class="word">{display_word(token.text)}</span>'
            f'<span class="direction">{direction}</span>'
            f'<span class="bar" style="width: {100 * abs(weight) / peak:.0f}%; background: rgb({rgb})"></span></li>'
        )
    return f'<ul class="signals">{"".join(rows)}</ul>'


def verdict(result: Explanation, model: str) -> str:
    probability = result.spam_probability
    is_spam = probability >= 0.5
    label, css = ("SPAM", "spam") if is_spam else ("NOT SPAM", "ham")
    return (
        f'<div class="verdict"><span class="badge {css}">{label}</span>'
        f'<div class="verdict-main"><div class="probability">{100 * probability:.1f}% '
        f"<small>spam probability · {html.escape(model)}</small></div>"
        f'<div class="meter" role="meter" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{100 * probability:.0f}"><span style="width: {100 * probability:.1f}%"></span></div>'
        f"</div></div>"
    )


def load_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        st.session_state["email"] = EXAMPLES[choice]
    # The pills act as one-shot actions; a lingering selection would look wrong once the text is edited.
    st.session_state["example"] = None


def main() -> None:
    st.set_page_config(page_title="Romanian spam detector", layout="centered")
    st.html(CSS)
    config = load_config()

    st.html(
        '<p class="eyebrow">NLP &amp; Text Mining</p>'
        '<h1 class="title">Romanian spam detector</h1>'
        '<p class="lede">Paste an email, subject on the first line. The model gives a spam probability '
        "and highlights the words that weighed most in its decision.</p>"
    )

    models = available_models(config)
    if not models:
        st.error("No trained model found in models/. Run `python -m src.models.baselines` first.")
        st.stop()

    model = st.segmented_control("Model", models, default=models[0], key="model", required=True)
    st.caption(MODEL_NOTES[model])

    st.pills("Try an example", list(EXAMPLES), key="example", on_change=load_example)
    email = st.text_area(
        "Email", key="email", height=220, max_chars=5000,
        placeholder="Subject on the first line, then the body of the email…",
    )

    if st.button("Check email", type="primary", disabled=not email.strip()):
        with st.spinner("Scoring the email and explaining the result…"):
            try:
                st.session_state["result"] = (email, model, explain_email(email, model, config))
            except ValueError as error:
                st.session_state.pop("result", None)
                st.warning(f"Nothing to classify: {error}.")

    stored = st.session_state.get("result")
    # A result belongs to one email and one model; editing either hides it until the next check.
    if stored and stored[0] == email and stored[1] == model:
        result = stored[2]
        st.html(verdict(result, model))
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.html(
                '<p class="section">What the model read</p>' + highlighted_email(result)
                + '<div class="legend">'
                f'<span><span class="swatch" style="background: rgba({SPAM_RGB}, 0.5)"></span>towards spam</span>'
                f'<span><span class="swatch" style="background: rgba({HAM_RGB}, 0.5)"></span>towards ham</span>'
                "</div>"
            )
        with right:
            st.html('<p class="section">Strongest signals</p>' + signal_list(result))

    st.html(
        '<p class="footnote">Trained on 974 Romanian emails generated for this project, so scores on a real '
        "inbox will be lower. Word weights come from LIME (1000 perturbed copies of the email) and show "
        "what drove this prediction, not general rules. Links, amounts and numbers are replaced by "
        "placeholders before the model sees the text.</p>"
    )


main()
