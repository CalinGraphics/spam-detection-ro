"""Translate the English corpus to Romanian with a MarianMT model.

Emails are split into sentence-based segments that fit the model's input limit.
Every translated segment is appended to a JSONL cache keyed by the segment's hash,
so an interrupted run resumes where it stopped and repeated segments (mailing list
footers, spam templates) are translated only once.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from src.utils.config import load_config

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_HAS_LETTERS = re.compile(r"[A-Za-z]{2,}")

logger = logging.getLogger(__name__)


def segment_hash(segment: str) -> str:
    return hashlib.sha1(segment.encode("utf-8")).hexdigest()


def split_into_segments(text: str, tokenizer: MarianTokenizer, max_tokens: int) -> list[list[str]]:
    """Split text into paragraphs, each a list of segments of at most max_tokens tokens."""
    paragraphs = []
    for block in _PARAGRAPH_BREAK.split(text):
        # Plain-text emails are hard-wrapped at ~72 columns, so lines inside a
        # paragraph are joined before looking for sentence boundaries.
        block = " ".join(block.split())
        if not block:
            continue

        segments, current, current_len = [], [], 0
        for sentence in _SENTENCE_END.split(block):
            n_tokens = len(tokenizer.tokenize(sentence))
            if n_tokens > max_tokens:
                if current:
                    segments.append(" ".join(current))
                    current, current_len = [], 0
                segments.extend(_split_long_sentence(sentence, tokenizer, max_tokens))
                continue
            if current_len + n_tokens > max_tokens:
                segments.append(" ".join(current))
                current, current_len = [], 0
            current.append(sentence)
            current_len += n_tokens
        if current:
            segments.append(" ".join(current))
        paragraphs.append(segments)
    return paragraphs


def _split_long_sentence(sentence: str, tokenizer: MarianTokenizer, max_tokens: int) -> list[str]:
    # Fallback for run-on text without punctuation (common in spam): cut on words.
    chunks, current, current_len = [], [], 0
    for word in sentence.split():
        n_tokens = len(tokenizer.tokenize(word))
        if current and current_len + n_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(word)
        current_len += n_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def load_cache(path: Path) -> dict[str, str]:
    cache = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                # A crash mid-write can leave a truncated last line.
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache[entry["key"]] = entry["translation"]
    return cache


def translate_segments(
    segments: list[str],
    cache_path: Path,
    model_name: str,
    batch_size: int,
    num_beams: int,
) -> dict[str, str]:
    """Translate the segments missing from the cache and return the full cache."""
    cache = load_cache(cache_path)
    todo = {segment_hash(s): s for s in segments if _HAS_LETTERS.search(s)}
    todo = {k: s for k, s in todo.items() if k not in cache}
    logger.info("%d segments cached, %d to translate", len(cache), len(todo))
    if not todo:
        return cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()

    # Longest first: similar lengths in a batch mean less padding, and an
    # out-of-memory error shows up at the start instead of hours later.
    items = sorted(todo.items(), key=lambda kv: len(kv[1]), reverse=True)
    with open(cache_path, "a", encoding="utf-8") as cache_file:
        for start in tqdm(range(0, len(items), batch_size), desc="translating"):
            batch = items[start : start + batch_size]
            inputs = tokenizer(
                [s for _, s in batch], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            # On noisy input Marian sometimes loops on a phrase until max_length,
            # which stalls the whole batch; a translation rarely needs 1.5x the source.
            # Unbreakable strings (base64 blobs) can still fill the 512-token input,
            # so the cap also has to stay within the position embeddings.
            max_new_tokens = min(
                int(inputs["input_ids"].shape[1] * 1.5) + 10,
                model.config.max_position_embeddings - 2,
            )
            with torch.inference_mode():
                outputs = model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for (key, _), translation in zip(batch, translations):
                cache[key] = translation
                cache_file.write(json.dumps({"key": key, "translation": translation}, ensure_ascii=False) + "\n")
            cache_file.flush()
    return cache


def assemble(paragraphs: list[list[str]], cache: dict[str, str]) -> str:
    # Segments without letters (separators, bare numbers) were never sent to the model.
    return "\n\n".join(
        " ".join(cache.get(segment_hash(s), s) for s in segments) for segments in paragraphs
    )


def translate_corpus(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    params = config["translation"]
    tokenizer = MarianTokenizer.from_pretrained(params["model_name"])
    paragraphs = [
        split_into_segments(text, tokenizer, params["max_segment_tokens"])
        for text in tqdm(df["text"], desc="segmenting")
    ]
    all_segments = [s for doc in paragraphs for segs in doc for s in segs]

    cache = translate_segments(
        all_segments,
        cache_path=config["paths"]["interim"] / "translation_cache.jsonl",
        model_name=params["model_name"],
        batch_size=params["batch_size"],
        num_beams=params["num_beams"],
    )

    out = df.rename(columns={"text": "text_en"})
    out.insert(0, "text", [assemble(doc, cache) for doc in paragraphs])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    interim = config["paths"]["interim"]
    df = pd.read_parquet(interim / "spamassassin_en.parquet")
    translated = translate_corpus(df, config)
    translated.to_parquet(interim / "spamassassin_ro.parquet", index=False)
    logger.info("saved %s", interim / "spamassassin_ro.parquet")
