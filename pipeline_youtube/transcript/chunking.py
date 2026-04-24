"""N-second window chunking for transcript snippets.

Implements decision (2) from the plan: window-based chunking instead of
punctuation-based. The rule is simple: start a new chunk whenever the
current chunk's span from its first snippet would exceed `window_seconds`
if we added the next snippet's start time.

This preserves snippet boundaries (we never split a snippet's text) and
produces natural-looking chunks similar to the dummy data in
`Permanent Note/08_YouTube学習/01_Scripts_Processing_Unit/`.

Optional filler-word stripping compresses transcripts before they are
sent to the LLM (cheaper cache footprint, less noise). The default
filler list is a conservative set of common Japanese hesitations; more
terms may be added via `config.json:filler_words`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import TranscriptSnippet

DEFAULT_FILLER_WORDS: tuple[str, ...] = (
    "えー",
    "えっと",
    "えーと",
    "えーっと",
    "あのー",
    "あの",
    "そのー",
    "まあ",
    "まー",
    "まぁ",
    "なんか",
    "みたいな",
    "っていう",
    "ていう",
    "という感じ",
)


@dataclass(frozen=True)
class Chunk:
    start: float  # seconds from video start
    text: str  # concatenated, whitespace-collapsed

    @property
    def mmss(self) -> str:
        """Format start as MM:SS for the markdown link label."""
        total = int(self.start)
        mm, ss = divmod(total, 60)
        return f"{mm:02d}:{ss:02d}"

    @property
    def start_int(self) -> int:
        """Integer seconds for the YouTube &t= query param."""
        return int(self.start)


def chunk_by_window(
    snippets: list[TranscriptSnippet],
    window_seconds: float = 30.0,
    *,
    filler_words: tuple[str, ...] | list[str] | None = None,
) -> list[Chunk]:
    """Group snippets into windows of roughly `window_seconds` each.

    Algorithm:
      - Start a new chunk when the next snippet's start time is at or
        beyond `chunk_start + window_seconds` (and the current chunk
        has at least one snippet so we don't emit empties).
      - Text is concatenated with single spaces and leading/trailing
        whitespace is stripped.
      - If `filler_words` is provided, each chunk text has filler terms
        and immediate duplicate tokens stripped before emission.

    Returns an empty list for empty input.
    """
    if not snippets:
        return []
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")

    fillers = tuple(filler_words) if filler_words is not None else ()
    chunks: list[Chunk] = []
    chunk_start: float = snippets[0].start
    chunk_texts: list[str] = []

    for snippet in snippets:
        if snippet.start >= chunk_start + window_seconds and chunk_texts:
            chunks.append(
                Chunk(start=chunk_start, text=_compress(_join_texts(chunk_texts), fillers))
            )
            chunk_start = snippet.start
            chunk_texts = []
        chunk_texts.append(snippet.text)

    if chunk_texts:
        chunks.append(Chunk(start=chunk_start, text=_compress(_join_texts(chunk_texts), fillers)))

    return chunks


def _join_texts(texts: list[str]) -> str:
    """Concatenate snippet texts, strip + collapse internal whitespace."""
    parts: list[str] = []
    for t in texts:
        stripped = " ".join(t.split())
        if stripped:
            parts.append(stripped)
    return " ".join(parts)


_DUP_WORD_RE = re.compile(r"(\S+?)(?:\s+\1){2,}")

# 2-token immediate repeat (e.g. ASR stutter "これ これ"), scoped to
# short Japanese tokens so legitimate English repetitions like
# "very very" are preserved.
_DUP_JP_SHORT_RE = re.compile(r"([\u3040-\u30FF\u4E00-\u9FFF]{1,4})\s+\1(?=\s|$)")


def _compress(text: str, fillers: tuple[str, ...]) -> str:
    """Strip filler words and collapse ASR-style token repeats.

    Steps:
      1. Remove filler tokens listed in `fillers`
      2. Collapse 3+ immediate repeats of any token (language-agnostic)
      3. Collapse 2-token immediate repeats of short Japanese tokens
         only (ASR stutter; leaves English like "very very" alone)
    """
    if not text:
        return text
    if fillers:
        for word in fillers:
            if not word:
                continue
            text = text.replace(word, " ")
        text = " ".join(text.split())
    text = _DUP_WORD_RE.sub(r"\1", text)
    text = _DUP_JP_SHORT_RE.sub(r"\1", text)
    return text
