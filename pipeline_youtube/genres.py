"""Genre classification (Router) for downstream pipeline behavior.

Stage 00.5 — placed between playlist metadata fetch and per-video processing.

Why
---
The pipeline's downstream stages (especially Stage 04 learning material
generation) want to behave differently for *coding* tutorials vs purely
*conceptual* content. A coding playlist benefits from:

- GitHub URL extraction from descriptions
- Concept-vs-Practice section split in 04 prompts
- (future) Code-block forced extraction

A philosophy lecture playlist would be hurt by all three. Rather than
trigger these heuristics from per-video fragile keyword matching, this
module classifies the *whole playlist* once with a single cheap LLM call
(haiku) and downstream features gate on the result.

Design
------
- Single `classify_playlist_genre()` entry point. One LLM call per
  playlist, ~$0.01 cost.
- Output is a `Genre` enum + free-text rationale (for logs).
- All errors (network, parse) collapse to `Genre.OTHER` so downstream
  features default to current behavior. The router is *advisory*, never
  blocking.
"""

from __future__ import annotations

import json
from enum import StrEnum

from .playlist import VideoMeta
from .providers.base import LLMError as ClaudeCliError
from .providers.registry import invoke_llm as invoke_claude
from .sanitize import sanitize_untrusted_text, wrap_untrusted


class Genre(StrEnum):
    """Coarse playlist genre.

    Designed to be stable across pipeline versions — adding new values
    is fine, renaming existing ones is a breaking change for any
    consumer that compares against `Genre.CODING.value`.
    """

    CODING = "coding"
    BUSINESS = "business"
    HUMANITIES = "humanities"
    SCIENCE = "science"
    LIFESTYLE = "lifestyle"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


# Genres for which the pipeline injects code-bearing enhancements
# (GitHub URL extraction, concept/practice split, etc.).
CODE_BEARING_GENRES: frozenset[Genre] = frozenset({Genre.CODING})


_ROUTER_SYSTEM_PROMPT = """あなたは YouTube プレイリストのジャンル分類器です。
入力されたプレイリスト名と動画タイトル群から **ひとつ** のジャンルを選び、JSON で返してください。
出力 JSON 以外の説明文は禁止。

利用可能なジャンル:
- coding: プログラミング・開発ツール・AI/コード生成 (Claude Code, Copilot, Cursor 等)
- business: 経営・マーケティング・生産性・起業
- humanities: 哲学・歴史・文学・芸術・宗教
- science: 数学・物理・化学・生物・宇宙
- lifestyle: 料理・旅行・健康・ファッション
- entertainment: ゲーム・音楽・アニメ・Vlog
- other: 上記いずれにも当てはまらない

出力形式: {"genre": "coding", "rationale": "理由を1〜2文で"}"""


# Cap the number of titles passed to the LLM. Most playlists fit well
# within this; very large playlists get a representative sample (first N).
_MAX_TITLES_IN_PROMPT = 30


def classify_playlist_genre(
    playlist_title: str,
    videos: list[VideoMeta],
    *,
    model: str = "haiku",
) -> tuple[Genre, str]:
    """Classify a playlist into a single Genre.

    Returns ``(genre, rationale)``. On any error (network, parse, unknown
    genre value) returns ``(Genre.OTHER, "<error description>")`` so the
    caller can proceed with default behavior.

    The function is intentionally side-effect free; the caller decides
    whether to log the result.
    """
    if not videos:
        return Genre.OTHER, "no videos"

    safe_title = sanitize_untrusted_text(
        playlist_title or "Untitled Playlist",
        300,
        context="genres.playlist_title",
    )
    title_lines: list[str] = []
    for v in videos[:_MAX_TITLES_IN_PROMPT]:
        safe = sanitize_untrusted_text(v.title or "(no title)", 200, context="genres.video_title")
        title_lines.append(f"- {safe}")
    titles_block = "\n".join(title_lines)

    extra = ""
    if len(videos) > _MAX_TITLES_IN_PROMPT:
        extra = f"\n(計 {len(videos)} 本中、先頭 {_MAX_TITLES_IN_PROMPT} 本を表示)"

    prompt = (
        f"プレイリスト: {safe_title}\n"
        f"動画タイトル ({min(len(videos), _MAX_TITLES_IN_PROMPT)} 本):\n"
        f"{wrap_untrusted(titles_block)}{extra}"
    )

    try:
        response = invoke_claude(
            prompt=prompt,
            system_prompt=_ROUTER_SYSTEM_PROMPT,
            model=model,
            max_retries=2,
        )
    except ClaudeCliError as e:
        return Genre.OTHER, f"router_call_failed: {str(e)[:200]}"

    text = response.text.strip()
    # Strip code fences if the model wrapped output despite instructions.
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()

    try:
        data = json.loads(text)
        genre_str = str(data["genre"]).lower()
        genre = Genre(genre_str)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return Genre.OTHER, f"router_parse_failed: {type(e).__name__}: {str(e)[:100]}"

    rationale = str(data.get("rationale", ""))[:300]
    return genre, rationale
