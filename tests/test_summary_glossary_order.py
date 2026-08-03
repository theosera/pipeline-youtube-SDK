"""Glossary substitution must happen before validation (S17 / PR-42).

``normalize_text`` splices glossary canonicals into the body. Running it after
``_validate_summary_output`` meant whatever the glossary held went to disk
unchecked — and the glossary is persistent (``glossary.json``), so one bad
canonical reaches every later summary, not just the run that introduced it.

Every assertion below reads the file Stage 02 actually wrote, so moving the
substitution back after validation — or calling it without keeping the result
— turns them red. A ``normalize_text`` unit test would not.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.glossary.schema import Glossary, GlossaryEntry, load_glossary
from pipeline_youtube.pipeline import create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import summary as summary_stage
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_NO_CACHE = Cache(None, enabled=False)

# The alias the model emits; each test swaps in a different canonical for it.
_ALIAS = "ビブコーディング"

_MODEL_OUTPUT = (
    f"ONE_LINER: {_ALIAS}入門\n\n"
    "## 全体サマリ\n\n"
    f"本動画は{_ALIAS}の基礎を解説する。\n\n"
    "## 要点タイムライン\n\n"
    f"### [00:00 ~ 01:30] {_ALIAS}とは\n"
    f"{_ALIAS}の定義を述べる。\n"
)


@pytest.fixture
def vault(tmp_path: Path):
    config.set_dry_run(False)
    yield tmp_path


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="_h3decBW12Q",
        title="ハーネス設計解説",
        url="https://www.youtube.com/watch?v=_h3decBW12Q",
        duration=932,
        channel="AI Channel",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


def _fake_response(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=120,
        output_tokens=300,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=0.005,
        duration_ms=4500,
        session_id="fake-session",
        stop_reason="end_turn",
    )


def _glossary(canonical: str) -> Glossary:
    return Glossary(entries=(GlossaryEntry(canonical=canonical, aliases=[_ALIAS]),))


def _run(vault: Path, monkeypatch, glossary: Glossary | None) -> tuple[str, str]:
    """Run Stage 02 and return the written note as ``(frontmatter, body)``.

    Split because the two halves have different guarantees: the body is what
    this change routes through the validator, while the ``one_liner``
    frontmatter field is extracted upstream of validation and never sees it
    (pinned in ``TestOneLinerStillBypassesValidation``).
    """
    video = _video()
    paths = create_placeholder_notes(
        video, datetime(2026, 4, 14, 21, 41), dry_run=False, vault_root=vault
    )
    transcript = build_result(
        video_id="_h3decBW12Q",
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[TranscriptSnippet(f"{_ALIAS}の導入", 0.0, 30.0)],
    )
    monkeypatch.setattr(summary_stage, "invoke_claude", lambda **kw: _fake_response(_MODEL_OUTPUT))
    summary_stage.run_stage_summary(
        video, paths["summary"], transcript, glossary=glossary, cache=_NO_CACHE
    )
    frontmatter, _, body = paths["summary"].read_text(encoding="utf-8").partition("\n---\n")
    return frontmatter, body


class TestSubstitutedCanonicalsAreValidated:
    """A hostile canonical must be sanitized, not written through."""

    def test_templater_token_in_a_canonical_is_stripped(self, vault, monkeypatch):
        canonical = '<%* await app.vault.adapter.write("x.md", "pwned") %>'
        _, body = _run(vault, monkeypatch, _glossary(canonical))

        assert "<%*" not in body
        assert "app.vault.adapter" not in body

    def test_active_html_in_a_canonical_is_stripped(self, vault, monkeypatch):
        _, body = _run(vault, monkeypatch, _glossary("<iframe src='https://evil.example'>VC"))

        assert "<iframe" not in body
        assert "VC" in body  # the benign remainder survives

    def test_disallowed_embed_in_a_canonical_is_dropped(self, vault, monkeypatch):
        _, body = _run(vault, monkeypatch, _glossary("![[../../secret]]"))

        assert "![[../../secret]]" not in body
        assert "dropped embed" in body

    def test_a_canonical_from_glossary_json_is_validated_the_same(
        self, vault, monkeypatch, tmp_path
    ):
        """The persistent dictionary is the path that outlives a single run.

        ``_promote_corrections_to_glossary`` writes here, and every later
        summary reads it — so a value that lands in this file is spliced into
        all of them, not just the run that introduced it.
        """
        path = tmp_path / "glossary.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "canonical": "<script>fetch('https://evil.example')</script>VC",
                            "aliases": [_ALIAS],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _, body = _run(vault, monkeypatch, load_glossary(path))

        assert "<script" not in body
        assert "VC" in body


class TestLegitimateCanonicalsSurvive:
    """The risk of validating last: a real term must not be mangled.

    This is the one thing the reordering could plausibly break, so the terms
    here are shaped like real glossary entries rather than minimal strings.
    """

    @pytest.mark.parametrize(
        "canonical",
        [
            "Vibe Coding",
            "バイブコーディング",
            "Claude Code (Anthropic)",
            "C++ / Rust 併用",
            "AI駆動開発 — 実践編",
            "https://docs.anthropic.com/claude-code",
            "田中/佐藤 [監修]",
            "100% 自動化",
        ],
    )
    def test_canonical_reaches_disk_verbatim(self, vault, monkeypatch, canonical):
        _, body = _run(vault, monkeypatch, _glossary(canonical))

        assert canonical in body
        assert _ALIAS not in body

    def test_no_glossary_leaves_the_body_alone(self, vault, monkeypatch):
        _, body = _run(vault, monkeypatch, None)

        assert _ALIAS in body


class TestOneLinerStillNormalizes:
    """Normalizing the whole body up front must keep covering the one-liner.

    It used to be normalized by its own ``normalize_text`` call in the caller;
    that call is gone, so this pins that folding the substitution into the
    pre-validation pass did not quietly drop it.
    """

    def test_one_liner_gets_the_canonical(self, vault, monkeypatch):
        frontmatter, _ = _run(vault, monkeypatch, _glossary("Vibe Coding"))

        assert 'one_liner: "Vibe Coding入門"' in frontmatter


class TestOneLinerStillBypassesValidation:
    """Pins a gap this change does *not* close, measured against the parent.

    ``_extract_one_liner`` runs before ``_validate_summary_output``, so the
    one-liner never reaches ``validate_chapter_body`` — only a homoglyph fold
    and ``_escape_yaml`` on the way into frontmatter. That is a missing check,
    not a mis-ordered one, so it is outside this change's perspective.

    Measured on the parent commit and on this one with the same hostile
    canonical:

        parent : frontmatter=True  body=True
        here   : frontmatter=True  body=False

    So the body is closed and the one-liner is untouched — pre-existing, not a
    regression. Flip this test when the one-liner gets its own validation.
    """

    def test_hostile_canonical_still_reaches_frontmatter(self, vault, monkeypatch):
        frontmatter, body = _run(
            vault, monkeypatch, _glossary("<script>fetch('https://evil.example')</script>VC")
        )

        assert "<script" in frontmatter
        assert "<script" not in body
