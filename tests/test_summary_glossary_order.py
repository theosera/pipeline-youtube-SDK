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


def _run_counting(
    vault: Path,
    monkeypatch,
    glossary: Glossary | None,
    model_output: str = _MODEL_OUTPUT,
) -> tuple[str, str, int]:
    """Run Stage 02; return ``(frontmatter, body, llm_call_count)``.

    The note is split because the two halves have different guarantees: the
    body is what this change routes through the validator, while the
    ``one_liner`` frontmatter field is extracted upstream of validation and
    never sees it (pinned in ``TestOneLinerStillBypassesValidation``).

    The call count is what distinguishes "validation rejected this and the
    repair loop ran" from "validation silently accepted it".
    """
    calls = {"n": 0}

    def _spy(**kw):
        calls["n"] += 1
        return _fake_response(model_output)

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
    monkeypatch.setattr(summary_stage, "invoke_claude", _spy)
    summary_stage.run_stage_summary(
        video, paths["summary"], transcript, glossary=glossary, cache=_NO_CACHE
    )
    frontmatter, _, body = paths["summary"].read_text(encoding="utf-8").partition("\n---\n")
    return frontmatter, body, calls["n"]


def _run(vault: Path, monkeypatch, glossary: Glossary | None) -> tuple[str, str]:
    frontmatter, body, _ = _run_counting(vault, monkeypatch, glossary)
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

    It matters more than a single note: ``glossary.json`` is persistent, so a
    canonical that lands there reaches the frontmatter of *every* later
    summary, not just the run that introduced it. Closing the body made the
    one-liner the remaining entrance.

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


class TestSubstitutionCannotSteerTheFormat:
    """Substituting before validation must not let the glossary drive structure.

    Both cases below are regressions this change introduced and then closed —
    measured against the parent commit, where the glossary could not reach
    either code path because substitution happened after validation.
    """

    # Required markers hidden where the strip will delete them.
    _HIDDEN_MARKERS = "<%\n## 全体サマリ\n## 要点タイムライン\n### [00:00 ~ 00:30] x\n%>"
    _NO_MARKERS = f"ONE_LINER: t\n\n必須見出しの無い出力です。{_ALIAS}\n"

    def test_a_canonical_cannot_fake_the_required_sections(self, vault, monkeypatch):
        """The structural check must see what actually gets written.

        Before the fix the check ran on the pre-strip body, so these markers
        satisfied it, were then stripped, and nothing raised — the repair loop
        never ran and a summary with no required section reached disk.
        """
        _, body, calls = _run_counting(
            vault, monkeypatch, _glossary(self._HIDDEN_MARKERS), self._NO_MARKERS
        )

        assert calls == summary_stage.MAX_SUMMARY_REPAIR_RETRIES + 1  # repair ran
        assert all(h in body for h in summary_stage._REQUIRED_H2)
        assert "自動生成に失敗" in body  # the degraded placeholder, not a silent pass

    def test_model_output_cannot_fake_them_either(self, vault, monkeypatch):
        """Same defect without any glossary — it predates this change.

        Kept because the fix closes both entrances, and a later refactor that
        only re-guards the glossary would leave this one open.
        """
        _, body, calls = _run_counting(
            vault, monkeypatch, None, self._NO_MARKERS + self._HIDDEN_MARKERS
        )

        assert calls == summary_stage.MAX_SUMMARY_REPAIR_RETRIES + 1
        assert all(h in body for h in summary_stage._REQUIRED_H2)

    def test_an_alias_cannot_rename_the_one_liner_marker(self, vault, monkeypatch):
        """``ONE_LINER:`` is format, not prose — the glossary must not touch it.

        Substituting before ``_extract_one_liner`` let an alias of ``ONE_LINER``
        rename the marker, so extraction returned nothing and the renamed
        marker stayed in the body.
        """
        frontmatter, body, _ = _run_counting(
            vault,
            monkeypatch,
            Glossary(entries=(GlossaryEntry(canonical="One-liner", aliases=["ONE_LINER"]),)),
            _MODEL_OUTPUT,
        )

        assert f'one_liner: "{_ALIAS}入門"' in frontmatter
        assert "One-liner:" not in body

    def test_unsanitizable_canonical_degrades_instead_of_aborting(self, vault, monkeypatch):
        """A canonical the sanitizer cannot settle must degrade, not escape.

        ``validate_chapter_body`` raises ``BodyValidationError`` when markup
        outlives its pass cap. Substituting before validation put the glossary
        on that path, and the error escaped ``run_stage_summary`` outright —
        so one canonical with deeply nested markup aborted *every* summary
        using it, permanently, because the glossary is persistent. Measured
        against the parent, where the glossary never reached the sanitizer.
        """
        nested = "<scr<scr<scr<scr<script>ipt>ipt>ipt>ipt>"
        _, body, calls = _run_counting(vault, monkeypatch, _glossary(nested))

        assert calls == summary_stage.MAX_SUMMARY_REPAIR_RETRIES + 1
        assert "自動生成に失敗" in body
        assert "<script" not in body
