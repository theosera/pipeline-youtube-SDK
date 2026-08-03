"""Stage 01 must validate the body before it reaches the vault (S03 / PR-05).

These tests deliberately assert on the *wiring*, not on
``validate_chapter_body`` itself (which has its own suite). Every check below
looks at either the bytes actually written to ``01_Scripts.md`` or the string
``_append_body`` was actually handed, so deleting the call **or** dropping its
return value turns them red — a helper-level test would stay green for both.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.code_fetch import CodeSnippet, VideoExtraMetadata
from pipeline_youtube.pipeline import create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import scripts as scripts_stage
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_NO_CACHE = Cache(None, enabled=False)
_EMPTY_EXTRA = VideoExtraMetadata(description=None)


@pytest.fixture
def vault(tmp_path: Path):
    config.set_dry_run(False)
    yield tmp_path


@pytest.fixture(autouse=True)
def _stub_extra_metadata(monkeypatch):
    monkeypatch.setattr(
        scripts_stage, "fetch_video_extra_metadata", lambda video_id, *, cache: _EMPTY_EXTRA
    )


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


def _stub_transcript(monkeypatch, texts: list[str]) -> None:
    """Feed `texts` through the fetcher chain as one caption cue each.

    Cues are 5s apart against a 30s window, so several share a chunk — the
    real cadence. It also keeps every space-free run short: one cue per chunk
    would make each chunk a single huge token and drag ``_DUP_WORD_RE``
    (unbounded token length, S10's subject) into this suite's runtime. Built
    that way first and the file took 220s.
    """

    def _fetch(video_id, languages, fetchers, **kw):
        return build_result(
            video_id=video_id,
            source=TranscriptSource.OFFICIAL,
            language="ja",
            snippets=[TranscriptSnippet(text, float(i * 5), 5.0) for i, text in enumerate(texts)],
        )

    monkeypatch.setattr(scripts_stage, "fetch_with_fallback", _fetch)


def _long_cues(target_chars: int) -> list[str]:
    """Distinct ~70-char cues totalling at least `target_chars` of body.

    Distinct on purpose: identical neighbours would be collapsed by
    ``_compress``'s repeat filters and the body would never reach the size
    the test is about.
    """
    cues: list[str] = []
    total = 0
    i = 0
    while total < target_chars:
        cue = f"{i:07d}番目の解説です。ハーネス設計の考え方を順に説明していきます。"
        cues.append(cue)
        total += len(cue) + 1  # +1 for the space _join_texts inserts
        i += 1
    return cues


def _run(vault: Path, monkeypatch, texts: list[str], **kw) -> Path:
    video = _video()
    paths = create_placeholder_notes(
        video, datetime(2026, 4, 14, 21, 41), dry_run=False, vault_root=vault
    )
    _stub_transcript(monkeypatch, texts)
    scripts_stage.run_stage_scripts(
        video, paths["scripts"], window_seconds=30.0, cache=_NO_CACHE, **kw
    )
    return paths["scripts"]


def _spy_on_append(monkeypatch) -> dict[str, str]:
    """Capture the body ``_append_body`` actually receives."""
    seen: dict[str, str] = {}
    real = scripts_stage._append_body

    def _spy(path: Path, body: str) -> None:
        seen["body"] = body
        real(path, body)

    monkeypatch.setattr(scripts_stage, "_append_body", _spy)
    return seen


class TestCaptionMarkupNeverReachesDisk:
    def test_templater_html_and_embeds_are_stripped(self, vault, monkeypatch):
        written = _run(
            vault,
            monkeypatch,
            [
                '<%* await app.vault.adapter.write("x.md", "pwned") %>',
                "<script>fetch('https://evil.example')</script>",
                "<iframe src='https://evil.example'></iframe>",
                "![[../../secret]]",
            ],
        ).read_text(encoding="utf-8")

        assert "<%*" not in written
        assert "<script" not in written
        assert "<iframe" not in written
        assert "![[../../secret]]" not in written
        # The drop is recorded rather than silent, so a reader can tell.
        assert "dropped embed" in written

    def test_append_body_receives_the_validated_string(self, vault, monkeypatch):
        """Assigning the return value is what makes the call do anything.

        Calling ``_validate_script_body`` and then writing ``full_body``
        anyway leaves this assertion looking at the raw caption text.
        """
        seen = _spy_on_append(monkeypatch)
        _run(vault, monkeypatch, ["<script>alert(1)</script> 本編です"])

        assert "<script" not in seen["body"]
        assert "本編です" in seen["body"]

    def test_plain_transcript_is_untouched(self, vault, monkeypatch):
        written = _run(
            vault, monkeypatch, ["本さん、最近ハーネス", "エンジニアリングが話題"]
        ).read_text(encoding="utf-8")

        assert "本さん、最近ハーネス" in written
        assert "[00:00](https://www.youtube.com/watch?v=_h3decBW12Q&t=0)" in written


class TestFetchedCodeIsValidatedToo:
    """The ``## 関連コード`` section shares the sink, so one call covers both."""

    def _run_with_snippet(self, vault, monkeypatch, content: str) -> str:
        monkeypatch.setattr(
            scripts_stage,
            "fetch_video_extra_metadata",
            lambda video_id, *, cache: VideoExtraMetadata(
                description="https://github.com/foo/bar/blob/main/x.html"
            ),
        )
        monkeypatch.setattr(
            scripts_stage,
            "fetch_snippets_for_urls",
            lambda urls, *, cache: [
                CodeSnippet(
                    source_url="https://github.com/foo/bar/blob/main/x.html",
                    raw_url="https://raw.githubusercontent.com/foo/bar/main/x.html",
                    filename="x.html",
                    language="html",
                    content=content,
                    truncated=False,
                )
            ],
        )
        return _run(vault, monkeypatch, ["普通の字幕です"], include_code_blocks=True).read_text(
            encoding="utf-8"
        )

    def test_markup_inside_fetched_code_is_stripped(self, vault, monkeypatch):
        written = self._run_with_snippet(
            vault, monkeypatch, "<script>fetch('https://evil.example')</script>"
        )

        assert "## 関連コード" in written
        assert "<script" not in written

    def test_fenced_code_gets_no_exemption(self, vault, monkeypatch):
        """Decision 1 = (A): accept the strip, do not special-case fences.

        ``validate_chapter_body`` is fence-blind, so legitimate HTML in a
        fetched snippet is removed along with the hostile kind. Pinned here
        because it is a real cost of (A), not an accident: exempting fences
        would mean one validator carrying two policies, and the same
        fence-blindness already applies to Stage 04 (S02).
        """
        written = self._run_with_snippet(
            vault, monkeypatch, "<style>body{color:red}</style>\nconst ok = 1;"
        )

        assert "const ok = 1;" in written
        assert "<style" not in written


class TestInputLengthCeiling:
    def test_oversized_body_is_refused_and_nothing_is_written(self, vault, monkeypatch):
        cues = _long_cues(scripts_stage.MAX_SCRIPT_BODY_CHARS + 20_000)

        video = _video()
        paths = create_placeholder_notes(
            video, datetime(2026, 4, 14, 21, 41), dry_run=False, vault_root=vault
        )
        before = paths["scripts"].read_text(encoding="utf-8")
        _stub_transcript(monkeypatch, cues)

        with pytest.raises(scripts_stage.ScriptBodyTooLargeError):
            scripts_stage.run_stage_scripts(
                video, paths["scripts"], window_seconds=30.0, cache=_NO_CACHE
            )

        # fail-closed: refused outright, not truncated to fit. A truncated body
        # would look complete to every reader downstream.
        assert paths["scripts"].read_text(encoding="utf-8") == before

    def test_ceiling_is_checked_before_the_sanitize_loop(self, vault, monkeypatch):
        """Order matters: the ceiling exists to bound what the loop is handed.

        Validating first would hand the unbounded body to the very loop the
        ceiling is meant to protect, so the check is worthless in that order.
        """
        calls = {"n": 0}

        def _counting_validate(body: str, allowed) -> str:
            calls["n"] += 1
            return body

        monkeypatch.setattr(scripts_stage, "validate_chapter_body", _counting_validate)

        cues = _long_cues(scripts_stage.MAX_SCRIPT_BODY_CHARS + 20_000)
        video = _video()
        paths = create_placeholder_notes(
            video, datetime(2026, 4, 14, 21, 41), dry_run=False, vault_root=vault
        )
        _stub_transcript(monkeypatch, cues)

        with pytest.raises(scripts_stage.ScriptBodyTooLargeError):
            scripts_stage.run_stage_scripts(
                video, paths["scripts"], window_seconds=30.0, cache=_NO_CACHE
            )

        assert calls["n"] == 0

    def test_refusal_message_carries_no_body_text(self, vault, monkeypatch):
        marker = "SECRETMARKER"
        cues = [marker + c for c in _long_cues(scripts_stage.MAX_SCRIPT_BODY_CHARS + 20_000)]
        video = _video()
        paths = create_placeholder_notes(
            video, datetime(2026, 4, 14, 21, 41), dry_run=False, vault_root=vault
        )
        _stub_transcript(monkeypatch, cues)

        with pytest.raises(scripts_stage.ScriptBodyTooLargeError) as excinfo:
            scripts_stage.run_stage_scripts(
                video, paths["scripts"], window_seconds=30.0, cache=_NO_CACHE
            )

        assert marker not in str(excinfo.value)

    def test_a_realistically_long_talk_still_writes(self, vault, monkeypatch):
        """The longest real 01_Scripts.md body measured in the vault is 79,483
        chars. This builds ~300k — nearly four times that, and far past the
        50,000 that bounds Stage 02/04 — to show the ceiling is not set where
        ordinary long-form talks land.
        """
        cues = _long_cues(300_000)
        written = _run(vault, monkeypatch, cues).read_text(encoding="utf-8")

        assert len(written) > 300_000
        # Content survives intact: first and last cue both present, verbatim.
        assert cues[0] in written
        assert cues[-1] in written


class TestCurrentValidatorLimits:
    """Pins what validation does *not* cover, without claiming it is desirable.

    ``validate_chapter_body`` removes five tag names. Event-handler attributes
    and ``javascript:`` URLs are outside that pattern by design (S01 documents
    it), so they reach disk. Fixing that means rewriting the validator, which
    is a different review perspective from wiring Stage 01 into it.

    The first case is the sharp edge and the reason this class exists: removal
    splices the surrounding text, so markup the browser would *not* have
    executed can come out executable. Measured with ``html.parser``:

        <im<script>g src=x onerror=alert(1)>   -> tag='im<script', attrs=[]
        <img src=x onerror=alert(1)>           -> tag='img', onerror=alert(1)

    Before this wiring Stage 01 wrote the first form (inert); it now writes
    the second (live). Strictly narrower than what it replaces — the same cue
    previously delivered `<script>` and `<%* … %>` untouched — but it is a new
    exposure, not a pre-existing one, and it is pinned here so the follow-up
    that hardens the validator has a test to flip.
    """

    def test_tag_removal_can_splice_a_live_handler(self, vault, monkeypatch):
        written = _run(vault, monkeypatch, ["<im<script>g src=x onerror=alert(1)>"]).read_text(
            encoding="utf-8"
        )

        assert "<img src=x onerror=alert(1)>" in written

    def test_javascript_urls_are_not_stripped(self, vault, monkeypatch):
        written = _run(vault, monkeypatch, ['<a href="javascript:alert(1)">x</a>']).read_text(
            encoding="utf-8"
        )

        assert 'href="javascript:alert(1)"' in written
