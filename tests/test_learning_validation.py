"""Tests for Stage 04 validating model output before it reaches the vault.

Stage 04 was the only writer putting model output on disk unvalidated; every
other path already routed through ``validate_chapter_body``.

These pin the **wiring**, not the helpers. A helper-only test stays green if
someone deletes the call or drops the return value, and that is precisely the
regression this guards against — so the assertions look at what actually lands
on disk, or at what ``_write_md`` actually receives.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import config
from pipeline_youtube.pipeline import compute_note_paths, create_placeholder_notes
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import learning as learning_stage
from pipeline_youtube.stages.learning import _MAX_OUTPUT_CHARS, LearningOutputError
from pipeline_youtube.synthesis.body_validator import extract_allowed_embeds

_NO_CACHE = Cache(None, enabled=False)

# The one embed Stage 03 produced, so the one target the allow-list admits.
_ALLOWED_ASSET = "2026-04-14-2141 test.webp"

_SUMMARY_BODY = """## 全体サマリ
テスト用の要約。

## 要点タイムライン

### [00:00 ~ 01:03] 導入
本文。
"""

_CAPTURE_BODY = f"""[00:00 ~ 01:03]
![[{_ALLOWED_ASSET}]]
"""


@pytest.fixture
def vault(tmp_path: Path):
    config.set_dry_run(False)
    yield tmp_path


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="_h3decBW12Q",
        title="テスト動画",
        url="https://www.youtube.com/watch?v=_h3decBW12Q",
        duration=945,
        channel="AI Channel",
        upload_date="20260414",
        playlist_title="Harness Engineering",
    )


def _response(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=5,
        output_tokens=400,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        total_cost_usd=0.004,
        duration_ms=5800,
        session_id="fake-session",
        stop_reason="end_turn",
    )


def _prepare(vault: Path):
    """Seed 02/03 so Stage 04 has real inputs and a real capture allow-list."""
    video = _video()
    run_time = datetime(2026, 4, 14, 21, 41)
    paths = create_placeholder_notes(video, run_time, vault_root=vault)
    paths["summary"].write_text(
        paths["summary"].read_text() + "\n" + _SUMMARY_BODY, encoding="utf-8"
    )
    paths["capture"].write_text(
        paths["capture"].read_text() + "\n" + _CAPTURE_BODY, encoding="utf-8"
    )
    learning_path = compute_note_paths(video, run_time, units=("learning",), vault_root=vault)[
        "learning"
    ]
    return video, run_time, paths, learning_path


def _run(vault: Path, monkeypatch: pytest.MonkeyPatch, llm_body: str, **kwargs) -> Path:
    """Run Stage 04 with a stubbed model response; return the 04 md path."""
    video, run_time, paths, learning_path = _prepare(vault)
    monkeypatch.setattr(learning_stage, "invoke_claude", lambda **kw: _response(llm_body))
    learning_stage.run_stage_learning(
        video,
        summary_md_path=paths["summary"],
        capture_md_path=paths["capture"],
        learning_md_path=learning_path,
        run_time=run_time,
        cache=_NO_CACHE,
        **kwargs,
    )
    return learning_path


def _written(vault: Path, monkeypatch: pytest.MonkeyPatch, llm_body: str) -> str:
    return _run(vault, monkeypatch, llm_body).read_text(encoding="utf-8")


class TestActiveMarkupNeverReachesTheVault:
    def test_templater_token_is_stripped(self, vault, monkeypatch):
        content = _written(
            vault, monkeypatch, "## 概念: x\n<% tp.file.include('[[evil]]') %>\n- 要点"
        )
        assert "<%" not in content
        assert "%>" not in content

    def test_iframe_is_stripped(self, vault, monkeypatch):
        content = _written(
            vault, monkeypatch, "## 概念: x\n<iframe src='https://attacker.example/'></iframe>\n"
        )
        assert "<iframe" not in content

    def test_script_is_stripped(self, vault, monkeypatch):
        content = _written(vault, monkeypatch, "## 概念: x\n<script>alert(1)</script>\n")
        assert "<script" not in content

    def test_ordinary_markdown_is_untouched(self, vault, monkeypatch):
        body = "## 概念: 見出し\n[00:00 ~ 01:03]\n- 要点 1\n- 要点 2\n"
        content = _written(vault, monkeypatch, body)
        assert body.strip() in content


class TestAllowListComesFromStage03Captures:
    def test_capture_filename_survives(self, vault, monkeypatch):
        content = _written(vault, monkeypatch, f"## 概念: x\n![[{_ALLOWED_ASSET}]]\n- 要点")
        assert f"![[{_ALLOWED_ASSET}]]" in content

    def test_embed_stage03_never_produced_is_dropped(self, vault, monkeypatch):
        content = _written(vault, monkeypatch, "## 概念: x\n![[../../secret.webp]]\n- 要点")
        assert "![[../../secret.webp]]" not in content
        assert "dropped embed" in content

    def test_injected_embed_cannot_register_itself_for_stage05(self, vault, monkeypatch):
        """The self-referential hole this stage had.

        Stage 05 builds its own allow-list by running ``extract_allowed_embeds``
        over the 04 bodies it reads back from disk. While Stage 04 wrote
        unvalidated output, an injected embed registered *itself* as allowed for
        the whole playlist. Validating before the write closes that without
        Stage 05 changing at all.
        """
        content = _written(
            vault,
            monkeypatch,
            f"## 概念: x\n![[evil.webp]]\n![[{_ALLOWED_ASSET}]]\n- 要点",
        )
        allowed = extract_allowed_embeds([content])
        assert "evil.webp" not in allowed
        assert _ALLOWED_ASSET in allowed


class TestHomoglyphFoldRunsBeforeTheStrip:
    """Order matters: fold first, then strip.

    Reversed, a Cyrillic-obfuscated tag passes the strip untouched — it is not
    a tag name yet — and only *then* folds into a live one.
    """

    def test_cyrillic_script_tag_does_not_survive(self, vault, monkeypatch):
        # U+0441 CYRILLIC SMALL LETTER ES, not Latin `c`.
        content = _written(vault, monkeypatch, "## 概念: x\n<sсript>alert(1)</sсript>\n")
        assert "<sсript" not in content  # folded
        assert "<script" not in content  # and then stripped

    def test_greek_object_tag_does_not_survive(self, vault, monkeypatch):
        # U+03BF GREEK SMALL LETTER OMICRON in place of Latin `o`.
        content = _written(
            vault, monkeypatch, "## 概念: x\n<οbject data='https://attacker.example/'>\n"
        )
        assert "<οbject" not in content  # folded
        assert "<object" not in content  # and then stripped

    def test_cyrillic_embed_tag_does_not_survive(self, vault, monkeypatch):
        # U+0435 CYRILLIC SMALL LETTER IE in place of Latin `e`.
        content = _written(vault, monkeypatch, "## 概念: x\n<еmbed src='x'>\n")
        assert "<еmbed" not in content
        assert "<embed" not in content

    def test_a_lookalike_outside_the_table_is_left_alone(self, vault, monkeypatch):
        """The fold is a fixed confusables subset, not a similarity heuristic.

        Greek epsilon has no ASCII-Latin entry, so ``<iframε>`` folds to
        nothing and is never a tag. Pinned so the homoglyph tests above are not
        misread as "any look-alike is normalised".
        """
        content = _written(vault, monkeypatch, "## 概念: x\n<iframε src='x'>\n")
        assert "<iframε" in content

    def test_folding_does_not_rewrite_embed_targets(self, vault, monkeypatch):
        """The fold is deliberately link-target-blind (detect-only there).

        Rewriting targets would break otherwise-valid links, so an allow-listed
        embed must come through byte-identical.
        """
        content = _written(vault, monkeypatch, f"## 概念: x\n![[{_ALLOWED_ASSET}]]\n")
        assert f"![[{_ALLOWED_ASSET}]]" in content


class TestOutputSizeIsCappedFailClosed:
    def test_oversized_body_raises(self, vault, monkeypatch):
        with pytest.raises(LearningOutputError):
            _run(vault, monkeypatch, "## 概念: x\n" + "あ" * (_MAX_OUTPUT_CHARS + 1))

    def test_oversized_body_writes_nothing(self, vault, monkeypatch):
        """fail-closed: refusing beats writing a truncated body.

        A truncated write would look validated to every downstream reader while
        silently missing its tail — worse than the runaway it guards against.
        """
        video, run_time, paths, learning_path = _prepare(vault)
        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: _response("## 概念: x\n" + "あ" * (_MAX_OUTPUT_CHARS + 1)),
        )
        with pytest.raises(LearningOutputError):
            learning_stage.run_stage_learning(
                video,
                summary_md_path=paths["summary"],
                capture_md_path=paths["capture"],
                learning_md_path=learning_path,
                run_time=run_time,
                cache=_NO_CACHE,
            )
        assert not learning_path.exists()

    def test_body_at_the_limit_is_accepted(self, vault, monkeypatch):
        content = _written(vault, monkeypatch, "あ" * _MAX_OUTPUT_CHARS)
        assert "あ" * 100 in content

    def test_error_message_carries_no_body_text(self, vault, monkeypatch):
        # The body is model output steered by attacker-controlled captions, and
        # this message reaches logs.
        marker = "MARKER-MUST-NOT-BE-LOGGED"
        with pytest.raises(LearningOutputError) as excinfo:
            _run(vault, monkeypatch, marker + "あ" * (_MAX_OUTPUT_CHARS + 1))
        assert marker not in str(excinfo.value)


class TestValidationIsWiredIntoTheWritePath:
    def test_write_md_receives_the_validated_body(self, vault, monkeypatch):
        """Pins the assignment, not merely the call.

        Dropping ``body =`` still runs the validator and throws its result away.
        Only inspecting what ``_write_md`` actually receives catches that.
        """
        captured: dict[str, str] = {}

        def _capture(video, run_time, path, body):  # noqa: ANN001, ANN202
            captured["body"] = body

        video, run_time, paths, learning_path = _prepare(vault)
        monkeypatch.setattr(
            learning_stage,
            "invoke_claude",
            lambda **kw: _response("## 概念: x\n<iframe src='https://attacker.example/'>\n"),
        )
        monkeypatch.setattr(learning_stage, "_write_md", _capture)

        learning_stage.run_stage_learning(
            video,
            summary_md_path=paths["summary"],
            capture_md_path=paths["capture"],
            learning_md_path=learning_path,
            run_time=run_time,
            cache=_NO_CACHE,
        )

        assert "<iframe" not in captured["body"]

    def test_dry_run_writes_nothing(self, vault, monkeypatch):
        learning_path = _run(vault, monkeypatch, "## 概念: x\n<iframe src='x'>\n", dry_run=True)
        assert not learning_path.exists()
