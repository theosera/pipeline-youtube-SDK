"""Verifies stages/handson.run_stage_handson orchestration with every
sub-stage stubbed: the 09-tree output layout, degraded-continuation
behavior, dry-run write suppression, and the no-transcript abort.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.handson.schemas import (
    HandsonMocOutput,
    HandsonPlan,
    Segment,
    SegmentLabel,
    StepBody,
    StepPlan,
)
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages import handson as handson_mod
from pipeline_youtube.stages.capture import CaptureOutcome, CaptureResult
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_NO_CACHE = Cache(None, enabled=False)
_RUN_TIME = datetime(2026, 7, 27, 12, 0)


def _video(duration: int | None = None) -> VideoMeta:
    return VideoMeta(
        video_id="vid001",
        title="Long Talk",
        url="https://www.youtube.com/watch?v=vid001",
        duration=duration,
        channel="Test",
        upload_date="20260601",
        playlist_title=None,
    )


def _transcript(total_sec: int = 9000):
    return build_result(
        video_id="vid001",
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[
            TranscriptSnippet(f"テキスト{i}", float(i * 30), 30.0) for i in range(total_sec // 30)
        ],
    )


_SEGMENTS = [
    Segment(0, 3600, SegmentLabel.LECTURE, "前半"),
    Segment(3600, 3900, SegmentLabel.QA, "失敗談"),
    Segment(3900, 9000, SegmentLabel.LECTURE, "後半"),
]

_PLAN = HandsonPlan(
    steps=(
        StepPlan(1, "前半を実践", 0, 3600, "前半を再現", ("q001",)),
        StepPlan(2, "後半を実践", 3900, 9000, "後半を再現"),
    ),
)


def _stub_pipeline(monkeypatch, *, capture: CaptureResult | None = None, transcript=None):
    """Wire every collaborator of run_stage_handson to a deterministic stub."""
    monkeypatch.setattr(
        handson_mod, "run_stage_scripts", lambda video, path, **kw: transcript or _transcript()
    )
    monkeypatch.setattr(
        handson_mod,
        "classify_segments",
        lambda *a, **kw: (list(_SEGMENTS), [], None),
    )
    monkeypatch.setattr(handson_mod, "plan_steps", lambda *a, **kw: (_PLAN, []))

    def fake_capture(video, ranges, *, assets_subfolder, **kw):
        if capture is not None:
            return capture
        outcomes = [
            CaptureOutcome(
                range=r,
                image_path=Path(f"/assets/{assets_subfolder}/pyt_{video.video_id}_h{i:02d}.webp"),
            )
            for i, r in enumerate(ranges)
        ]
        return CaptureResult(ranges=ranges, outcomes=outcomes, capture_format="webp")

    monkeypatch.setattr(handson_mod, "capture_step_clips", fake_capture)
    monkeypatch.setattr(
        handson_mod,
        "generate_step_body",
        lambda video, step, *a, **kw: (
            StepBody(step.index, step.label, "## ゴール\n\nx\n\n## 手順\n\n1. y"),
            [],
        ),
    )
    monkeypatch.setattr(
        handson_mod,
        "generate_moc_summary",
        lambda *a, **kw: (
            HandsonMocOutput(
                title="Long Talk ハンズオン",
                moc_markdown="# Long Talk ハンズオン\n- [[01_前半を実践]]\n- [[02_後半を実践]]",
                summary_markdown="## Q&A から\n- [1:00:00] 失敗談",
            ),
            [],
        ),
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / ".obsidian").mkdir()
    return tmp_path


def _run(vault: Path, **overrides):
    kwargs = {
        "run_time": _RUN_TIME,
        "folder_title": "Long Talk",
        "models": {
            "handson_segment": "sonnet",
            "handson_plan": "sonnet",
            "handson_step": "opus",
            "handson_moc": "opus",
        },
        "dry_run": False,
        "cache": _NO_CACHE,
        "vault_root": vault,
    }
    kwargs.update(overrides)
    return handson_mod.run_stage_handson(_video(duration=9000), **kwargs)


class TestRunStageHandson:
    def test_success_writes_the_09_tree(self, vault: Path, monkeypatch):
        _stub_pipeline(monkeypatch)
        result = _run(vault)
        assert result.error is None

        base = vault / "Permanent Note/09_YouTube学習_Session_only"
        folder = "2026-07-27-1200 Long Talk"
        scripts = base / "01_Scripts_Processing_Unit" / folder
        assert list(scripts.glob("*.md")), "transcript placeholder must exist under 01 unit"

        handson = base / "05_Synthesis" / folder
        assert (handson / "00_MOC.md").exists()
        assert (handson / "01_前半を実践.md").exists()
        assert (handson / "02_後半を実践.md").exists()
        assert (handson / "99_QA_Tipsまとめ.md").exists()
        meta = json.loads((handson / "_meta/handson_meta.json").read_text(encoding="utf-8"))
        assert meta["video_id"] == "vid001"
        assert meta["unassigned_insight_ids"] == []
        # 02 / 04 units must not appear (mode does not run those stages).
        assert not (base / "02_Summary_Processing_Unit").exists()
        assert not (base / "04_Learning_Material").exists()

        # The step note embeds its clip (positional mapping).
        step1 = (handson / "01_前半を実践.md").read_text(encoding="utf-8")
        assert f"![[{folder}/pyt_vid001_h00.webp]]" in step1

    def test_duration_derived_from_snippets_when_missing(self, vault: Path, monkeypatch):
        _stub_pipeline(monkeypatch)
        result = handson_mod.run_stage_handson(
            _video(duration=None),
            run_time=_RUN_TIME,
            folder_title="Long Talk",
            models={
                "handson_segment": "sonnet",
                "handson_plan": "sonnet",
                "handson_step": "opus",
                "handson_moc": "opus",
            },
            dry_run=False,
            cache=_NO_CACHE,
            vault_root=vault,
        )
        assert result.error is None
        handson = vault / "Permanent Note/09_YouTube学習_Session_only/05_Synthesis"
        meta_files = list(handson.rglob("handson_meta.json"))
        assert json.loads(meta_files[0].read_text(encoding="utf-8"))["duration_sec"] == 9000

    def test_no_transcript_aborts_with_error(self, vault: Path, monkeypatch):
        empty = build_result(
            video_id="vid001", source=TranscriptSource.ERROR, language=None, snippets=[]
        )
        _stub_pipeline(monkeypatch, transcript=empty)
        result = _run(vault)
        assert result.error == "no_transcript_snippets"
        assert result.moc_path is None

    def test_capture_total_failure_still_writes_notes(self, vault: Path, monkeypatch):
        failed = CaptureResult(ranges=[], error="download_failed: boom")
        _stub_pipeline(monkeypatch, capture=failed)
        result = _run(vault)
        assert result.error is None
        handson = vault / "Permanent Note/09_YouTube学習_Session_only/05_Synthesis"
        step_files = list(handson.rglob("01_*.md"))
        assert step_files, "steps must be written even when capture failed"
        assert "![[" not in step_files[0].read_text(encoding="utf-8")

    def test_dry_run_writes_nothing(self, vault: Path, monkeypatch):
        _stub_pipeline(monkeypatch)
        result = _run(vault, dry_run=True)
        assert result.error is None
        assert result.plan is not None and len(result.plan.steps) == 2
        assert result.moc_path is None
        assert not (vault / "Permanent Note/09_YouTube学習_Session_only").exists()

    def test_unexpected_exception_returns_error_result(self, vault: Path, monkeypatch):
        _stub_pipeline(monkeypatch)

        def explode(*a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(handson_mod, "classify_segments", explode)
        result = _run(vault)
        assert result.error is not None
        assert "RuntimeError" in result.error
