"""Tests for WS5: 3-phase separation via `--resume-reviewed`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import video_processing as vp_mod
from pipeline_youtube.pipeline import LEARNING_BASE, UNIT_DIRS
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.resume import (
    _filter_to_reviewed,
    _find_existing_04_md,
    _find_reviewed_summary_md,
    _find_summary_md,
    _find_unit_md,
    _unit_folder_candidates,
)
from pipeline_youtube.services.cache import Cache


def _vid(video_id: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=f"title {video_id}",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="ch",
        upload_date=None,
        playlist_title="testlist",
    )


def _write_summary(path: Path, video_id: str, reviewed: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\nreviewed: "{reviewed}"\n---\n\nbody\n',
        encoding="utf-8",
    )


def _write_capture(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\n---\n\n[00:00 ~ 00:05]\n![[capture.webp]]\n',
        encoding="utf-8",
    )


def _write_learning(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\n---\n\nlearning body\n',
        encoding="utf-8",
    )


# 11-char YouTube-shaped IDs (matches the M3 hardened extractor format).
_VID_A = "aaaaaaaaaaA"
_VID_B = "bbbbbbbbbbB"
_VID_C = "ccccccccccC"
_VID_1 = "vid1xxxxxxA"


class TestFindSummaryMd:
    def test_canonical_folder(self, tmp_path: Path):
        dt = datetime(2026, 4, 18, 8, 0)
        canonical = f"{LEARNING_BASE}/{UNIT_DIRS['summary']}/2026-04-18-0800 testlist"
        summary = tmp_path / canonical / "note.md"
        _write_summary(summary, _VID_1, "true")

        found = _find_summary_md(_VID_1, "testlist", dt, vault_root=tmp_path)
        assert found == summary

    def test_missing_returns_none(self, tmp_path: Path):
        assert (
            _find_summary_md("missingxxxx", "testlist", datetime(2026, 4, 18), vault_root=tmp_path)
            is None
        )


class TestFilterToReviewed:
    @pytest.fixture
    def vault(self, tmp_path: Path):
        dt = datetime(2026, 4, 18, 8, 0)
        folder = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"] / "2026-04-18-0800 testlist"
        _write_summary(folder / "a.md", _VID_A, "true")
        _write_summary(folder / "b.md", _VID_B, "false")
        _write_summary(folder / "c.md", _VID_C, "true")
        return dt

    def test_keeps_only_reviewed_true(self, vault, tmp_path: Path):
        to_process = [(1, _vid(_VID_A)), (2, _vid(_VID_B)), (3, _vid(_VID_C))]
        kept = _filter_to_reviewed(to_process, "testlist", vault, vault_root=tmp_path)
        assert [v.video_id for _, v in kept] == [_VID_A, _VID_C]

    def test_videos_without_summary_are_skipped(self, vault, tmp_path: Path):
        to_process = [(1, _vid("unknownXXXX"))]
        kept = _filter_to_reviewed(to_process, "testlist", vault, vault_root=tmp_path)
        assert kept == []

    def test_case_insensitive_true(self, tmp_path: Path):
        dt = datetime(2026, 4, 18, 8, 0)
        folder = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"] / "2026-04-18-0800 testlist"
        _write_summary(folder / "a.md", _VID_A, "TRUE")
        kept = _filter_to_reviewed([(1, _vid(_VID_A))], "testlist", dt, vault_root=tmp_path)
        assert len(kept) == 1

    def test_older_reviewed_summary_wins_over_newer_unreviewed(self, tmp_path: Path):
        # Same-day Phase 1 rerun leaves a newer folder with reviewed=false.
        # Phase 3 must still find the older summary the operator marked true.
        summary_base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        older = summary_base / "2026-04-18-0800 testlist"
        newer = summary_base / "2026-04-18-1000 testlist"
        _write_summary(older / "a.md", _VID_A, "true")
        _write_summary(newer / "a.md", _VID_A, "false")
        resume_time = datetime(2026, 4, 18, 12, 0)

        found = _find_reviewed_summary_md(_VID_A, "testlist", resume_time, vault_root=tmp_path)
        kept = _filter_to_reviewed(
            [(1, _vid(_VID_A))], "testlist", resume_time, vault_root=tmp_path
        )

        assert found == older / "a.md"
        assert [v.video_id for _, v in kept] == [_VID_A]
        # Newest-any lookup still sees the unreviewed rerun first — that is why
        # Phase 3 must not use it as the reviewed gate.
        assert (
            _find_summary_md(_VID_A, "testlist", resume_time, vault_root=tmp_path) == newer / "a.md"
        )


def _write_capture(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\n---\n\n[00:00 ~ 00:05]\n![[capture.webp]]\n',
        encoding="utf-8",
    )


def _write_learning(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\ndate: 2026-04-18 08:00\ntitle: "x"\nplaylist: "testlist"\n'
        f'video_id: "{video_id}"\n---\n\nlearning body\n',
        encoding="utf-8",
    )


class TestResumeReviewedProcessing:
    """Phase 3 reuses the reviewed Phase 1 notes instead of regenerating them."""

    def test_existing_04_lookup_keeps_checkpoint_skips_in_original_folder(self, tmp_path: Path):
        dt = datetime(2026, 4, 18, 12, 0)
        phase1_folder = "2026-04-18-0800 testlist"
        learning = tmp_path / LEARNING_BASE / UNIT_DIRS["learning"] / phase1_folder / "a.md"
        _write_learning(learning, _VID_A)

        assert _find_existing_04_md(_VID_A, "testlist", dt, vault_root=tmp_path) == learning

    def test_runs_only_stage_04_against_existing_reviewed_notes(self, tmp_path: Path, monkeypatch):
        resume_time = datetime(2026, 4, 18, 12, 0)
        phase1_folder = "2026-04-18-0800 testlist"
        summary = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"] / phase1_folder / "a.md"
        capture = tmp_path / LEARNING_BASE / UNIT_DIRS["capture"] / phase1_folder / "a.md"
        _write_summary(summary, _VID_A, "true")
        _write_capture(capture, _VID_A)

        def forbidden_stage(*args, **kwargs):
            raise AssertionError("stages 01-03 must be skipped during --resume-reviewed")

        monkeypatch.setattr(vp_mod, "run_stage_scripts", forbidden_stage)
        monkeypatch.setattr(vp_mod, "run_stage_summary", forbidden_stage)
        monkeypatch.setattr(vp_mod, "run_stage_capture", forbidden_stage)
        monkeypatch.setattr(vp_mod, "create_placeholder_notes", forbidden_stage)

        def fake_learning(video, summary_md_path, capture_md_path, learning_md_path, **kwargs):
            assert summary_md_path == summary
            assert capture_md_path == capture
            assert kwargs["run_time"] == resume_time
            learning_md_path.parent.mkdir(parents=True, exist_ok=True)
            learning_md_path.write_text(
                f'---\nvideo_id: "{video.video_id}"\n---\n\nlearning body\n',
                encoding="utf-8",
            )
            return LLMResponse(
                text="learning body",
                model="sonnet",
                input_tokens=1,
                output_tokens=2,
                total_cost_usd=0.01,
            )

        monkeypatch.setattr(vp_mod, "run_stage_learning", fake_learning)

        result = vp_mod._process_video(
            _vid(_VID_A),
            resume_time,
            dry_run=False,
            capture_format="auto",
            models={"stage_02": "sonnet", "stage_04": "sonnet"},
            resume_reviewed=True,
            playlist_title="testlist",
            cache=Cache(None, enabled=False),
            vault_root=tmp_path,
        )

        assert result.ok
        assert result.learning_md_body and result.learning_md_body.strip() == "learning body"
        assert result.learning_md_path == (
            tmp_path / LEARNING_BASE / UNIT_DIRS["learning"] / phase1_folder / "a.md"
        )
        # Must not create a sibling Phase-3 folder from wall-clock run_time.
        assert not (
            tmp_path / LEARNING_BASE / UNIT_DIRS["summary"] / "2026-04-18-1200 testlist"
        ).exists()

    def test_uses_older_reviewed_summary_when_newer_unreviewed_exists(
        self, tmp_path: Path, monkeypatch
    ):
        resume_time = datetime(2026, 4, 18, 12, 0)
        older_folder = "2026-04-18-0800 testlist"
        newer_folder = "2026-04-18-1000 testlist"
        summary_base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        capture_base = tmp_path / LEARNING_BASE / UNIT_DIRS["capture"]
        _write_summary(summary_base / older_folder / "a.md", _VID_A, "true")
        _write_summary(summary_base / newer_folder / "a.md", _VID_A, "false")
        _write_capture(capture_base / older_folder / "a.md", _VID_A)
        _write_capture(capture_base / newer_folder / "a.md", _VID_A)

        def fake_learning(video, summary_md_path, capture_md_path, learning_md_path, **kwargs):
            assert summary_md_path == summary_base / older_folder / "a.md"
            assert capture_md_path == capture_base / older_folder / "a.md"
            learning_md_path.parent.mkdir(parents=True, exist_ok=True)
            learning_md_path.write_text(
                f'---\nvideo_id: "{video.video_id}"\n---\n\nlearning body\n',
                encoding="utf-8",
            )
            return LLMResponse(
                text="learning body",
                model="sonnet",
                input_tokens=1,
                output_tokens=2,
                total_cost_usd=0.01,
            )

        monkeypatch.setattr(vp_mod, "run_stage_learning", fake_learning)

        result = vp_mod._process_video(
            _vid(_VID_A),
            resume_time,
            dry_run=False,
            capture_format="auto",
            models={"stage_02": "sonnet", "stage_04": "sonnet"},
            resume_reviewed=True,
            playlist_title="testlist",
            cache=Cache(None, enabled=False),
            vault_root=tmp_path,
        )

        assert result.ok
        assert result.learning_md_path == (
            tmp_path / LEARNING_BASE / UNIT_DIRS["learning"] / older_folder / "a.md"
        )

    def test_missing_playlist_title_reports_the_real_cause(self, tmp_path: Path):
        # Falling back to "" would search a folder name that cannot exist, so
        # every video would fail with the generic "summary not found".
        result = vp_mod._process_video(
            _vid(_VID_A),
            datetime(2026, 4, 18, 12, 0),
            dry_run=False,
            capture_format="auto",
            models={"stage_02": "sonnet", "stage_04": "sonnet"},
            resume_reviewed=True,
            playlist_title=None,
            cache=Cache(None, enabled=False),
            vault_root=tmp_path,
        )

        assert not result.ok
        assert result.error == "resume_reviewed_missing_playlist_title"

    def test_pinned_capture_lookup_does_not_fall_back_to_another_run(self, tmp_path: Path):
        # Two Phase 1 runs on one day. The reviewed summary lives in the 0800
        # run, whose capture note is missing. Returning the 1000 run's capture
        # would pair the reviewed summary with another run's images.
        reviewed_folder = "2026-04-18-0800 testlist"
        other_folder = "2026-04-18-1000 testlist"
        capture_base = tmp_path / LEARNING_BASE / UNIT_DIRS["capture"]
        _write_capture(capture_base / other_folder / "a.md", _VID_A)
        (capture_base / reviewed_folder).mkdir(parents=True, exist_ok=True)

        found = _find_unit_md(
            _VID_A,
            "testlist",
            datetime(2026, 4, 18, 12, 0),
            "capture",
            vault_root=tmp_path,
            preferred_folder_name=reviewed_folder,
        )

        assert found is None

    def test_same_day_folder_candidates_are_newest_first(self, tmp_path: Path):
        # iterdir() order is filesystem-dependent; the fallback must not depend
        # on it when two Phase 1 runs exist for the same playlist.
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        for name in ("2026-04-18-0800 testlist", "2026-04-18-1000 testlist"):
            (base / name).mkdir(parents=True, exist_ok=True)

        candidates = list(_unit_folder_candidates(base, "testlist", datetime(2026, 4, 18, 12, 0)))

        # First entry is the canonical HHmm folder for run_date (may not exist).
        assert candidates[0].name == "2026-04-18-1200 testlist"
        assert [c.name for c in candidates[1:]] == [
            "2026-04-18-1000 testlist",
            "2026-04-18-0800 testlist",
        ]

    def test_earlier_day_folders_come_after_same_day(self, tmp_path: Path):
        # Phase 1 → human review → Phase 3 routinely crosses midnight, so
        # earlier days must be reachable. Today still wins when both exist.
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        for name in (
            "2026-04-16-0900 testlist",
            "2026-04-17-2100 testlist",
            "2026-04-18-0800 testlist",
        ):
            (base / name).mkdir(parents=True, exist_ok=True)

        candidates = list(_unit_folder_candidates(base, "testlist", datetime(2026, 4, 18, 12, 0)))

        assert [c.name for c in candidates] == [
            "2026-04-18-1200 testlist",  # canonical (may not exist)
            "2026-04-18-0800 testlist",  # same day
            "2026-04-17-2100 testlist",  # then earlier days, newest first
            "2026-04-16-0900 testlist",
        ]

    def test_future_dated_and_undated_folders_are_ignored(self, tmp_path: Path):
        # A future folder (clock skew / hand-typed --run-timestamp) must never
        # outrank today's run, and widening past today must not start matching
        # directories that merely share a word with the title.
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        for name in (
            "2026-04-19-0900 testlist",
            "2026-04-17-2100 testlist",
            "archive testlist",
        ):
            (base / name).mkdir(parents=True, exist_ok=True)

        candidates = list(_unit_folder_candidates(base, "testlist", datetime(2026, 4, 18, 12, 0)))

        assert [c.name for c in candidates] == [
            "2026-04-18-1200 testlist",
            "2026-04-17-2100 testlist",
        ]

    def test_historical_folders_need_an_exact_title(self, tmp_path: Path):
        # Substring matching is safe within one day (the run just made those
        # folders) but across all history it would admit a *different* playlist
        # whose title merely contains this one — and if that run covered the
        # same video, Phase 3 would consume its artifacts.
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        for name in (
            "2026-04-17-0900 testlist Advanced",
            "2026-04-17-2100 testlist",
            "2026-04-18-0800 testlist Advanced",
        ):
            (base / name).mkdir(parents=True, exist_ok=True)

        candidates = list(_unit_folder_candidates(base, "testlist", datetime(2026, 4, 18, 12, 0)))

        assert [c.name for c in candidates] == [
            "2026-04-18-1200 testlist",
            # Same day keeps the substring rule, so the Advanced folder stays.
            "2026-04-18-0800 testlist Advanced",
            # Earlier days require an exact title, so only the plain one is kept.
            "2026-04-17-2100 testlist",
        ]

    def test_legacy_folder_without_hhmm_still_matches_exactly(self, tmp_path: Path):
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        (base / "2026-04-17 testlist").mkdir(parents=True, exist_ok=True)

        candidates = list(_unit_folder_candidates(base, "testlist", datetime(2026, 4, 18, 12, 0)))

        assert [c.name for c in candidates] == [
            "2026-04-18-1200 testlist",
            "2026-04-17 testlist",
        ]

    def test_reviewed_summary_from_a_previous_day_is_found(self, tmp_path: Path):
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        yesterday = base / "2026-04-17-2100 testlist"
        _write_summary(yesterday / "a.md", _VID_A, "true")
        resume_time = datetime(2026, 4, 18, 9, 0)

        found = _find_reviewed_summary_md(_VID_A, "testlist", resume_time, vault_root=tmp_path)
        kept = _filter_to_reviewed(
            [(1, _vid(_VID_A))], "testlist", resume_time, vault_root=tmp_path
        )

        assert found == yesterday / "a.md"
        assert [v.video_id for _, v in kept] == [_VID_A]

    def test_today_unreviewed_does_not_hide_yesterday_reviewed(self, tmp_path: Path):
        base = tmp_path / LEARNING_BASE / UNIT_DIRS["summary"]
        yesterday = base / "2026-04-17-2100 testlist"
        today = base / "2026-04-18-0800 testlist"
        _write_summary(yesterday / "a.md", _VID_A, "true")
        _write_summary(today / "a.md", _VID_A, "false")

        found = _find_reviewed_summary_md(
            _VID_A, "testlist", datetime(2026, 4, 18, 9, 0), vault_root=tmp_path
        )
        assert found == yesterday / "a.md"
