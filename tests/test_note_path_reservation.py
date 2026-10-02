"""Tests for reserving note paths across concurrent video tasks (#182).

With ``--concurrency`` >= 2, two videos whose sanitized titles match share
a note stem. Each task must end up with its own set of paths, and the
paths it writes to must be the ones it reserved.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_youtube import pipeline as pipeline_mod
from pipeline_youtube import pipeline_runner as pr_mod
from pipeline_youtube import video_processing as vp_mod
from pipeline_youtube.pipeline import UNIT_DIRS, NoteReservations, reserve_note_paths
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.run_result import VideoRunResult
from pipeline_youtube.services.cache import Cache

RUN_TIME = datetime(2026, 8, 5, 9, 0)
ALL_UNITS = ("scripts", "summary", "capture", "learning")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def run() -> NoteReservations:
    """One pipeline run's registry, shared by every reservation in the test."""
    return NoteReservations()


def _video(video_id: str, title: str = "講座 第1回") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel="ch",
        upload_date=None,
        playlist_title="PL",
    )


def _suffix(path: Path, base: Path) -> str:
    return path.stem[len(base.stem) :]


class TestSequentialReservation:
    def test_same_title_gets_distinct_paths_for_every_unit(self, vault, run):
        first = reserve_note_paths(
            _video("aaaaaaaaaaa"), RUN_TIME, reservations=run, vault_root=vault
        )
        second = reserve_note_paths(
            _video("bbbbbbbbbbb"), RUN_TIME, reservations=run, vault_root=vault
        )
        for unit in ALL_UNITS:
            assert first[unit] != second[unit], unit

    def test_suffix_is_shared_across_units(self, vault, run):
        reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, reservations=run, vault_root=vault)
        first = reserve_note_paths(
            _video("aaaaaaaaaaa"), RUN_TIME, reservations=run, vault_root=vault
        )
        second = reserve_note_paths(
            _video("bbbbbbbbbbb"), RUN_TIME, reservations=run, vault_root=vault
        )
        base = first["scripts"].with_name(first["scripts"].name.replace("-2", ""))
        assert {_suffix(p, base) for p in first.values()} == {"-2"}
        assert {_suffix(p, base) for p in second.values()} == {"-3"}

    def test_placeholders_exist_for_01_to_03_only(self, vault, run):
        paths = reserve_note_paths(
            _video("aaaaaaaaaaa"), RUN_TIME, reservations=run, vault_root=vault
        )
        for unit in ("scripts", "summary", "capture"):
            assert paths[unit].is_file(), unit
        assert not paths["learning"].exists()

    def test_reservation_without_files_still_blocks_a_later_video(self, vault, run):
        # 04 is never written as an empty file, so nothing on disk marks it as
        # taken. A dry-run reservation writes nothing at all and isolates that
        # case: only the in-process reservation keeps the next video away.
        first = reserve_note_paths(
            _video("aaaaaaaaaaa"), RUN_TIME, dry_run=True, reservations=run, vault_root=vault
        )
        second = reserve_note_paths(
            _video("bbbbbbbbbbb"), RUN_TIME, reservations=run, vault_root=vault
        )
        for unit in ALL_UNITS:
            assert first[unit] != second[unit], unit

    def test_skips_a_suffix_taken_in_any_unit(self, vault, run):
        # A stray file in 04 alone must push the whole set to the next suffix,
        # so 01-04 of one video keep the same stem.
        paths = reserve_note_paths(
            _video("aaaaaaaaaaa"), RUN_TIME, reservations=run, vault_root=vault
        )
        learning_dir = paths["learning"].parent
        stray = learning_dir / paths["learning"].name.replace(".md", "-2.md")
        learning_dir.mkdir(parents=True, exist_ok=True)
        stray.write_text("x", encoding="utf-8")
        second = reserve_note_paths(
            _video("bbbbbbbbbbb"), RUN_TIME, reservations=run, vault_root=vault
        )
        base = paths["scripts"]
        assert {_suffix(p, base) for p in second.values()} == {"-3"}

    def test_dry_run_writes_nothing(self, vault, run):
        paths = reserve_note_paths(
            _video("aaaaaaaaaaa"), RUN_TIME, dry_run=True, reservations=run, vault_root=vault
        )
        for unit in ALL_UNITS:
            assert not paths[unit].exists(), unit
        assert set(paths) == set(UNIT_DIRS)


class TestConcurrentReservation:
    @pytest.mark.parametrize("slow_record", [False, True], ids=["plain", "slow-record"])
    def test_threads_never_share_a_path(self, vault, run, slow_record):
        if slow_record:
            # Widen the gap between choosing a suffix and recording it, so the
            # threads interleave there unless the choice is serialized.
            class SlowRecordSet(set[str]):
                def update(self, *others: Iterable[str]) -> None:
                    time.sleep(0.01)
                    super().update(*others)

            run.keys = SlowRecordSet()
        videos = [_video(f"v{i:010d}") for i in range(8)]
        barrier = threading.Barrier(len(videos))
        results: dict[str, dict[str, Path]] = {}
        errors: list[Exception] = []

        def worker(video: VideoMeta) -> None:
            try:
                barrier.wait()
                results[video.video_id] = reserve_note_paths(
                    video, RUN_TIME, reservations=run, vault_root=vault
                )
            except Exception as exc:  # surfaced below with the full trace
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(v,)) for v in videos]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        for unit in ALL_UNITS:
            chosen = [results[v.video_id][unit] for v in videos]
            assert len(set(chosen)) == len(videos), unit


def _case_insensitive(folder: Path) -> bool:
    probe = folder / "CaseProbe.tmp"
    probe.write_text("x", encoding="utf-8")
    try:
        return (folder / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


def _reserve_pair(vault: Path, run: NoteReservations, first: str, second: str) -> tuple[Path, Path]:
    barrier = threading.Barrier(2)
    out: dict[str, Path] = {}

    def worker(key: str, title: str) -> None:
        barrier.wait()
        out[key] = reserve_note_paths(
            _video(f"{key * 11}", title), RUN_TIME, reservations=run, vault_root=vault
        )["scripts"]

    threads = [
        threading.Thread(target=worker, args=("a", first)),
        threading.Thread(target=worker, args=("b", second)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out["a"], out["b"]


def _folded(path: Path) -> str:
    # Independent of the code under test: how APFS compares names.
    return unicodedata.normalize("NFC", str(path)).casefold()


def _same_file(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and os.path.samefile(a, b)


class TestSpellingVariants:
    """Titles one volume treats as the same file must not share a note (#184)."""

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Foo", "foo"),
            (unicodedata.normalize("NFC", "が講座"), unicodedata.normalize("NFD", "が講座")),
        ],
        ids=["case", "nfc-nfd"],
    )
    def test_concurrent_variants_get_separate_files(self, vault, run, first, second):
        a, b = _reserve_pair(vault, run, first, second)
        assert not _same_file(a, b)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Foo", "foo"),
            (unicodedata.normalize("NFC", "が講座"), unicodedata.normalize("NFD", "が講座")),
        ],
        ids=["case", "nfc-nfd"],
    )
    def test_registry_key_folds_variants(self, vault, run, first, second):
        # A dry run writes nothing, so only the registry key can keep the
        # second spelling off the first one's path.
        a = reserve_note_paths(
            _video("aaaaaaaaaaa", first), RUN_TIME, dry_run=True, reservations=run, vault_root=vault
        )
        b = reserve_note_paths(
            _video("bbbbbbbbbbb", second),
            RUN_TIME,
            dry_run=True,
            reservations=run,
            vault_root=vault,
        )
        assert _folded(a["scripts"]) != _folded(b["scripts"])

    def test_placeholders_exist_before_the_lock_is_released(self, vault, run, monkeypatch):
        # With the registry key left unfolded, only placeholders written under
        # the lock let the second spelling see the first one's file on a
        # case-insensitive volume.
        if not _case_insensitive(vault):
            pytest.skip("needs a case-insensitive volume (e.g. APFS default)")
        monkeypatch.setattr(pipeline_mod, "_reservation_key", str)
        a, b = _reserve_pair(vault, run, "Foo", "foo")
        assert not _same_file(a, b)


class TestReservationScope:
    """A registry lives for one pipeline run, not for the process.

    Several runs can share an interpreter (the SDK is a library). A registry
    that outlived its run made the next run skip paths nothing occupies.
    """

    def test_dry_run_reservation_does_not_reach_a_later_run(self, vault):
        # The dry run writes no file, so only a registry that outlived it
        # could push the real run off the free path (it picked -2).
        video = _video("aaaaaaaaaaa")
        dry = reserve_note_paths(
            video, RUN_TIME, dry_run=True, reservations=NoteReservations(), vault_root=vault
        )
        real = reserve_note_paths(
            video, RUN_TIME, reservations=NoteReservations(), vault_root=vault
        )
        assert real == dry
        assert not any(p.stem.endswith("-2") for p in real.values())

    def test_a_new_run_starts_empty_and_leaves_the_earlier_one_alone(self, vault):
        earlier = NoteReservations()
        for video_id in ("aaaaaaaaaaa", "bbbbbbbbbbb"):
            reserve_note_paths(
                _video(video_id), RUN_TIME, dry_run=True, reservations=earlier, vault_root=vault
            )
        later = NoteReservations()
        reserve_note_paths(
            _video("ccccccccccc"), RUN_TIME, dry_run=True, reservations=later, vault_root=vault
        )
        assert len(earlier.keys) == 2 * len(ALL_UNITS)
        assert len(later.keys) == len(ALL_UNITS)

    def test_concurrent_tasks_of_a_run_share_its_registry(self, vault, monkeypatch):
        seen: list[NoteReservations] = []

        def fake_process_video(video: VideoMeta, run_time: datetime, **kw) -> VideoRunResult:
            seen.append(kw["reservations"])
            return VideoRunResult(video=video)

        monkeypatch.setattr(vp_mod, "_process_video", fake_process_video)
        run = NoteReservations()
        asyncio.run(
            vp_mod._run_videos_concurrent(
                [_video("aaaaaaaaaaa"), _video("bbbbbbbbbbb")],
                RUN_TIME,
                concurrency=2,
                dry_run=True,
                capture_format="webp",
                models={},
                cache=Cache(None, enabled=False),
                vault_root=vault,
                reservations=run,
            )
        )
        assert len(seen) == 2
        assert all(r is run for r in seen)


def _invoke(
    vault: Path,
    monkeypatch: pytest.MonkeyPatch,
    videos: list[VideoMeta],
    *,
    dry_run: bool,
    concurrency: int,
    label: str = "",
) -> list[dict[str, Path]]:
    """Run ``_process_all_videos`` once and return the paths each video got.

    Stages 01-04 are replaced by a worker that keeps the real reservation and,
    on a real run, writes ``label`` into every reserved note. Checkpoint,
    transcript warm-up, proper-noun sheet and Stage 05 are off, so only the
    registry the runner creates decides the paths.
    """
    assigned: list[dict[str, Path]] = []

    def fake_process_video(
        video: VideoMeta,
        run_time: datetime,
        *,
        dry_run: bool,
        vault_root: Path,
        reservations: NoteReservations,
        **_kw: object,
    ) -> VideoRunResult:
        paths = reserve_note_paths(
            video, run_time, reservations=reservations, dry_run=dry_run, vault_root=vault_root
        )
        if not dry_run:
            for unit, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{label} {video.video_id} {unit}\n", encoding="utf-8")
        assigned.append(paths)
        return VideoRunResult(video=video, learning_md_path=paths["learning"], learning_md_body="")

    monkeypatch.setattr(pr_mod, "_process_video", fake_process_video)
    monkeypatch.setattr(vp_mod, "_process_video", fake_process_video)
    request = SimpleNamespace(
        force_video=(), concurrency=concurrency, capture_format="webp", min_playlist_size=1
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(glossary=None, transcript_correction=False, use_innertube=True),
        vault_root=vault,
        models={},
        filler_words=(),
        capture_backend=None,
        cache=Cache(None, enabled=False),
    )
    resolved = SimpleNamespace(playlist_title="PL", code_bearing=False, media_map={})
    plan = SimpleNamespace(
        allow_checkpoint=False,
        allow_transcript_warmup=False,
        filter_reviewed_only=False,
        dry_run=dry_run,
        stop_after_capture=False,
        run_synthesis=False,
    )
    pr_mod._process_all_videos(request, runtime, resolved, videos, RUN_TIME, None, None, [], plan)
    return assigned


def _all_paths(assigned: list[dict[str, Path]]) -> set[Path]:
    return {p for paths in assigned for p in paths.values()}


def _notes(vault: Path) -> list[Path]:
    return sorted(vault.rglob("*.md"))


@pytest.mark.parametrize("concurrency", [1, 2])
class TestRegistryPerInvocation:
    """``_process_all_videos`` gives every invocation a new registry.

    ``TestReservationScope`` builds its registries itself, so it cannot tell
    whether the runner makes one per call. These tests call the runner several
    times in one interpreter, as an SDK caller would. Which of two same-title
    videos gets the bare stem is not fixed under concurrency, so they compare
    sets of paths.
    """

    def test_dry_run_then_real_run_keeps_the_free_stem(self, vault, monkeypatch, concurrency):
        video = _video("aaaaaaaaaaa")
        dry = _invoke(vault, monkeypatch, [video], dry_run=True, concurrency=concurrency)
        assert _notes(vault) == []
        real = _invoke(vault, monkeypatch, [video], dry_run=False, concurrency=concurrency)
        assert real == dry
        assert not any(p.stem.endswith("-2") for p in _all_paths(real))

    def test_repeated_dry_runs_hand_out_the_same_paths(self, vault, monkeypatch, concurrency):
        # A dry run writes nothing, so only the run's own registry keeps two
        # same-title videos apart, and only a new registry per run lets the
        # next run start again from the bare stem.
        videos = [_video("aaaaaaaaaaa"), _video("bbbbbbbbbbb")]
        first = _invoke(vault, monkeypatch, videos, dry_run=True, concurrency=concurrency)
        second = _invoke(vault, monkeypatch, videos, dry_run=True, concurrency=concurrency)
        stem = first[0]["scripts"].stem.removesuffix("-2")
        assert len(_all_paths(first)) == 2 * len(ALL_UNITS)
        assert {p.stem for p in _all_paths(first)} == {stem, f"{stem}-2"}
        assert _all_paths(second) == _all_paths(first)
        assert _notes(vault) == []

    def test_real_runs_move_on_only_for_notes_on_disk(self, vault, monkeypatch, concurrency):
        video = _video("aaaaaaaaaaa")
        first = _invoke(
            vault, monkeypatch, [video], dry_run=False, concurrency=concurrency, label="first"
        )
        kept = {p: p.read_bytes() for p in _all_paths(first)}

        second = _invoke(
            vault, monkeypatch, [video], dry_run=False, concurrency=concurrency, label="second"
        )
        assert all(p.stem.endswith("-2") for p in _all_paths(second))
        assert {p: p.read_bytes() for p in kept} == kept

        # With the earlier notes gone, nothing is left to skip: a registry
        # carried over from the first two runs would push this run to -3.
        for note in _notes(vault):
            note.unlink()
        third = _invoke(vault, monkeypatch, [video], dry_run=False, concurrency=concurrency)
        assert third == first
