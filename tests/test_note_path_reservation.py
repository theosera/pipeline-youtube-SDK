"""Tests for reserving note paths across concurrent video tasks (#182).

With ``--concurrency`` >= 2, two videos whose sanitized titles match share
a note stem. Each task must end up with its own set of paths, and the
paths it writes to must be the ones it reserved.
"""

from __future__ import annotations

import os
import threading
import time
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube import pipeline as pipeline_mod
from pipeline_youtube.pipeline import UNIT_DIRS, reserve_note_paths
from pipeline_youtube.playlist import VideoMeta

RUN_TIME = datetime(2026, 8, 5, 9, 0)
ALL_UNITS = ("scripts", "summary", "capture", "learning")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return tmp_path


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
    def test_same_title_gets_distinct_paths_for_every_unit(self, vault):
        first = reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, vault_root=vault)
        second = reserve_note_paths(_video("bbbbbbbbbbb"), RUN_TIME, vault_root=vault)
        for unit in ALL_UNITS:
            assert first[unit] != second[unit], unit

    def test_suffix_is_shared_across_units(self, vault):
        reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, vault_root=vault)
        first = reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, vault_root=vault)
        second = reserve_note_paths(_video("bbbbbbbbbbb"), RUN_TIME, vault_root=vault)
        base = first["scripts"].with_name(first["scripts"].name.replace("-2", ""))
        assert {_suffix(p, base) for p in first.values()} == {"-2"}
        assert {_suffix(p, base) for p in second.values()} == {"-3"}

    def test_placeholders_exist_for_01_to_03_only(self, vault):
        paths = reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, vault_root=vault)
        for unit in ("scripts", "summary", "capture"):
            assert paths[unit].is_file(), unit
        assert not paths["learning"].exists()

    def test_reservation_without_files_still_blocks_a_later_video(self, vault):
        # 04 is never written as an empty file, so nothing on disk marks it as
        # taken. A dry-run reservation writes nothing at all and isolates that
        # case: only the in-process reservation keeps the next video away.
        first = reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, dry_run=True, vault_root=vault)
        second = reserve_note_paths(_video("bbbbbbbbbbb"), RUN_TIME, vault_root=vault)
        for unit in ALL_UNITS:
            assert first[unit] != second[unit], unit

    def test_skips_a_suffix_taken_in_any_unit(self, vault):
        # A stray file in 04 alone must push the whole set to the next suffix,
        # so 01-04 of one video keep the same stem.
        paths = reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, vault_root=vault)
        learning_dir = paths["learning"].parent
        stray = learning_dir / paths["learning"].name.replace(".md", "-2.md")
        learning_dir.mkdir(parents=True, exist_ok=True)
        stray.write_text("x", encoding="utf-8")
        second = reserve_note_paths(_video("bbbbbbbbbbb"), RUN_TIME, vault_root=vault)
        base = paths["scripts"]
        assert {_suffix(p, base) for p in second.values()} == {"-3"}

    def test_dry_run_writes_nothing(self, vault):
        paths = reserve_note_paths(_video("aaaaaaaaaaa"), RUN_TIME, dry_run=True, vault_root=vault)
        for unit in ALL_UNITS:
            assert not paths[unit].exists(), unit
        assert set(paths) == set(UNIT_DIRS)


class TestConcurrentReservation:
    @pytest.mark.parametrize("slow_record", [False, True], ids=["plain", "slow-record"])
    def test_threads_never_share_a_path(self, vault, monkeypatch, slow_record):
        if slow_record:
            # Widen the gap between choosing a suffix and recording it, so the
            # threads interleave there unless the choice is serialized.
            class SlowRecordSet(set[str]):
                def update(self, *others: Iterable[str]) -> None:
                    time.sleep(0.01)
                    super().update(*others)

            monkeypatch.setattr(pipeline_mod, "_reserved_paths", SlowRecordSet())
        videos = [_video(f"v{i:010d}") for i in range(8)]
        barrier = threading.Barrier(len(videos))
        results: dict[str, dict[str, Path]] = {}
        errors: list[Exception] = []

        def worker(video: VideoMeta) -> None:
            try:
                barrier.wait()
                results[video.video_id] = reserve_note_paths(video, RUN_TIME, vault_root=vault)
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


def _reserve_pair(vault: Path, first: str, second: str) -> tuple[Path, Path]:
    barrier = threading.Barrier(2)
    out: dict[str, Path] = {}

    def worker(key: str, title: str) -> None:
        barrier.wait()
        out[key] = reserve_note_paths(_video(f"{key * 11}", title), RUN_TIME, vault_root=vault)[
            "scripts"
        ]

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
    def test_concurrent_variants_get_separate_files(self, vault, first, second):
        a, b = _reserve_pair(vault, first, second)
        assert not _same_file(a, b)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("Foo", "foo"),
            (unicodedata.normalize("NFC", "が講座"), unicodedata.normalize("NFD", "が講座")),
        ],
        ids=["case", "nfc-nfd"],
    )
    def test_registry_key_folds_variants(self, vault, first, second):
        # A dry run writes nothing, so only the registry key can keep the
        # second spelling off the first one's path.
        a = reserve_note_paths(
            _video("aaaaaaaaaaa", first), RUN_TIME, dry_run=True, vault_root=vault
        )
        b = reserve_note_paths(
            _video("bbbbbbbbbbb", second), RUN_TIME, dry_run=True, vault_root=vault
        )
        assert _folded(a["scripts"]) != _folded(b["scripts"])

    def test_placeholders_exist_before_the_lock_is_released(self, vault, monkeypatch):
        # With the registry key left unfolded, only placeholders written under
        # the lock let the second spelling see the first one's file on a
        # case-insensitive volume.
        if not _case_insensitive(vault):
            pytest.skip("needs a case-insensitive volume (e.g. APFS default)")
        monkeypatch.setattr(pipeline_mod, "_reservation_key", str)
        a, b = _reserve_pair(vault, "Foo", "foo")
        assert not _same_file(a, b)
