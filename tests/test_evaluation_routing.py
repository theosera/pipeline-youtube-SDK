"""Deterministic routing/aggregation — scaffold stubs (logic TODO).

These document the intended deterministic behavior of
``evaluation/routing.py``. They are skipped until the bodies land.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scaffold: routing logic TODO")


def test_route_splits_04_vs_05() -> None:
    """route_findings partitions a mixed report into (for_04, for_05)."""


def test_has_blocking_findings_only_counts_high() -> None:
    """has_blocking_findings is True iff a severity=='high' finding exists."""


def test_aggregate_merges_both_perspectives() -> None:
    """aggregate_reports places coverage/pedagogy in their slots and merges."""


def test_target_video_ids_for_04_dedupes_and_sorts() -> None:
    """target_video_ids_for_04 returns a deduped, sorted id list."""
