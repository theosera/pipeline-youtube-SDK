"""Evaluation orchestrator loop — scaffold stubs (logic TODO).

These document the intended loop contract for ``stages/evaluation.py``
(stop conditions, auto-routing, subset-regen alignment). Skipped until the
loop body lands. They will mock ``call_coverage_evaluator`` /
``call_pedagogy_evaluator`` / ``run_stage_learning`` / ``run_stage_synthesis``
following the ``tests/test_synthesis_stage.py`` conventions.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scaffold: evaluation loop logic TODO")


def test_no_findings_short_circuits() -> None:
    """Zero blocking findings on iter 0 → no regen, synthesis NOT re-run."""


def test_loop_stops_at_two() -> None:
    """Always-blocking findings → exactly 2 iterations (max_loops_reached)."""


def test_auto_route_04_triggers_learning_regen() -> None:
    """A '04' finding re-runs Stage 04 for the target id, then Stage 05."""


def test_auto_route_05_only_skips_learning_regen() -> None:
    """'05'-only findings re-run Stage 05 once; Stage 04 is NOT re-run."""


def test_eval_loop_zero_is_noop() -> None:
    """max_loops=0 returns a passthrough (skipped); synthesis untouched."""


def test_subset_regen_preserves_alignment() -> None:
    """Regenerating one video keeps len(bodies)==len(videos), others unchanged."""


def test_regen_failure_keeps_last_good() -> None:
    """A failing Stage 05 re-run stops the loop and retains the prior result."""
