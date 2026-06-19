"""Per-video run result type and cost reporting.

Extracted from `main.py` so the CLI entry point stays a thin orchestrator.
Holds the `VideoRunResult` record threaded through stages 01-04, the shared
frontmatter-stripping helper, and the end-of-run cost breakdown table.
"""

from __future__ import annotations

from typing import Any

import click

from .domain.results import VideoRunResult as VideoRunResult


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + 4 :].lstrip()


def _print_cost_breakdown(
    video_results: list[VideoRunResult],
    synthesis_result: Any = None,
) -> None:
    """Print a per-stage / per-agent cost table summing across all videos."""
    stage_totals: dict[str, tuple[str, float]] = {}

    def _add(label: str, model: str | None, cost: float | None) -> None:
        if cost is None:
            return
        existing = stage_totals.get(label)
        prev_model = existing[0] if existing else (model or "?")
        prev_cost = existing[1] if existing else 0.0
        stage_totals[label] = (prev_model or model or "?", prev_cost + cost)

    for r in video_results:
        _add("stage_01", r.transcript_model, r.transcript_cost_usd)
        _add("stage_02", r.summary_model, r.summary_cost_usd)
        _add("stage_04", r.learning_model, r.learning_cost_usd)

    if synthesis_result is not None and getattr(synthesis_result, "agent_results", None):
        # With profile-aware orchestration, the agent_results sequence
        # varies (parallel α spawns multiple, reviewer adds one, etc.).
        # Aggregate by the prompt's system-prompt role rather than by
        # positional role labels.
        for agent_res in synthesis_result.agent_results:
            _add("synthesis", agent_res.response.model, agent_res.total_cost_usd)

    if not stage_totals:
        return

    click.echo("\n=== Cost breakdown ===")
    total = 0.0
    for label, (model, cost) in stage_totals.items():
        click.echo(f"  {label:<9} ({model:<7}) ${cost:>7.3f}")
        total += cost
    click.echo(f"  {'total':<9} {'':<9} ${total:>7.3f}")
