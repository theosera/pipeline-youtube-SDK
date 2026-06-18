"""実行レポート出力 (console reporting)。

動画処理サマリ・Stage 05 結果・コスト/トークン内訳を console へ出す。
集計の HOW (``_print_cost_breakdown``) は ``run_result`` が持ち、ここは表示のみ。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from .run_result import VideoRunResult, _print_cost_breakdown

if TYPE_CHECKING:
    from .stages.synthesis import SynthesisStageResult


def report_video_summary(
    total_videos: int, succeeded: list[VideoRunResult], failed: list[VideoRunResult]
) -> None:
    """Print the stages 01-04 success/failure summary."""
    click.echo("\n=== Video processing summary ===")
    click.echo(f"succeeded: {len(succeeded)}/{total_videos}")
    for f in failed:
        click.echo(f"  FAIL {f.video.video_id}: {f.error}")


def report_synthesis(result: SynthesisStageResult) -> None:
    """Print the Stage 05 outcome (skip/error or MOC + chapters + cost)."""
    if result.skipped:
        click.echo(f"[skip] {result.skip_reason}")
    elif result.error:
        click.echo(f"[error] synthesis: {result.error}")
    else:
        click.echo(f"MOC:       {result.moc_path}")
        click.echo(f"chapters:  {len(result.chapter_paths)}")
        for p in result.chapter_paths:
            click.echo(f"  - {p.name}")
        click.echo(f"meta:      {result.meta_path}")
        click.echo(
            f"tokens:    in={result.total_input_tokens}"
            f" out={result.total_output_tokens}"
            f" cache_read={result.total_cache_read_tokens}"
            f" cache_create={result.total_cache_creation_tokens}"
        )
        click.echo(f"cost:      ${result.total_cost_usd:.3f}")
        click.echo(f"duration:  {result.total_duration_ms / 1000:.1f}s")


def report_costs(results: list[VideoRunResult], synthesis_result: SynthesisStageResult) -> None:
    """Print the aggregated per-stage cost/token breakdown for the whole run."""
    _print_cost_breakdown(results, synthesis_result)
