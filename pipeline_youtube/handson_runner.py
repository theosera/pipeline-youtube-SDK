"""ハンズオンモード周辺制御 (handson runner)。

単一長編動画 → ハンズオン教材の実行を起動する「工程管理者」。
統合本体の HOW は ``stages/handson`` (と ``handson/`` パッケージ) が持ち、
ここは runtime から材料を取り出して起動し、結果をレポートへ渡すのみ
(``synthesis_runner`` と対になる入口層モジュール)。
"""

from __future__ import annotations

import click

from .cli_types import CliRequest, ExecutionPlan, ResolvedInput, Runtime
from .reporting import report_handson
from .stages.handson import run_stage_handson


def run_handson(
    request: CliRequest, runtime: Runtime, resolved: ResolvedInput, plan: ExecutionPlan
) -> None:
    """Run the hands-on flow for the (validated) single video and report."""
    video = resolved.videos[0]
    click.echo("\n=== Hands-on mode (single long-form video) ===")
    click.echo(f"run_time: {plan.run_time.isoformat(timespec='seconds')}")
    click.echo(f"video: {video.video_id} {video.title}")

    result = run_stage_handson(
        video,
        run_time=plan.run_time,
        folder_title=resolved.playlist_title,
        models=runtime.models,
        filler_words=runtime.filler_words,
        capture_format=request.capture_format,
        capture_backend=runtime.capture_backend,
        code_bearing=resolved.code_bearing,
        correct_model=(
            runtime.models["stage_01_correct"] if runtime.cfg.transcript_correction else None
        ),
        known_terms=None,
        use_innertube=runtime.cfg.use_innertube,
        dry_run=plan.dry_run,
        cache=runtime.cache,
        vault_root=runtime.vault_root,
    )
    report_handson(result)
