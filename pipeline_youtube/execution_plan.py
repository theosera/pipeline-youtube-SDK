"""実行モードの決定 (execution plan)。

検証済みリクエストと解決済み入力から、この実行が
normal / local-media / synthesis-only / resume-reviewed / sub-agent(parent|worker)
のどれかを判定し、共有 run_time と worker のシャード範囲を確定する。
「作業計画書」。実際の処理は ``pipeline_runner`` がこの計画に沿って行う。
"""

from __future__ import annotations

import click

from .cli_types import CliRequest, ExecutionPlan, ResolvedInput, RunMode, Runtime
from .parallel import parse_video_range
from .resume import _parse_run_timestamp


def _decide_mode(request: CliRequest) -> RunMode:
    """Map request flags to a RunMode by fixed precedence (sub-agent → shard → phase flags → local-media → normal)."""
    # Order mirrors the original dispatch precedence in cli(): sub-agent
    # orchestration / shard slicing take priority over the phase flags, which
    # are mutually exclusive (enforced in cli_validation); local-media is the
    # offline variant of the normal flow.
    if request.sub_agents > 1:
        return RunMode.SUB_AGENT_PARENT
    if request.video_range is not None:
        return RunMode.SUB_AGENT_WORKER
    if request.synthesis_only:
        return RunMode.SYNTHESIS_ONLY
    if request.resume_reviewed:
        return RunMode.RESUME_REVIEWED
    if request.local_media:
        return RunMode.LOCAL_MEDIA
    return RunMode.NORMAL


def build_plan(request: CliRequest, runtime: Runtime, resolved: ResolvedInput) -> ExecutionPlan:
    """Decide the run mode, resolve the shared run_time and worker shard slice."""
    run_time = _parse_run_timestamp(request.run_timestamp)
    video_range: tuple[int, int] | None = None
    if request.video_range is not None:
        try:
            video_range = parse_video_range(request.video_range)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    return ExecutionPlan(mode=_decide_mode(request), run_time=run_time, video_range=video_range)
