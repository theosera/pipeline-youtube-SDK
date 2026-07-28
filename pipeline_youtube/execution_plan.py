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
    if request.handson:
        # Exclusivity vs. every other flag is enforced in cli_validation, so
        # the position here (above the phase flags) is a readability choice.
        return RunMode.HANDSON
    if request.synthesis_only:
        return RunMode.SYNTHESIS_ONLY
    if request.resume_reviewed:
        return RunMode.RESUME_REVIEWED
    if request.local_media:
        return RunMode.LOCAL_MEDIA
    return RunMode.NORMAL


def build_plan(request: CliRequest, runtime: Runtime, resolved: ResolvedInput) -> ExecutionPlan:
    """Decide the run mode, resolve run_time / shard slice, and fix the derived
    execution-control flags.

    実行判断 (どの段を走らせ、どこで止めるか) はここで request から確定し、以降は
    ``pipeline_runner`` が plan を参照する。derived bool は request の 1:1 コピー
    (``local_media`` は path の有無)。
    """
    run_time = _parse_run_timestamp(request.run_timestamp)
    video_range: tuple[int, int] | None = None
    if request.video_range is not None:
        try:
            video_range = parse_video_range(request.video_range)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
    mode = _decide_mode(request)
    if mode is RunMode.HANDSON and len(resolved.videos) != 1:
        # URL validation cannot catch this (`watch?v=X&list=Y` expands to a
        # playlist only after the metadata fetch), so the single-video
        # contract is enforced here, where request and resolved input meet.
        raise click.UsageError(
            f"--handson requires exactly one video, got {len(resolved.videos)} "
            "(playlist URL?). Pass a single-video URL without a list= parameter."
        )
    return ExecutionPlan(
        mode=mode,
        run_time=run_time,
        video_range=video_range,
        is_sub_agent_parent=mode is RunMode.SUB_AGENT_PARENT,
        is_sub_agent_worker=mode is RunMode.SUB_AGENT_WORKER,
        run_handson=mode is RunMode.HANDSON,
        run_video_stages=not request.synthesis_only,
        run_synthesis=not request.skip_synthesis,
        stop_after_capture=request.stop_after_capture,
        filter_reviewed_only=request.resume_reviewed,
        # Phase 3 must re-run Stage 04 from reviewed 02/03 notes (docs/cli.md).
        # Checkpoint would skip any same-day 04 — including leftovers from an
        # earlier full run — and feed Stage 05 those bodies without ever
        # checking `reviewed: true`.
        allow_checkpoint=not request.dry_run and not request.resume_reviewed,
        allow_proper_noun_sheet=runtime.cfg.transcript_correction and not request.dry_run,
        allow_transcript_warmup=not request.resume_reviewed and request.local_media is None,
        dry_run=request.dry_run,
    )
