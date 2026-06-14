"""Sub-agent orchestration for stages 01-04.

Opt-in via ``--sub-agents N`` (default 1 keeps the original single-process
flow untouched). When ``N > 1`` the playlist is split into ``N`` contiguous
shards and each shard runs stages 01-04 as an **independent worker process**
(``python -m pipeline_youtube.main ... --video-range a:b --skip-synthesis``).
Stage 05 synthesis then runs once over the merged 01-04 output, exactly as in a
normal run — the integration phase is deliberately left unchanged.

Why separate processes (and not just ``--concurrency``): each worker is fully
isolated (its own logs, its own crash domain), so one shard failing or hanging
never takes the others down, and per-shard output is easy to inspect after the
fact. A single fixed ``run_time`` is threaded to every worker via
``--run-timestamp`` so all shards write into the *same* dated playlist folder,
which is what lets the final Stage 05 pass find every shard's output.

The pure helpers (`split_into_shards`, `strip_cli_option`, `parse_video_range`,
`build_worker_argv`, `build_synthesis_argv`) carry the testable logic;
`orchestrate_sub_agents` is the thin subprocess glue on top.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import BinaryIO

import click

DEFAULT_SUB_AGENTS = 3

# Re-invocation target. Workers are launched as ``python -m <_WORKER_MODULE>``
# so the entry point is independent of how the parent was started (console
# script vs ``python -m``).
_WORKER_MODULE = "pipeline_youtube.main"


def split_into_shards(n: int, shard_count: int = DEFAULT_SUB_AGENTS) -> list[tuple[int, int]]:
    """Split ``n`` items into up to ``shard_count`` contiguous ``[start, end)`` ranges.

    The chunk size is ``ceil(n / shard_count)``, so any remainder lands on the
    final shard (e.g. ``n=23, shard_count=3`` → ``[(0, 8), (8, 16), (16, 23)]``).
    Empty trailing ranges are dropped: a playlist smaller than ``shard_count``
    yields one shard per item rather than idle zero-video workers.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if n <= 0:
        return []
    chunk = ceil(n / shard_count)
    return [(start, min(start + chunk, n)) for start in range(0, n, chunk)]


def parse_video_range(spec: str) -> tuple[int, int]:
    """Parse a ``"start:end"`` shard spec into a ``(start, end)`` half-open range.

    Indices are 0-based and ``end`` is exclusive, matching `split_into_shards`.
    """
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--video-range must be 'start:end', got {spec!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"--video-range bounds must be integers, got {spec!r}") from exc
    if start < 0 or end < start:
        raise ValueError(f"--video-range must satisfy 0 <= start <= end, got {spec!r}")
    return start, end


def strip_cli_option(argv: list[str], option: str) -> list[str]:
    """Return ``argv`` with ``--option VALUE`` and ``--option=VALUE`` removed.

    Lets a worker inherit every other user flag verbatim (by re-using the
    parent's argv) without the orchestrator having to enumerate each option.
    """
    out: list[str] = []
    eq_prefix = f"{option}="
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == option:
            skip_next = True
            continue
        if token.startswith(eq_prefix):
            continue
        out.append(token)
    return out


def _code_bearing_flag(code_bearing: bool) -> str:
    return "--code-bearing" if code_bearing else "--no-code-bearing"


def build_worker_argv(
    base_argv: list[str], *, run_timestamp: str, start: int, end: int, code_bearing: bool
) -> list[str]:
    """Build the argv for a single shard worker process.

    ``base_argv`` is the user's original CLI args with ``--sub-agents`` already
    stripped. ``--sub-agents 1`` prevents recursive orchestration;
    ``--run-timestamp`` pins the shared playlist folder; ``--code-bearing`` /
    ``--no-code-bearing`` pins the parent's single genre decision so shards
    cannot diverge; ``--skip-synthesis`` leaves Stage 05 to the post-merge pass.
    """
    return [
        sys.executable,
        "-m",
        _WORKER_MODULE,
        *base_argv,
        "--sub-agents",
        "1",
        "--run-timestamp",
        run_timestamp,
        _code_bearing_flag(code_bearing),
        "--video-range",
        f"{start}:{end}",
        "--skip-synthesis",
    ]


def build_synthesis_argv(
    base_argv: list[str], *, run_timestamp: str, code_bearing: bool
) -> list[str]:
    """Build the argv for the post-merge Stage 05 pass (``--synthesis-only``)."""
    return [
        sys.executable,
        "-m",
        _WORKER_MODULE,
        *base_argv,
        "--sub-agents",
        "1",
        "--run-timestamp",
        run_timestamp,
        _code_bearing_flag(code_bearing),
        "--synthesis-only",
    ]


@dataclass
class _Worker:
    index: int
    start: int
    end: int
    proc: subprocess.Popen[bytes]
    log_path: Path
    log_file: BinaryIO


def _terminate_workers(workers: list[_Worker]) -> None:
    """Best-effort teardown: stop every started worker and close its log file."""
    for worker in workers:
        with contextlib.suppress(Exception):
            worker.proc.terminate()
        with contextlib.suppress(Exception):
            worker.proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            worker.log_file.close()


def orchestrate_sub_agents(
    *,
    total_videos: int,
    shard_count: int,
    run_time: datetime,
    logs_dir: Path,
    base_argv: list[str],
    run_synthesis: bool,
    code_bearing: bool,
) -> int:
    """Run stages 01-04 across parallel shard workers, then Stage 05 once.

    Returns a process exit code: 0 when at least one shard produced output and
    (when requested) Stage 05 succeeded; 1 otherwise.
    """
    shards = split_into_shards(total_videos, shard_count)
    if not shards:
        click.echo("No videos to process.")
        return 1

    ts_iso = run_time.isoformat()
    stamp = run_time.strftime("%Y%m%d-%H%M%S")
    click.echo(
        f"\n=== Sub-agents: {len(shards)} parallel worker(s) "
        f"for stages 01-04 ({total_videos} videos) ==="
    )

    # Launch all shards, rolling back already-started workers if any fails to
    # spawn — otherwise a mid-launch error would orphan live workers (still
    # writing to the vault) and leak their log file handles.
    workers: list[_Worker] = []
    try:
        for idx, (start, end) in enumerate(shards, 1):
            argv = build_worker_argv(
                base_argv, run_timestamp=ts_iso, start=start, end=end, code_bearing=code_bearing
            )
            log_path = logs_dir / f"sub_agent_{idx}_{stamp}.log"
            log_file = log_path.open("wb")
            click.echo(
                f"  [agent {idx}] videos {start + 1}-{end} ({end - start}) → {log_path.name}"
            )
            try:
                proc = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT)
            except Exception:
                log_file.close()
                raise
            workers.append(_Worker(idx, start, end, proc, log_path, log_file))
    except Exception:
        _terminate_workers(workers)
        raise

    failures: list[int] = []
    for worker in workers:
        returncode = worker.proc.wait()
        worker.log_file.close()
        if returncode == 0:
            click.echo(f"  [agent {worker.index}] ok — log: {worker.log_path}")
        else:
            click.echo(
                f"  [agent {worker.index}] FAILED (exit {returncode}) — log: {worker.log_path}"
            )
            failures.append(worker.index)

    all_failed = len(failures) == len(shards)
    if failures:
        click.echo(f"sub-agents: {len(failures)}/{len(shards)} worker(s) failed: {failures}")
    else:
        click.echo(f"sub-agents: all {len(shards)} worker(s) completed")

    if not run_synthesis:
        click.echo("[skip] --skip-synthesis: stage 05 bypassed")
        return 1 if all_failed else 0

    if all_failed:
        click.echo("[skip] all shards failed; stage 05 has no input")
        return 1

    click.echo("\n=== Stage 05 Synthesis (post sub-agents) ===")
    synth_returncode = subprocess.run(  # noqa: S603 — argv is built from our own tokens
        build_synthesis_argv(base_argv, run_timestamp=ts_iso, code_bearing=code_bearing),
        check=False,
    ).returncode
    if synth_returncode != 0:
        click.echo(f"stage 05 synthesis failed (exit {synth_returncode})")
        return 1
    return 0
