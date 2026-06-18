"""CLI 引数を実行要求 (``CliRequest``) へ変換し、全体実行を起動する起点。

``cli.py`` (Click 定義) が組み立てた ``CliRequest`` を受け取り、
検証 → runtime 構築 → 入力解決 → 実行計画 → 実行、の順に配線するだけ。
各段階の HOW は専用モジュール (``cli_validation`` / ``runtime`` /
``input_resolver`` / ``execution_plan`` / ``pipeline_runner``) が持つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cli_validation import validate_request
from .execution_plan import build_plan
from .input_resolver import resolve_input
from .pipeline_runner import run_pipeline
from .runtime import build_runtime


@dataclass(frozen=True, slots=True)
class CliRequest:
    """Parsed CLI options (the "受付票"). Immutable snapshot of one invocation."""

    url: str | None
    dry_run: bool
    concurrency: int
    sub_agents: int
    video_range: str | None
    run_timestamp: str | None
    code_bearing_override: bool | None
    transcript_concurrency: int | None
    llm_concurrency: int | None
    download_concurrency: int | None
    cache_dir: Path | None
    no_cache: bool
    cache_llm_synthesis: bool
    skip_synthesis: bool
    synthesis_only: bool
    folder_name: str | None
    eval_loop: int
    force_video: tuple[str, ...]
    capture_format: str
    model: str
    min_playlist_size: int
    max_chapters: int | None
    config_path: Path | None
    stop_after_capture: bool
    resume_reviewed: bool
    capture_backend: str | None
    synthesis_timeout: int | None
    synthesis_profile: str | None
    provider: str | None
    hybrid: bool
    local_media: Path | None


def run(request: CliRequest) -> None:
    """Execute one pipeline invocation end-to-end (the composition sequence)."""
    validate_request(request)
    runtime = build_runtime(request)
    resolved = resolve_input(request, runtime)
    plan = build_plan(request, runtime, resolved)
    run_pipeline(request, runtime, resolved, plan)
