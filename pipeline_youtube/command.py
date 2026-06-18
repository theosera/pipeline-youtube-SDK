"""CLI 引数を実行要求 (``CliRequest``) へ変換し、全体実行を起動する起点。

``cli.py`` (Click 定義) が組み立てた ``CliRequest`` を受け取り、
検証 → runtime 構築 → 入力解決 → 実行計画 → 実行、の順に配線するだけ。
各段階の HOW は専用モジュール (``cli_validation`` / ``runtime`` /
``input_resolver`` / ``execution_plan`` / ``pipeline_runner``) が持つ。
"""

from __future__ import annotations

from .cli_types import CliRequest
from .cli_validation import validate_request
from .execution_plan import build_plan
from .input_resolver import resolve_input
from .pipeline_runner import run_pipeline
from .runtime import build_runtime


def run(request: CliRequest) -> None:
    """Execute one pipeline invocation end-to-end (the composition sequence)."""
    validate_request(request)
    runtime = build_runtime(request)
    resolved = resolve_input(request, runtime)
    plan = build_plan(request, runtime, resolved)
    run_pipeline(request, runtime, resolved, plan)
