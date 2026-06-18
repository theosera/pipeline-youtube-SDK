"""CLI オプションの組み合わせ検証。

未実装フラグ・矛盾フラグ・必須入力欠落を、本処理に入る前に弾く。
設定 (config.json) を読む前に成立する純粋な前提チェックだけをここに置く
(config 依存の前提チェック、例: docker backend × local-media は ``runtime`` 側)。
"""

from __future__ import annotations

import sys

import click

from .cli_types import CliRequest
from .playlist import validate_youtube_url


def validate_request(request: CliRequest) -> None:
    """Reject unimplemented / mutually-exclusive / incomplete option sets."""
    # The evaluation phase and the explicit-folder resume flow are scaffolded
    # (schemas, fixed-role agents, package, and these flags exist) but their
    # execution paths are not implemented yet. Reject them up front so the CLI
    # never crashes mid-run with NotImplementedError; drop a gate when wiring
    # its real implementation.
    if request.eval_loop > 0:
        raise click.UsageError("--eval-loop is not implemented yet (scaffolding in progress).")
    if request.folder_name:
        raise click.UsageError(
            "--folder-name resume is not implemented yet (scaffolding in progress)."
        )

    if not request.url and not request.local_media:
        click.echo("Usage: pipeline-youtube <playlist-or-video-url> [options]")
        click.echo("   or: pipeline-youtube --local-media <dir>   (fully offline)")
        sys.exit(2)

    if request.url:
        try:
            validate_youtube_url(request.url)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    # Mutually-exclusive phase flags
    phase_flags = sum(
        bool(x)
        for x in (request.stop_after_capture, request.resume_reviewed, request.synthesis_only)
    )
    if phase_flags > 1:
        raise click.UsageError(
            "--stop-after-capture, --resume-reviewed, and --synthesis-only are mutually exclusive."
        )

    # Sub-agent orchestration owns the full 01-04 → 05 flow; it is incompatible
    # with the alternate phase flags and with --dry-run (workers write the 04
    # files that the post-merge Stage 05 reads back).
    if request.sub_agents > 1 and (
        request.dry_run
        or request.synthesis_only
        or request.resume_reviewed
        or request.stop_after_capture
    ):
        raise click.UsageError(
            "--sub-agents > 1 cannot be combined with --dry-run or the "
            "--synthesis-only / --resume-reviewed / --stop-after-capture phase flags."
        )
