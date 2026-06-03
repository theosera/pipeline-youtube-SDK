"""CLI wiring for the evaluation phase + resume flags.

``--eval-loop`` range enforcement and ``--help`` exposure are handled by
click at parse time, so these run without a vault or network.
"""

from __future__ import annotations

from click.testing import CliRunner

from pipeline_youtube.main import cli


def test_help_lists_eval_and_folder_flags() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--eval-loop" in result.output
    assert "--folder-name" in result.output


def test_eval_loop_rejects_out_of_range() -> None:
    result = CliRunner().invoke(
        cli,
        ["--eval-loop", "3", "https://www.youtube.com/playlist?list=PLxxxx"],
    )
    assert result.exit_code != 0
    assert "eval-loop" in result.output or "Invalid value" in result.output
