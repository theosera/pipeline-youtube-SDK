"""CLI wiring for --provider / --hybrid.

Choice validation and --help exposure are handled by click at parse time,
so these run without a vault or network.
"""

from __future__ import annotations

from click.testing import CliRunner

from pipeline_youtube.main import cli


def test_help_lists_provider_and_hybrid_flags() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--provider" in result.output
    assert "--hybrid" in result.output


def test_provider_rejects_unknown_choice() -> None:
    result = CliRunner().invoke(
        cli,
        ["--provider", "openai", "https://www.youtube.com/playlist?list=PLxxxx"],
    )
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "provider" in result.output
