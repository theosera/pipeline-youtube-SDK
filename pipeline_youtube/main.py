"""Console entrypoint for pipeline-youtube-sdk.

Thin entry (合成ルートの入口): the Click command lives in ``cli.py`` and the
orchestration sequence in ``command.py``; the per-stage HOW lives in the
``stages`` / ``providers`` / ``transcript`` / ``synthesis`` packages and the
extracted steps (``cli_validation`` / ``runtime`` / ``input_resolver`` /
``execution_plan`` / ``pipeline_runner`` / ``synthesis_runner`` / ``reporting``).

Re-exports ``cli`` so the ``[project.scripts]`` target ``pipeline_youtube.main:cli``
stays stable. See docs/main-architecture.md for the wiring map.
"""

from __future__ import annotations

from .cli import cli

__all__ = ["cli"]


if __name__ == "__main__":
    cli()
