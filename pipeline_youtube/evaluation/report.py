"""Evaluation artifact writers (loop_N.json + human-readable summary.md).

SCAFFOLD — signatures complete; bodies are stubs (TODO).

Artifacts are colocated with the synthesis ``_meta`` dir so everything for
a playlist stays together::

    05_Synthesis/{folder}/_meta/evaluation/
        loop_0.json     # EvaluationIteration 0 (report + routing + regen ids)
        loop_1.json     # iteration 1 (if it ran)
        summary.md      # per-loop blocking findings / regen actions / stop reason

The eval dir is derived the same way synthesis derives its dir, reusing
``synthesis.SYNTHESIS_BASE`` / ``META_SUBDIR`` and the ``folder_name_override``
so eval writes into the EXACT folder Stage 05 used.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import EvaluationIteration, EvaluationLoopResult

_EVAL_SUBDIR = "evaluation"


def resolve_eval_dir(folder_name: str) -> Path:
    """Return ``05_Synthesis/{folder_name}/_meta/evaluation`` under the vault.

    Reuses ``synthesis.SYNTHESIS_BASE`` / ``META_SUBDIR`` and
    ``config.get_vault_root`` + ``sanitize.ensure_safe_path`` exactly like
    ``stages/synthesis.py`` so the path matches Stage 05's output folder.

    TODO(scaffold): build and return the path (mkdir is the writers' job).
    """
    raise NotImplementedError("scaffold: eval dir resolution TODO")


def write_evaluation_loop(iteration: EvaluationIteration, eval_dir: Path) -> Path:
    """Serialize one ``EvaluationIteration`` to ``loop_{n}.json``.

    ``json.dumps(..., ensure_ascii=False, indent=2)`` over a plain-dict
    projection of the frozen dataclasses. Returns the written path.

    TODO(scaffold): mkdir, project to dict, write.
    """
    raise NotImplementedError("scaffold: loop json writer TODO")


def write_evaluation_summary(result: EvaluationLoopResult, eval_dir: Path) -> Path:
    """Render a human-readable markdown summary of all loops. Returns path.

    TODO(scaffold): render per-loop blocking findings, regenerated video
    ids, whether 05 was re-run, and the final stop reason.
    """
    raise NotImplementedError("scaffold: summary writer TODO")


__all__ = [
    "resolve_eval_dir",
    "write_evaluation_loop",
    "write_evaluation_summary",
]
