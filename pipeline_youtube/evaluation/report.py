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

import json
from pathlib import Path

from .schemas import EvaluationIteration, EvaluationLoopResult, Finding

_EVAL_SUBDIR = "evaluation"


def resolve_eval_dir(folder_name: str, *, vault_root: Path) -> Path:
    """Return ``05_Synthesis/{folder_name}/_meta/evaluation`` under the vault.

    Reuses ``synthesis.SYNTHESIS_BASE`` / ``META_SUBDIR`` and
    ``path_safety.ensure_safe_path`` exactly like ``stages/synthesis.py`` so the
    path matches Stage 05's output folder. The directory is NOT created here —
    that is the writers' responsibility.

    ``vault_root`` is injected by the caller (``runtime.vault_root``).
    """
    from ..path_safety import ensure_safe_path
    from ..stages.synthesis import META_SUBDIR, SYNTHESIS_BASE

    safe_rel = ensure_safe_path(f"{SYNTHESIS_BASE}/{folder_name}", vault_root=vault_root)
    return vault_root / safe_rel / META_SUBDIR / _EVAL_SUBDIR


def _finding_to_dict(f: Finding) -> dict[str, object]:
    return {
        "finding_id": f.finding_id,
        "perspective": f.perspective,
        "severity": f.severity,
        "target_scope": f.target_scope,
        "target_video_id": f.target_video_id,
        "topic_ids": f.topic_ids,
        "chapter_index": f.chapter_index,
        "description": f.description,
        "suggested_fix": f.suggested_fix,
    }


def write_evaluation_loop(iteration: EvaluationIteration, eval_dir: Path) -> Path:
    """Serialize one ``EvaluationIteration`` to ``loop_{n}.json``. Returns path.

    Plain-dict projection of the frozen dataclasses via
    ``json.dumps(..., ensure_ascii=False, indent=2)``.
    """
    eval_dir.mkdir(parents=True, exist_ok=True)
    report = iteration.report
    payload: dict[str, object] = {
        "iteration": iteration.iteration,
        "perspectives": {
            "coverage": {
                "summary": report.coverage.summary,
                "findings": [_finding_to_dict(f) for f in report.coverage.findings],
            },
            "pedagogy": {
                "summary": report.pedagogy.summary,
                "findings": [_finding_to_dict(f) for f in report.pedagogy.findings],
            },
            "fidelity": {
                "summary": report.fidelity.summary,
                "findings": [_finding_to_dict(f) for f in report.fidelity.findings],
            },
        },
        "blocking_count": len(report.blocking_findings),
        "regenerated_video_ids": iteration.regenerated_video_ids,
        "synthesis_rerun": iteration.synthesis_rerun,
        "stopped": iteration.stopped,
        "stop_reason": iteration.stop_reason,
    }
    path = eval_dir / f"loop_{iteration.iteration}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_evaluation_summary(result: EvaluationLoopResult, eval_dir: Path) -> Path:
    """Render a human-readable markdown summary of all loops. Returns path."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# 評価サマリ", "", f"- ループ実行回数: {result.loops_run}"]
    if result.skipped:
        lines.append(f"- スキップ: {result.skip_reason}")
    if result.error:
        lines.append(f"- エラー: {result.error}")
    for it in result.iterations:
        report = it.report
        lines.append("")
        lines.append(f"## ループ {it.iteration}")
        lines.append(
            f"- blocking(high): {len(report.blocking_findings)} / "
            f"全 finding: {len(report.all_findings)}"
        )
        lines.append(
            f"- coverage: {len(report.coverage.findings)} / "
            f"pedagogy: {len(report.pedagogy.findings)} / "
            f"fidelity: {len(report.fidelity.findings)}"
        )
        if it.regenerated_video_ids:
            lines.append(f"- 04 再生成: {', '.join(it.regenerated_video_ids)}")
        lines.append(f"- 05 再合成: {'あり' if it.synthesis_rerun else 'なし'}")
        if it.stop_reason:
            lines.append(f"- 停止理由: {it.stop_reason}")
        for f in report.blocking_findings:
            target = f.target_video_id or f.target_scope
            lines.append(f"  - [{f.perspective}] {f.description} (→ {target})")
    path = eval_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "resolve_eval_dir",
    "write_evaluation_loop",
    "write_evaluation_summary",
]
