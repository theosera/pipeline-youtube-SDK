"""Fixed-role LLM evaluator sub-agents for the Evaluation phase.

Two specialists, locked from the start (mirrors the α/β/Leader/Reviewer
fixed-role idiom in ``synthesis/agents.py``):

- **CoverageEvaluator** (``role="eval_coverage"``) ← ``coverage_rubric.md``
- **PedagogyEvaluator** (``role="eval_pedagogy"``) ← ``pedagogy_rubric.md``

The third perspective, *fidelity*, is deterministic (``evaluation.fidelity``)
and has no LLM evaluator here. Routing/aggregation is deterministic Python
(``evaluation/routing.py``). Each evaluator returns
``(EvaluatorReport, AgentCallResult)``, reusing ``synthesis.agents``'s
``AgentCallResult`` / ``_wrap_result`` and JSON-block serializers (DRY).

Both calls are advisory: parsing never raises mid-loop (see
``schemas.parse_*``), so a malformed evaluator output degrades to an empty
report rather than aborting the evaluation pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..playlist import VideoMeta
from ..providers.registry import invoke_llm as invoke_claude
from ..sanitize import wrap_untrusted
from ..synthesis.agents import (
    AgentCallResult,
    _chapters_to_json_block,
    _coverage_to_json_block,
    _leader_output_to_json_block,
    _wrap_result,
    format_learning_materials,
)
from .schemas import (
    EvaluatorReport,
    parse_coverage_evaluator_output,
    parse_pedagogy_evaluator_output,
)
from .skills import load_rubric

if TYPE_CHECKING:
    from ..services.cache import Cache
    from ..stages.synthesis import SynthesisStageResult

# =====================================================
# Fixed role-bound system prompts (rubric baked in at construction)
# =====================================================

_SYSTEM_PREAMBLE = (
    "あなたは YouTube 学習パイプラインの成果物 (05 Synthesis + 04 学習素材) を評価する"
    "固定役割の専門評価エージェントです。以下のルーブリックの観点に厳密に従い、"
    "厳密な JSON のみを返してください (前後に散文を付けない)。\n\n"
)


def _build_system_prompt(rubric: str) -> str:
    """Bake an evaluator's rubric skill into its fixed system prompt."""
    return _SYSTEM_PREAMBLE + rubric


# Bound ONCE at import — each role carries its evaluation skill from the start.
COVERAGE_SYSTEM_PROMPT = _build_system_prompt(load_rubric("coverage_rubric.md"))
PEDAGOGY_SYSTEM_PROMPT = _build_system_prompt(load_rubric("pedagogy_rubric.md"))

_DEFAULT_EVAL_TIMEOUT = 900


def call_coverage_evaluator(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    synthesis_result: SynthesisStageResult,
    *,
    model: str = "sonnet",
    playlist_title: str | None = None,
    summary_bodies: dict[str, str] | None = None,
    timeout: int = _DEFAULT_EVAL_TIMEOUT,
    cache: Cache | None = None,
) -> tuple[EvaluatorReport, AgentCallResult]:
    """Coverage/Duplication evaluator (one advisory pass).

    Inputs: the 05 output, the 04 bodies, and the deterministic
    ``coverage.missing_topic_ids`` injected as a hard pre-signal.
    ``summary_bodies`` is accepted for interface stability (optional 02
    fidelity input) but not yet fed into the prompt; ``None`` is fine.

    Returns an empty report (with a synthetic no-op call) when synthesis
    produced no Leader output to evaluate.
    """
    if synthesis_result.leader_output is None:
        return EvaluatorReport(perspective="coverage"), _noop_call(
            COVERAGE_SYSTEM_PROMPT, model, timeout, "eval_coverage", cache=cache
        )

    missing = synthesis_result.coverage.missing_topic_ids if synthesis_result.coverage else []
    prompt = (
        "以下の 05 Synthesis 成果物と 04 学習素材を、網羅性・重複の観点で評価し、"
        "Finding[] JSON を返してください。\n\n"
        f"## 決定論シグナル: 未カバー topic_id\n\n{wrap_untrusted(str(missing))}\n\n"
        "## 05 Leader 出力\n\n"
        f"{wrap_untrusted(_leader_output_to_json_block(synthesis_result.leader_output))}\n\n"
        "## 04 学習素材\n\n"
        f"{wrap_untrusted(format_learning_materials(videos, learning_md_bodies))}"
    )
    response = invoke_claude(
        prompt=prompt,
        append_system_prompt=COVERAGE_SYSTEM_PROMPT,
        model=model,
        timeout=timeout,
        role="eval_coverage",
        cache=cache,
    )
    return parse_coverage_evaluator_output(response.text), _wrap_result(response)


def call_pedagogy_evaluator(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    synthesis_result: SynthesisStageResult,
    *,
    model: str = "sonnet",
    playlist_title: str | None = None,
    timeout: int = _DEFAULT_EVAL_TIMEOUT,
    cache: Cache | None = None,
) -> tuple[EvaluatorReport, AgentCallResult]:
    """Pedagogical-quality evaluator (one advisory pass).

    Judges chapter ordering, difficulty progression, clarity, learner
    usefulness from the rendered 05 output + the β chapter plan (does not
    need 02/03). Returns an empty report when there is no Leader output.
    """
    if synthesis_result.leader_output is None:
        return EvaluatorReport(perspective="pedagogy"), _noop_call(
            PEDAGOGY_SYSTEM_PROMPT, model, timeout, "eval_pedagogy", cache=cache
        )

    parts = [
        "以下の 05 Synthesis 成果物を、教育的品質 (章順序・難易度勾配・明快さ・"
        "学習者有用性) の観点で評価し、Finding[] JSON を返してください。",
        "## 05 Leader 出力\n\n"
        f"{wrap_untrusted(_leader_output_to_json_block(synthesis_result.leader_output))}",
        f"## β 章立て\n\n{wrap_untrusted(_chapters_to_json_block(synthesis_result.chapters))}",
    ]
    if synthesis_result.coverage is not None:
        parts.append(
            f"## カバレッジ\n\n{wrap_untrusted(_coverage_to_json_block(synthesis_result.coverage))}"
        )
    response = invoke_claude(
        prompt="\n\n".join(parts),
        append_system_prompt=PEDAGOGY_SYSTEM_PROMPT,
        model=model,
        timeout=timeout,
        role="eval_pedagogy",
        cache=cache,
    )
    return parse_pedagogy_evaluator_output(response.text), _wrap_result(response)


def _noop_call(
    system_prompt: str, model: str, timeout: int, role: str, *, cache: Cache | None = None
) -> AgentCallResult:
    """Synthetic call used when there is nothing to evaluate.

    Still goes through the provider so cost/usage logging stays uniform,
    but the empty prompt yields a trivial response that the caller ignores.
    """
    return _wrap_result(
        invoke_claude(
            prompt="(no synthesis output to evaluate)",
            append_system_prompt=system_prompt,
            model=model,
            timeout=timeout,
            role=role,
            cache=cache,
        )
    )


__all__ = [
    "COVERAGE_SYSTEM_PROMPT",
    "PEDAGOGY_SYSTEM_PROMPT",
    "call_coverage_evaluator",
    "call_pedagogy_evaluator",
]
