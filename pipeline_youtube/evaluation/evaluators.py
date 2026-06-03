"""Fixed-role evaluator sub-agents for the Evaluation phase.

SCAFFOLD — system prompts are bound from each agent's rubric skill at
import time (fixed role-division); the ``call_*`` bodies are stubs (TODO).

Two specialists, locked from the start (mirrors the α/β/Leader/Reviewer
fixed-role idiom in ``synthesis/agents.py``):

- **CoverageEvaluator** (``role="eval_coverage"``) ← ``coverage_rubric.md``
- **PedagogyEvaluator** (``role="eval_pedagogy"``) ← ``pedagogy_rubric.md``

Routing/aggregation is deterministic Python (``evaluation/routing.py``);
there is intentionally no LLM "router" role (add ``eval_router`` later if
ever needed). Each evaluator returns ``(EvaluatorReport, AgentCallResult)``,
reusing ``synthesis.agents.AgentCallResult`` / ``_wrap_result`` and the
``synthesis.agents`` JSON-block serializers (DRY).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..playlist import VideoMeta
from ..synthesis.agents import AgentCallResult
from .schemas import EvaluatorReport
from .skills import load_rubric

if TYPE_CHECKING:
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
) -> tuple[EvaluatorReport, AgentCallResult]:
    """Coverage/Duplication evaluator (one iteration).

    Inputs: the 05 output (serialized via ``synthesis.agents`` JSON blocks),
    the 04 bodies (``synthesis.agents.format_learning_materials``), and the
    deterministic ``synthesis_result.coverage.missing_topic_ids`` injected
    as a hard pre-signal. ``summary_bodies`` (02 text keyed by video_id) is
    OPTIONAL fidelity input — must be tolerated as ``None`` (e.g. under
    ``--synthesis-only``).

    Calls ``invoke_llm(role="eval_coverage", append_system_prompt=
    COVERAGE_SYSTEM_PROMPT, ...)``, parses with
    ``parse_coverage_evaluator_output``, wraps via ``_wrap_result``.

    TODO(scaffold): build prompt + invoke + parse + wrap.
    """
    raise NotImplementedError("scaffold: coverage evaluator call TODO")


def call_pedagogy_evaluator(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    synthesis_result: SynthesisStageResult,
    *,
    model: str = "sonnet",
    playlist_title: str | None = None,
    timeout: int = _DEFAULT_EVAL_TIMEOUT,
) -> tuple[EvaluatorReport, AgentCallResult]:
    """Pedagogical-quality evaluator (one iteration).

    Judges chapter ordering, difficulty progression, clarity, learner
    usefulness. Does not need 02/03. Calls
    ``invoke_llm(role="eval_pedagogy", append_system_prompt=
    PEDAGOGY_SYSTEM_PROMPT, ...)``, parses with
    ``parse_pedagogy_evaluator_output``, wraps via ``_wrap_result``.

    TODO(scaffold): build prompt + invoke + parse + wrap.
    """
    raise NotImplementedError("scaffold: pedagogy evaluator call TODO")


__all__ = [
    "COVERAGE_SYSTEM_PROMPT",
    "PEDAGOGY_SYSTEM_PROMPT",
    "call_coverage_evaluator",
    "call_pedagogy_evaluator",
]
