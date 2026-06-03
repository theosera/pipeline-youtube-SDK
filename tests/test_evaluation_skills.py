"""Evaluation per-agent skill (rubric) binding — fixed role-division.

These cover the IMPLEMENTED scaffold: rubrics load and are baked into each
evaluator's system prompt at import time (one skill per fixed role).
"""

from __future__ import annotations

import pytest

from pipeline_youtube.evaluation.skills import load_rubric


def test_load_rubric_returns_nonempty() -> None:
    assert load_rubric("coverage_rubric.md").strip()
    assert load_rubric("pedagogy_rubric.md").strip()


def test_load_rubric_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown evaluation rubric"):
        load_rubric("does_not_exist.md")


def test_system_prompts_bind_their_own_rubric() -> None:
    from pipeline_youtube.evaluation import evaluators

    coverage = load_rubric("coverage_rubric.md")
    pedagogy = load_rubric("pedagogy_rubric.md")

    # Each fixed role carries its OWN skill, baked in from the start.
    assert coverage in evaluators.COVERAGE_SYSTEM_PROMPT
    assert pedagogy in evaluators.PEDAGOGY_SYSTEM_PROMPT
    # Role-division: the two specialists do not share a rubric.
    assert evaluators.COVERAGE_SYSTEM_PROMPT != evaluators.PEDAGOGY_SYSTEM_PROMPT
