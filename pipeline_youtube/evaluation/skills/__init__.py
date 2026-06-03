"""Per-evaluator skill (rubric) loader.

Each evaluator sub-agent is a FIXED-ROLE specialist (role-division locked
from the start, like α/β/Leader/Reviewer in synthesis). Its evaluation
rubric lives here as a versioned ``*.md`` artifact and is bound to the
agent at construction time (``evaluators.py`` loads it at import and bakes
it into that agent's system prompt) — not injected dynamically or shared.

Placement rationale: ``.claude/skills/`` is Claude Code's *dev-time*
assistance dir. These rubrics are consumed by the SDK *runtime* pipeline
agents, so they live inside the package and are read via
``importlib.resources`` (works from wheels / zipimports).
"""

from __future__ import annotations

from importlib import resources

_RUBRIC_NAMES = frozenset({"coverage_rubric.md", "pedagogy_rubric.md"})


def load_rubric(name: str) -> str:
    """Return the text of the bundled rubric ``name`` (e.g. ``coverage_rubric.md``).

    Raises ``ValueError`` for an unknown rubric name so a typo fails loudly
    at construction time rather than yielding an agent with no skill.
    """
    if name not in _RUBRIC_NAMES:
        raise ValueError(f"unknown evaluation rubric: {name!r} (known: {sorted(_RUBRIC_NAMES)})")
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


__all__ = ["load_rubric"]
