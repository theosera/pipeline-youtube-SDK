"""Runtime backend selection for ``--provider`` / ``--hybrid`` (no LLM).

Design (chosen): **config.json is the source of truth** (方式X). With no
flag, the per-stage ``models`` map is used verbatim (the no-flag path is
left completely untouched), so heterogeneous setups keep working. The
flags are coarse per-run overrides on top:

- ``--provider P``  → route EVERY stage to ``P``'s default model for this
  run (single-model run).
- ``--hybrid``      → keep the **heavy** stages (``stage_04``, ``leader``)
  on Anthropic even when an open/local provider is otherwise selected.

``apply_selection`` returns ``(effective_models, warnings)``:

- ``effective_models`` — ``{stage: {provider, model}}`` for the registry.
  It is the single source of truth: the CLI passes it to
  ``configure_providers`` AND derives each stage's explicit ``model=``
  name from it via ``registry.resolve_role`` (so provider and model never
  disagree, and a ``--provider`` override changes both).
- ``warnings`` — advisory text (printed by the caller) when an OPEN/local
  provider is explicitly selected for the heavy stages without
  ``--hybrid``. Nothing is silently rerouted (least surprise).

Pure and deterministic → unit-testable without a provider. The flags ride
along in ``--sub-agents`` worker argv automatically.
"""

from __future__ import annotations

from typing import Any

# Stages whose output quality is most sensitive to model strength (long
# coherent generation / strict format). Mirrors the design rationale.
HEAVY_STAGES = ("stage_04", "leader")

OPEN_PROVIDERS = frozenset({"ollama", "lmstudio"})

# Per-provider default model when config has no ``default_model`` for it.
_PROVIDER_FALLBACK_MODEL = {
    "anthropic": "sonnet",
    "ollama": "qwen3:8b",
    "lmstudio": "qwen3-8b",
}


def _model_for(provider: str, providers_cfg: dict[str, Any]) -> str:
    """Default model for ``provider``: config ``default_model`` else fallback."""
    cfg = providers_cfg.get(provider)
    if isinstance(cfg, dict):
        default_model = cfg.get("default_model")
        if isinstance(default_model, str) and default_model.strip():
            return default_model
    return _PROVIDER_FALLBACK_MODEL.get(provider, _PROVIDER_FALLBACK_MODEL["ollama"])


def apply_selection(
    models_cfg: dict[str, Any],
    providers_cfg: dict[str, Any],
    stages: frozenset[str] | tuple[str, ...],
    *,
    provider: str | None = None,
    hybrid: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(effective_models, warnings)``.

    ``effective_models`` is a shallow copy of ``models_cfg`` with the flag
    overrides applied; it is passed to ``registry.configure_providers`` and
    is the single source for both provider and (via ``resolve_role``) the
    per-stage model name. ``warnings`` is advisory text.
    """
    effective: dict[str, Any] = {
        key: (dict(val) if isinstance(val, dict) else val) for key, val in models_cfg.items()
    }

    if provider is not None:
        model = _model_for(provider, providers_cfg)
        for stage in stages:
            effective[stage] = {"provider": provider, "model": model}

    if hybrid:
        anthropic_model = _model_for("anthropic", providers_cfg)
        for stage in HEAVY_STAGES:
            effective[stage] = {"provider": "anthropic", "model": anthropic_model}

    warnings: list[str] = []
    if provider in OPEN_PROVIDERS and not hybrid:
        warnings.append(
            f"⚠ オープン/ローカル backend ({provider}) で重い工程 "
            f"({', '.join(HEAVY_STAGES)}) を実行します。Stage 04 / 05(Leader) は"
            "書式崩れ・一貫性低下・repair リトライ増の可能性があります。"
            "--hybrid を付けると leader / stage_04 だけ Anthropic に引き上げます。"
        )

    return effective, warnings


__all__ = ["HEAVY_STAGES", "OPEN_PROVIDERS", "apply_selection"]
