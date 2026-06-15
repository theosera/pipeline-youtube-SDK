"""Runtime backend selection for ``--provider`` / ``--hybrid`` (no LLM).

Design (chosen): **config.json is the source of truth** (方式X). With no
flag, the per-stage ``models`` map is used verbatim, so a heterogeneous
setup (different models per stage, including different local models) keeps
working. The flags are coarse per-run overrides on top:

- ``--provider P``  → route EVERY stage to ``P``'s default model for this
  run (single-model run), ignoring config's per-stage providers.
- ``--hybrid``      → keep the **heavy** stages (``stage_04``, ``leader``)
  on Anthropic even when an open/local provider is otherwise selected
  (the opt-in escape hatch discussed in the design).

When the heavy stages end up on an open/local backend, ``apply_selection``
returns an advisory warning string (the caller prints it). Nothing is
silently rerouted — the user keeps control (least surprise).

``apply_selection`` is pure and deterministic so it is unit-testable
without a provider; the resulting map is handed to
``registry.configure_providers``. Because ``--sub-agents`` forwards the
original argv to each worker, these flags propagate to shards for free.
"""

from __future__ import annotations

from typing import Any

# Stages whose output quality is most sensitive to model strength (long
# coherent generation / strict format). Mirrors the design rationale.
HEAVY_STAGES = ("stage_04", "leader")

OPEN_PROVIDERS = frozenset({"ollama", "lmstudio"})

# Matches registry._DEFAULT_PROVIDER / _DEFAULT_MODEL (the fallback used by
# resolve_role for an unconfigured stage).
_DEFAULT_PROVIDER = "ollama"

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


def _effective_provider(entry: object) -> str:
    """Resolve a ``models[stage]`` entry to its provider name.

    Mirrors ``registry.resolve_role``: a dict uses its ``provider`` (default
    provider if absent); a legacy string is Anthropic when it is a known
    Anthropic model alias, else the default provider; anything else (missing)
    falls back to the default provider.
    """
    if isinstance(entry, dict):
        provider = entry.get("provider")
        return provider if isinstance(provider, str) and provider else _DEFAULT_PROVIDER
    if isinstance(entry, str):
        from .anthropic_sdk import _MODEL_ALIASES

        return "anthropic" if entry.lower() in _MODEL_ALIASES else _DEFAULT_PROVIDER
    return _DEFAULT_PROVIDER


def apply_selection(
    models_cfg: dict[str, Any],
    providers_cfg: dict[str, Any],
    stages: frozenset[str] | tuple[str, ...],
    *,
    provider: str | None = None,
    hybrid: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(effective_models, warnings)`` after applying the flags.

    ``effective_models`` is a shallow copy of ``models_cfg`` with the
    overrides applied; it is what should be passed to
    ``registry.configure_providers``. ``warnings`` is advisory text to print
    (empty when the heavy stages are not on an open/local backend).
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

    return effective, _heavy_open_warnings(effective)


def _heavy_open_warnings(effective: dict[str, Any]) -> list[str]:
    """Warn (once) when any heavy stage resolves to an open/local backend."""
    open_heavy = [
        s for s in HEAVY_STAGES if _effective_provider(effective.get(s)) in OPEN_PROVIDERS
    ]
    if not open_heavy:
        return []
    return [
        f"⚠ オープン/ローカル backend で重い工程 ({', '.join(open_heavy)}) を実行します。"
        "Stage 04 / 05(Leader) は書式崩れ・一貫性低下・repair リトライ増の可能性があります。"
        "--hybrid を付けると leader / stage_04 だけ Anthropic に引き上げます。"
    ]


__all__ = ["HEAVY_STAGES", "OPEN_PROVIDERS", "apply_selection"]
