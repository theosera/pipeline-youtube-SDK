"""LLM provider / per-stage model の設定 (runtime 配線から分離)。

config.json の ``providers`` と ``--provider`` / ``--hybrid`` override から provider
registry を初期化し、各 Stage の具体的なモデル名を解決して返す。SDK 固有のマルチ LLM
設定で、``runtime.build_runtime`` はこの関数を呼ぶだけ (HOW はここに閉じる)。
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .cli_config import _MODEL_KEYS, CliConfig
from .cli_types import CliRequest
from .providers.registry import configure_providers, resolve_role
from .providers.selection import apply_selection


def configure_provider_models(
    request: CliRequest, cfg: CliConfig, cfg_path: Path
) -> dict[str, str]:
    """Initialize LLM providers from config + CLI overrides; return per-stage models.

    Returns the per-stage concrete model-name map used as the explicit ``model=``
    arg for ``invoke_llm`` (see the resolve_role note below).
    """
    # Initialize LLM providers from config.json, applying the runtime
    # --provider / --hybrid overrides (config is the source of truth when
    # neither is given). See providers/selection.py.
    config_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    providers_raw = config_data.get("providers", {})
    if (request.provider == "anthropic" or request.hybrid) and "anthropic" not in providers_raw:
        raise click.UsageError(
            "--provider anthropic / --hybrid requires the 'anthropic' provider in config.json."
        )
    # Seed from cfg.models — the NORMALIZED map _load_config builds with the
    # per-stage fallbacks already applied (router→"haiku", other unspecified
    # stages→the --model value). Using it (not the raw config) keeps --model
    # and partial-config fallbacks honored for missing roles; resolve_role
    # handles both the object ({provider, model}) and legacy string forms.
    effective_models, model_warnings = apply_selection(
        cfg.models, providers_raw, _MODEL_KEYS, provider=request.provider, hybrid=request.hybrid
    )
    for warning in model_warnings:
        click.echo(warning)
    configure_providers(providers_raw, effective_models)
    # Resolve each stage's concrete model NAME from the SAME effective map that
    # drives provider resolution, and pass THAT as the explicit `model=` arg.
    # invoke_llm only substitutes the role-resolved model when the caller
    # passes "default", so a per-stage object config (`{provider, model}`) or a
    # --provider override must be flattened to a model-name string here — else
    # the dict / a mismatched config model name would reach the provider.
    models = {stage: resolve_role(stage)[1] for stage in _MODEL_KEYS}
    if request.provider or request.hybrid:
        click.echo(
            f"model selection: provider={request.provider or 'config'} hybrid={request.hybrid}"
        )
    click.echo(
        f"providers: {', '.join(providers_raw.keys()) if providers_raw else 'default (ollama)'}"
    )
    click.echo("llm_backends: SDK mode (no claude CLI dependency)")
    return models
