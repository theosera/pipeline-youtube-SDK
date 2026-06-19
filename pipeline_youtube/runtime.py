"""実行時依存の組み立て (composition of runtime dependencies)。

config.json を読み、provider / cache / whisper / capture backend / logger を
初期化し、その結果を不変の ``Runtime`` にまとめて返す。``main`` 起動時の
「道具を揃える係」。HOW (各 configure_*) は専用モジュールが持ち、ここは配線のみ。
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .capture_runtime import resolve_capture_backend
from .cli_config import _MODEL_KEYS, DEFAULT_CONFIG_PATH, _load_config
from .cli_types import CliRequest, Runtime
from .config import VaultRootError, set_dry_run, set_vault_root
from .providers.registry import (
    configure_llm_cache,
    configure_llm_concurrency,
    configure_providers,
    resolve_role,
)
from .providers.selection import apply_selection
from .sanitize import configure_alert_sink
from .stages.capture import sweep_stale_tmp
from .transcript.whisper_fallback import configure_whisper, describe_whisper


def build_runtime(request: CliRequest) -> Runtime:
    """Load config and initialize providers / cache / whisper / capture / logger."""
    cfg_path = request.config_path or DEFAULT_CONFIG_PATH
    cfg = _load_config(cfg_path, fallback_model=request.model)
    try:
        set_vault_root(cfg.vault_root, strict=True)
    except VaultRootError as exc:
        raise click.UsageError(str(exc)) from exc
    set_dry_run(request.dry_run)
    configure_whisper(backend=cfg.whisper_backend, model=cfg.whisper_model)
    vault_root = cfg.vault_root
    filler_words = cfg.filler_words

    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_alert_sink(logs_dir / "sanitize_alerts.jsonl")

    swept = sweep_stale_tmp(project_root / "tmp")
    if swept:
        click.echo(f"swept {swept} stale tmp video file(s)")

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

    # Persistent cache + per-role LLM cache policy. ``--no-cache`` is the
    # master off switch; otherwise deterministic artifacts (transcript/video/
    # code) and Stage 02/04/router LLM output are cached, while Stage 05
    # synthesis is opt-in via ``--cache-llm-synthesis``.
    from .cache import configure_cache
    from .stages.capture import configure_download_concurrency
    from .transcript.whisper_fallback import (
        configure_whisper_concurrency,
        configure_whisper_max_audio_seconds,
    )

    cache = configure_cache(
        request.cache_dir or cfg.cache_dir,
        enabled=not request.no_cache,
        max_video_bytes=cfg.cache_max_video_bytes,
    )
    configure_llm_cache(stages=True, synthesis=request.cache_llm_synthesis)
    if cfg.whisper_concurrency:
        configure_whisper_concurrency(cfg.whisper_concurrency)
    if cfg.whisper_max_audio_seconds is not None:
        configure_whisper_max_audio_seconds(cfg.whisper_max_audio_seconds)
    # Resource-class caps (Phase 3 A): CLI flag overrides config; None=unbounded.
    configure_llm_concurrency(request.llm_concurrency or cfg.llm_concurrency)
    configure_download_concurrency(request.download_concurrency or cfg.download_concurrency)
    click.echo(
        f"cache: {'disabled' if not cache.enabled else cache.root} "
        f"(llm synthesis cache: {'on' if request.cache_llm_synthesis else 'off'})"
    )

    # Resolve the Stage 03 capture backend (host / docker preflight / local-media
    # guard). HOW lives in capture_runtime; here we just wire it.
    active_capture_backend = resolve_capture_backend(request, cfg, vault_root, project_root)

    effective_synthesis_timeout = request.synthesis_timeout or cfg.synthesis_timeout
    effective_synthesis_profile = request.synthesis_profile or cfg.synthesis_profile or "auto"

    click.echo(f"vault_root: {vault_root}")
    click.echo(f"dry_run: {request.dry_run}")
    click.echo(f"model: {request.model}")
    click.echo(f"whisper: {describe_whisper()}")
    click.echo(f"capture_format: {request.capture_format}")
    click.echo(f"concurrency: {request.concurrency}")
    click.echo(f"min_playlist_size: {request.min_playlist_size}")
    click.echo(
        f"max_chapters: {request.max_chapters if request.max_chapters is not None else 'auto'}"
    )
    click.echo(
        f"synthesis_timeout: {effective_synthesis_timeout}s"
        if effective_synthesis_timeout
        else "synthesis_timeout: auto"
    )
    click.echo(f"synthesis_profile: {effective_synthesis_profile}")

    return Runtime(
        cfg=cfg,
        vault_root=vault_root,
        filler_words=filler_words,
        project_root=project_root,
        logs_dir=logs_dir,
        models=models,
        cache=cache,
        capture_backend=active_capture_backend,
        synthesis_timeout=effective_synthesis_timeout,
        synthesis_profile=effective_synthesis_profile,
    )
