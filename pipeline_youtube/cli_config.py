"""config.json loading and the typed `CliConfig` it produces.

Extracted from `main.py`. Owns the per-run configuration model plus the
model-key / synthesis-profile / capture-backend constants that both the CLI
option definitions and the loader validate against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from .cache import DEFAULT_MAX_VIDEO_BYTES
from .glossary import (
    Glossary,
    GlossaryConflictError,
    GlossaryParseError,
    Normalizer,
    load_glossary,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_MODEL_KEYS = frozenset(
    {
        "router",
        "stage_01_correct",
        "stage_02",
        "stage_04",
        "alpha",
        "beta",
        "leader",
        "reviewer",
        "eval_coverage",
        "eval_pedagogy",
    }
)
# "gamma" accepted silently for backward-compat with existing config.json,
# but the γ LLM role has been replaced by a Python set diff — the value is ignored.
_DEPRECATED_MODEL_KEYS = frozenset({"gamma"})

_SYNTHESIS_PROFILE_CHOICES = ("auto", "standard", "parallel", "full", "parallel+full")


_CAPTURE_BACKENDS = frozenset({"host", "docker"})


@dataclass(frozen=True)
class CliConfig:
    vault_root: Path
    models: dict[str, str]
    filler_words: tuple[str, ...]
    # Stage 03 execution backend. "host" runs yt-dlp/ffmpeg directly;
    # "docker" isolates them in the hardened image built from
    # docker/Dockerfile.capture. See docs/docker.md.
    capture_backend: str = "host"
    capture_docker_image: str = "pipeline-youtube-capture:latest"
    synthesis_timeout: int | None = None
    synthesis_profile: str | None = None
    # Persistent cache (see cache.py). cache_dir=None → default ~/.cache root.
    cache_dir: Path | None = None
    cache_max_video_bytes: int = DEFAULT_MAX_VIDEO_BYTES
    # Max concurrent Whisper transcriptions (GPU/RAM bound). None → keep default.
    whisper_concurrency: int | None = None
    # Skip Whisper for audio longer than this (seconds). None → keep default
    # (DEFAULT_WHISPER_MAX_AUDIO_SECONDS). Guards against multi-hour CPU runs.
    whisper_max_audio_seconds: int | None = None
    # Fan-out for the upfront transcript cache warm-up (network-bound).
    # None → use scripts.DEFAULT_TRANSCRIPT_CONCURRENCY.
    transcript_concurrency: int | None = None
    # Resource-class caps (Phase 3 A), independent of --concurrency.
    # None → unbounded (prior behavior).
    llm_concurrency: int | None = None
    download_concurrency: int | None = None
    # Proper-noun normalization glossary (Stage 02). None → no normalization
    # (prior behavior); set via config.json "glossary_path".
    glossary: Glossary | None = None
    # Resolved path of the glossary.json above (when configured), so user
    # corrections from the per-playlist proper-noun sheet can be promoted back
    # into it. None when no glossary_path is configured.
    glossary_path: Path | None = None
    # Local-transcription backend/model (Stage 01 Whisper tier). backend:
    # "auto" (MLX on Apple Silicon, else openai), "mlx", or "openai".
    # model: None → backend default. Set via config.json
    # "whisper_backend"/"whisper_model".
    whisper_backend: str = "auto"
    whisper_model: str | None = None
    # Stage 01b: when True, run the chunked transcript through an LLM +
    # web-search correction pass (role stage_01_correct, default opus on
    # Anthropic) before rendering. Opt-in because it is a paid, slower call.
    transcript_correction: bool = False
    # Tier 0 (InnerTube iOS-client captions). True (default) → tried first in
    # the YouTube fallback chain (warm-up + per-video Stage 01); False → skipped
    # entirely. Turn off on datacenter/cloud IPs where InnerTube reliably 403s,
    # so the chain goes straight to youtube-transcript-api / Whisper.
    use_innertube: bool = True


def _load_config(config_path: Path, fallback_model: str) -> CliConfig:
    """Load config.json. Unknown keys are ignored; `models` is optional.

    Any missing model key falls back to `fallback_model` (CLI --model).
    Unrecognized model keys raise UsageError so typos are caught early.
    """
    if not config_path.exists():
        raise click.UsageError(
            f"config.json not found at {config_path}. "
            "Copy config.example.json to config.json and set vault_root."
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    vault_root = data.get("vault_root")
    if not vault_root or vault_root == "/path/to/your/Obsidian Vault":
        raise click.UsageError("config.json vault_root is not configured.")
    path = Path(vault_root).expanduser()
    if not path.exists():
        raise click.UsageError(f"vault_root does not exist: {path}")

    models_raw = data.get("models") or {}
    if not isinstance(models_raw, dict):
        raise click.UsageError("config.json: 'models' must be an object")
    unknown = set(models_raw) - _MODEL_KEYS - _DEPRECATED_MODEL_KEYS
    if unknown:
        raise click.UsageError(
            f"config.json: unknown model keys {sorted(unknown)!r}; "
            f"expected any of {sorted(_MODEL_KEYS)!r}"
        )
    # Router defaults to haiku regardless of fallback_model — it's a single
    # cheap classification call where speed/cost beats reasoning depth.
    # Stage 01b transcript correction defaults to opus (fact-checks proper
    # nouns via web search; reasoning depth matters most).
    _per_key_default = {"router": "haiku", "stage_01_correct": "opus"}
    models = {
        key: models_raw.get(key, _per_key_default.get(key, fallback_model)) for key in _MODEL_KEYS
    }

    filler_raw = data.get("filler_words")
    if filler_raw is None:
        from .transcript.chunking import DEFAULT_FILLER_WORDS

        filler = DEFAULT_FILLER_WORDS
    else:
        if not isinstance(filler_raw, list) or not all(isinstance(x, str) for x in filler_raw):
            raise click.UsageError("config.json: 'filler_words' must be a list of strings")
        filler = tuple(filler_raw)

    capture_backend = str(data.get("capture_backend") or "host").lower()
    if capture_backend not in _CAPTURE_BACKENDS:
        raise click.UsageError(
            f"config.json: capture_backend must be one of {sorted(_CAPTURE_BACKENDS)!r}, "
            f"got {capture_backend!r}"
        )
    capture_docker_image = str(
        data.get("capture_docker_image") or "pipeline-youtube-capture:latest"
    )

    synthesis_timeout_raw = data.get("synthesis_timeout")
    if synthesis_timeout_raw is None or synthesis_timeout_raw == "auto":
        synthesis_timeout: int | None = None
    elif isinstance(synthesis_timeout_raw, int) and synthesis_timeout_raw > 0:
        synthesis_timeout = synthesis_timeout_raw
    else:
        raise click.UsageError(
            f'config.json: synthesis_timeout must be a positive integer or "auto", '
            f"got {synthesis_timeout_raw!r}"
        )

    synthesis_profile_raw = data.get("synthesis_profile")
    if synthesis_profile_raw is None:
        synthesis_profile: str | None = None
    elif (
        isinstance(synthesis_profile_raw, str)
        and synthesis_profile_raw in _SYNTHESIS_PROFILE_CHOICES
    ):
        synthesis_profile = synthesis_profile_raw
    else:
        raise click.UsageError(
            f"config.json: synthesis_profile must be one of "
            f"{list(_SYNTHESIS_PROFILE_CHOICES)!r}, got {synthesis_profile_raw!r}"
        )

    cache_dir_raw = data.get("cache_dir")
    cache_dir = Path(str(cache_dir_raw)).expanduser() if cache_dir_raw else None

    max_video_raw = data.get("cache_max_video_bytes")
    if max_video_raw is None:
        cache_max_video_bytes = DEFAULT_MAX_VIDEO_BYTES
    elif isinstance(max_video_raw, int) and max_video_raw > 0:
        cache_max_video_bytes = max_video_raw
    else:
        raise click.UsageError("config.json: cache_max_video_bytes must be a positive integer")

    whisper_conc_raw = data.get("whisper_concurrency")
    if whisper_conc_raw is None:
        whisper_concurrency: int | None = None
    elif isinstance(whisper_conc_raw, int) and whisper_conc_raw > 0:
        whisper_concurrency = whisper_conc_raw
    else:
        raise click.UsageError("config.json: whisper_concurrency must be a positive integer")

    transcript_conc_raw = data.get("transcript_concurrency")
    if transcript_conc_raw is None:
        transcript_concurrency: int | None = None
    elif isinstance(transcript_conc_raw, int) and transcript_conc_raw > 0:
        transcript_concurrency = transcript_conc_raw
    else:
        raise click.UsageError("config.json: transcript_concurrency must be a positive integer")

    def _positive_int_or_none(key: str) -> int | None:
        raw = data.get(key)
        if raw is None:
            return None
        if isinstance(raw, int) and raw > 0:
            return raw
        raise click.UsageError(f"config.json: {key} must be a positive integer")

    llm_concurrency = _positive_int_or_none("llm_concurrency")
    download_concurrency = _positive_int_or_none("download_concurrency")
    whisper_max_audio_seconds = _positive_int_or_none("whisper_max_audio_seconds")

    glossary, glossary_path = _load_glossary_from_config(data, config_path)

    whisper_backend = str(data.get("whisper_backend") or "auto").lower()
    if whisper_backend not in {"auto", "mlx", "openai"}:
        raise click.UsageError(
            "config.json: whisper_backend must be one of ['auto', 'mlx', 'openai'], "
            f"got {whisper_backend!r}"
        )
    whisper_model_raw = data.get("whisper_model")
    if whisper_model_raw is None or whisper_model_raw == "":
        whisper_model = None
    elif isinstance(whisper_model_raw, str):
        whisper_model = whisper_model_raw
    else:
        raise click.UsageError("config.json: whisper_model must be a string or null")

    transcript_correction_raw = data.get("transcript_correction", False)
    if not isinstance(transcript_correction_raw, bool):
        raise click.UsageError("config.json: transcript_correction must be a boolean")

    use_innertube_raw = data.get("use_innertube", True)
    if not isinstance(use_innertube_raw, bool):
        raise click.UsageError("config.json: use_innertube must be a boolean")

    return CliConfig(
        vault_root=path,
        models=models,
        filler_words=filler,
        capture_backend=capture_backend,
        capture_docker_image=capture_docker_image,
        synthesis_timeout=synthesis_timeout,
        synthesis_profile=synthesis_profile,
        cache_dir=cache_dir,
        cache_max_video_bytes=cache_max_video_bytes,
        whisper_concurrency=whisper_concurrency,
        whisper_max_audio_seconds=whisper_max_audio_seconds,
        transcript_concurrency=transcript_concurrency,
        llm_concurrency=llm_concurrency,
        download_concurrency=download_concurrency,
        glossary=glossary,
        glossary_path=glossary_path,
        whisper_backend=whisper_backend,
        whisper_model=whisper_model,
        transcript_correction=transcript_correction_raw,
        use_innertube=use_innertube_raw,
    )


def _load_glossary_from_config(
    data: dict[str, Any], config_path: Path
) -> tuple[Glossary | None, Path | None]:
    """Load the optional proper-noun glossary referenced by ``glossary_path``.

    Returns ``(glossary, resolved_path)``. ``glossary_path`` is optional (absent
    → ``(None, None)`` → Stage 02 normalization disabled). A relative path
    resolves against config.json's directory so the glossary travels with the
    config. A malformed/missing file is a configuration error surfaced as
    ``UsageError`` (fail fast, not silently skipped). The resolved path is
    returned so user corrections can later be promoted back into the file.
    """
    raw = data.get("glossary_path")
    if raw is None:
        return None, None
    if not isinstance(raw, str) or not raw.strip():
        raise click.UsageError("config.json: glossary_path must be a non-empty string")
    glossary_path = Path(raw).expanduser()
    if not glossary_path.is_absolute():
        glossary_path = (config_path.parent / glossary_path).resolve()
    try:
        glossary = load_glossary(glossary_path)
    except (GlossaryParseError, OSError) as exc:
        raise click.UsageError(f"config.json: glossary_path could not be loaded: {exc}") from exc
    # Fail fast on variant conflicts now (build the index once at startup)
    # rather than deep inside per-video Stage 02, after transcript + LLM work.
    try:
        Normalizer(glossary)
    except GlossaryConflictError as exc:
        raise click.UsageError(f"config.json: glossary_path has a variant conflict: {exc}") from exc
    return glossary, glossary_path
