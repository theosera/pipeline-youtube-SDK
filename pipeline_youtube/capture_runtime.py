"""Stage 03 capture backend の解決 (runtime 配線から分離)。

CLI フラグ / config.json から capture backend を決定し、docker モードの preflight と
``--local-media`` との非互換チェックを行う。``runtime.build_runtime`` はこの関数を
呼ぶだけ (HOW はここに閉じる)。preflight は「この実行で capture が走るか」が分かって
から行う — capture を使わない経路 (``--synthesis-only``) で docker 不在のために失敗
させないため。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from .cli_config import CliConfig
from .cli_types import CliRequest
from .stages.capture import ASSETS_REL_PATH
from .stages.capture_backend import DockerBackendNotReady, DockerCaptureBackend


def resolve_capture_backend(
    request: CliRequest, cfg: CliConfig, vault_root: Path, project_root: Path
) -> Any:
    """Return the active capture backend (``None`` for the host backend).

    CLI flag beats config.json; both default to "host". For docker mode the
    preflight runs only when capture will actually run this invocation.
    """
    backend_choice = request.capture_backend or cfg.capture_backend
    # Capture runs in every mode except --synthesis-only (which only re-runs
    # Stage 05 over existing 04 md). In particular --resume-reviewed still calls
    # _process_video()/Stage 03, so it must run the docker preflight and be
    # subject to the local-media guard below.
    will_run_capture = not request.synthesis_only

    # --local-media files live outside the container's bind mounts (tmp/ + the
    # Vault assets folder), so the docker backend's ffmpeg can't read them.
    # Reject the combination up front instead of failing per-video deep inside
    # Stage 03.
    if request.local_media and backend_choice == "docker" and will_run_capture:
        raise click.UsageError(
            "--local-media is incompatible with the docker capture backend: the "
            "hardened container only mounts tmp/ and the Vault assets folder, so "
            "your media directory is not visible to ffmpeg. Re-run with the host "
            "backend (--capture-backend host)."
        )

    if backend_choice != "docker":
        click.echo("capture_backend: host")
        return None

    assets_dir = vault_root / ASSETS_REL_PATH
    assets_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    backend = DockerCaptureBackend(
        tmp_dir=tmp_dir,
        assets_dir=assets_dir,
        image=cfg.capture_docker_image,
    )
    if will_run_capture:
        try:
            backend.preflight()
        except DockerBackendNotReady as exc:
            raise click.UsageError(str(exc)) from exc
        click.echo(f"capture_backend: docker ({cfg.capture_docker_image})")
    else:
        click.echo(
            f"capture_backend: docker ({cfg.capture_docker_image}) "
            "[preflight deferred: capture not needed this run]"
        )
    return backend
