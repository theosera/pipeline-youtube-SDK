"""CaptureBackend abstraction for Stage 03.

Why this exists
---------------
Stage 03 shells out to `yt-dlp` (network I/O to youtube.com) and
`ffmpeg` / `gif2webp` (local media processing). Threat Model §11 R1
flags these as the **largest residual risk** because a malicious
YouTube response or a crafted video file could exercise 0day bugs in
those binaries while they're running with full host privileges.

To address R1 without breaking the "個人ローカル" usage model, we
introduce two interchangeable backends:

- ``HostCaptureBackend``: current behavior — invokes yt-dlp/ffmpeg
  directly as subprocesses on the host. Default.
- ``DockerCaptureBackend``: invokes the same binaries inside a
  hardened container built from ``docker/Dockerfile.capture`` (non-root,
  cap-drop ALL, read-only root FS, tmpfs `/tmp`, bind-mount only
  `tmp/` + the Vault `_assets/pipeline-youtube/` subfolder).

The backend is picked at CLI startup via ``config.json`` or the
``--capture-backend`` flag; the LLM stages (02 / 04 / 05) call providers
via SDK (cloud API key or a local Ollama / LM Studio endpoint) and
continue to run on the host as-is.

Both backends implement the same narrow protocol below. Callers never
branch on the backend type — they just pass args through and the
backend chooses how to execute.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Protocol


class CaptureBackendError(RuntimeError):
    """Raised when a backend cannot perform the requested operation."""


class CaptureBackend(Protocol):
    """Narrow interface every Stage 03 runner must implement.

    Methods mirror the host subprocess calls that capture.py used to
    issue directly. Both backends raise CaptureBackendError (or
    subprocess.CalledProcessError for ffmpeg/gif2webp exit codes —
    callers already handle that).
    """

    name: ClassVar[str]  # "host" or "docker", surfaced in logs

    def download_video(self, url: str, dest: Path, *, resolution: str) -> None:
        """Download `url` to `dest` at <= `resolution` height."""
        ...

    def ffmpeg(self, args: list[str], *, timeout: int) -> None:
        """Run `ffmpeg <args>` (args excludes the program name)."""
        ...

    def gif2webp(self, args: list[str], *, timeout: int) -> None:
        """Run `gif2webp <args>` (args excludes the program name)."""
        ...

    def ffmpeg_encoders(self) -> frozenset[str]:
        """Return the set of ffmpeg encoder names available via this backend."""
        ...

    def has_gif2webp(self) -> bool:
        """Whether `gif2webp` is invokable via this backend."""
        ...


# yt-dlp `outtmpl` is `{stem}.%(ext)s`. `merge_output_format=mp4` usually
# writes dest directly; some videos only mux as mkv/webm. An interrupted
# prior download of the same id leaves DASH fragments named `{stem}.f137.mp4`
# beside them. A naive `{stem}.*` glob sorts those fragments first (`.f` <
# `.m`/`.w`) and would rename the leftover onto dest, so ffmpeg then captures
# from a partial file. When the format chain falls through to the last
# `best[height<=R]`, nothing is merged, `merge_output_format` does not apply,
# and the file keeps that format's own container (`.3gp` for YouTube itag 17;
# `.flv` for single-file FLV formats).
_YTDLP_MEDIA_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".m4v", ".3gp", ".flv"})


def _ytdlp_containers(dest: Path) -> list[Path]:
    """Complete same-stem containers beside ``dest`` (``dest`` itself included).

    Only a file whose *entire* stem equals ``dest.stem`` counts — that
    excludes DASH fragments (``{stem}.f137.mp4``) and ``.part`` temps
    (``{stem}.mp4.part``).
    """
    stem = dest.stem
    return [
        p
        for p in dest.parent.glob(f"{stem}.*")
        if p.is_file() and p.stem == stem and p.suffix.lower() in _YTDLP_MEDIA_SUFFIXES
    ]


def _clear_ytdlp_outputs(dest: Path) -> None:
    """Remove complete containers an earlier download left beside ``dest``.

    Run before yt-dlp so the only container afterwards is this run's: mtime
    alone cannot tell them apart when yt-dlp reuses an old complete file
    instead of writing a new one. Fragments and ``.part`` temps are kept so
    an interrupted download can still resume.
    """
    for p in _ytdlp_containers(dest):
        p.unlink()


def _adopt_ytdlp_output(dest: Path) -> None:
    """Point ``dest`` at yt-dlp's file when it used a different extension.

    Candidates are ``_ytdlp_containers``. Callers clear leftovers first with
    ``_clear_ytdlp_outputs``; if several still remain, the newest mtime wins.
    """
    if dest.exists():
        return
    candidates = _ytdlp_containers(dest)
    if not candidates:
        raise FileNotFoundError(f"yt-dlp produced no file for {dest}")
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    chosen.rename(dest)


# =====================================================
# Host backend (default — current behavior)
# =====================================================


@dataclass(frozen=True)
class HostCaptureBackend:
    """Default backend: direct host subprocess + yt-dlp Python API."""

    name: ClassVar[str] = "host"

    def download_video(self, url: str, dest: Path, *, resolution: str) -> None:
        # Imported lazily so `uv run --no-extras` setups that don't have
        # yt-dlp in the main deps still allow host mode to error at
        # call time instead of at module import time.
        import yt_dlp  # type: ignore[import-untyped]

        _clear_ytdlp_outputs(dest)

        fmt = (
            f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={resolution}][ext=mp4]/"
            f"best[height<={resolution}]"
        )
        ydl_opts: dict[str, Any] = {
            "format": fmt,
            "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may have written with a different extension (e.g. .mkv).
        _adopt_ytdlp_output(dest)

        with contextlib.suppress(OSError):
            os.chmod(dest, 0o600)

    def ffmpeg(self, args: list[str], *, timeout: int) -> None:
        subprocess.run(["ffmpeg", *args], capture_output=True, check=True, timeout=timeout)

    def gif2webp(self, args: list[str], *, timeout: int) -> None:
        subprocess.run(["gif2webp", *args], capture_output=True, check=True, timeout=timeout)

    def ffmpeg_encoders(self) -> frozenset[str]:
        return _host_ffmpeg_encoders()

    def has_gif2webp(self) -> bool:
        return shutil.which("gif2webp") is not None


@lru_cache(maxsize=1)
def _host_ffmpeg_encoders() -> frozenset[str]:
    """Cached `ffmpeg -encoders` parse. Scoped to host backend only."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return frozenset()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] and parts[0][0] in {"V", "A", "S"}:
            encoders.add(parts[1])
    return frozenset(encoders)


# =====================================================
# Docker backend (R1 mitigation — Threat Model §11)
# =====================================================


DEFAULT_DOCKER_IMAGE = "pipeline-youtube-capture:latest"


def _caller_uid_gid() -> tuple[int, int]:
    """Return the host process's effective UID/GID for `--user` mapping.

    Bind-mounted `tmp/` and `_assets/pipeline-youtube/` are created on
    the host with the caller's ownership, so the container must run as
    the same UID/GID to be able to write them back. Hard-coding
    1000:1000 breaks on hosts where the pipeline runs as a different
    user (including root-owned CI environments); use the caller's IDs
    instead. Windows (where `os.getuid` is missing) falls back to
    1000:1000 — Docker Desktop on Windows doesn't enforce host-side
    ownership on bind mounts so the fallback is benign.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return 1000, 1000
    return getuid(), getgid()


class DockerBackendNotReady(CaptureBackendError):
    """Raised when the docker daemon is unavailable or the image is missing."""


@dataclass(frozen=True)
class DockerCaptureBackend:
    """R1-mitigation backend: runs yt-dlp / ffmpeg / gif2webp in a hardened container.

    Every invocation issues a **one-off** ``docker run --rm`` with
    hardened flags. State persists only on bind-mounted host paths
    (``tmp_dir`` and ``assets_dir``). The container itself has no
    capabilities, no network beyond bridge mode (yt-dlp needs HTTPS),
    and a read-only root filesystem.

    The backend does not manage image lifecycle — the user builds
    the image via ``docker build -f docker/Dockerfile.capture -t
    pipeline-youtube-capture:latest .`` (documented in
    docs/docker.md). ``preflight()`` verifies presence.
    """

    tmp_dir: Path
    assets_dir: Path
    image: str = DEFAULT_DOCKER_IMAGE
    docker_bin: str = "docker"
    name: ClassVar[str] = "docker"

    def preflight(self) -> None:
        """Fail early if docker CLI / daemon / image is unavailable.

        Called from `run_stage_capture` before any per-video work so
        the user sees one clear error instead of per-video failures.
        """
        if shutil.which(self.docker_bin) is None:
            raise DockerBackendNotReady(
                f"docker CLI not found in PATH ({self.docker_bin!r}). "
                "Install Docker Desktop or switch capture_backend to 'host'."
            )
        try:
            subprocess.run(
                [self.docker_bin, "image", "inspect", self.image],
                capture_output=True,
                check=True,
                timeout=15,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-200:]
            raise DockerBackendNotReady(
                f"docker image {self.image!r} not found. Build it with:\n"
                f"  docker build -f docker/Dockerfile.capture -t {self.image} .\n"
                f"stderr: {stderr}"
            ) from e
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise DockerBackendNotReady(
                f"docker daemon unreachable: {type(e).__name__}: {e}"
            ) from e

    def _base_args(self, *, network: bool) -> list[str]:
        """Common hardening flags shared by every docker run invocation.

        Flags:
          --rm                         : auto-remove container on exit
          --read-only                  : root FS is read-only
          --cap-drop=ALL               : no kernel capabilities
          --security-opt=no-new-privs  : no setuid escalation
          --user {uid}:{gid}           : caller's UID/GID, never root
          --tmpfs /tmp:...             : writable scratch for yt-dlp/ffmpeg
          --network=none | bridge      : off for ffmpeg, on for yt-dlp
          -v tmp:/work                 : mp4 lives here
          -v assets:/assets            : extracted images land here
        """
        net = "bridge" if network else "none"
        uid, gid = _caller_uid_gid()
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--user={uid}:{gid}",
            "--tmpfs=/tmp:rw,size=512m,nosuid,nodev",
            f"--network={net}",
            "-v",
            f"{self.tmp_dir}:/work",
            "-v",
            f"{self.assets_dir}:/assets",
        ]

    def _host_to_container(self, host_path: Path) -> str:
        """Translate a host Path to its in-container equivalent.

        Raises CaptureBackendError if the path isn't under one of the
        bind-mounted directories — this prevents accidentally asking
        the container to read files it has no mount for.
        """
        resolved = host_path.resolve()
        tmp_resolved = self.tmp_dir.resolve()
        assets_resolved = self.assets_dir.resolve()
        try:
            return "/work/" + str(resolved.relative_to(tmp_resolved))
        except ValueError:
            pass
        try:
            return "/assets/" + str(resolved.relative_to(assets_resolved))
        except ValueError:
            pass
        raise CaptureBackendError(
            f"path {host_path!s} is not under tmp ({self.tmp_dir}) "
            f"or assets ({self.assets_dir}); DockerCaptureBackend cannot mount it."
        )

    def _translate_args(self, args: list[str]) -> list[str]:
        """Replace host absolute paths in args with their container paths.

        Detection: only **absolute** paths are candidates for rewriting;
        flags (leading `-`) and everything else (option values like
        ``"libwebp"``, format strings like ``"fps=5,scale=..."``,
        relative paths) pass through unchanged. Stage 03 always hands
        us absolute paths — ``_assert_not_flaglike`` upstream ensures
        they start with ``/`` — so there is no production path where a
        relative path needs container translation.
        """
        translated: list[str] = []
        for a in args:
            if a.startswith("-") or not Path(a).is_absolute():
                translated.append(a)
                continue
            try:
                translated.append(self._host_to_container(Path(a)))
            except CaptureBackendError:
                # Not a path we can mount — let the container see the
                # raw value and fail cleanly if it actually needed it.
                translated.append(a)
        return translated

    def download_video(self, url: str, dest: Path, *, resolution: str) -> None:
        container_dest = self._host_to_container(dest)
        _clear_ytdlp_outputs(dest)
        fmt = (
            f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={resolution}][ext=mp4]/"
            f"best[height<={resolution}]"
        )
        # `yt-dlp` runs with network=bridge; everything else stays off.
        cmd = [
            *self._base_args(network=True),
            self.image,
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "--no-progress",
            "--merge-output-format",
            "mp4",
            "-f",
            fmt,
            "-o",
            # Preserve host's ".{ext}" templating so the in-container
            # output lands at /work/{stem}.{ext}, which maps back to
            # {tmp_dir}/{stem}.{ext} on the host.
            str(Path(container_dest).with_suffix("")) + ".%(ext)s",
            url,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-300:]
            raise CaptureBackendError(
                f"yt-dlp (docker) exited {e.returncode}; stderr: {stderr}"
            ) from e

        # yt-dlp may have appended a non-mp4 extension; handle same as host.
        _adopt_ytdlp_output(dest)

        with contextlib.suppress(OSError):
            os.chmod(dest, 0o600)

    def ffmpeg(self, args: list[str], *, timeout: int) -> None:
        translated = self._translate_args(args)
        cmd = [*self._base_args(network=False), self.image, "ffmpeg", *translated]
        subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)

    def gif2webp(self, args: list[str], *, timeout: int) -> None:
        translated = self._translate_args(args)
        cmd = [*self._base_args(network=False), self.image, "gif2webp", *translated]
        subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)

    def ffmpeg_encoders(self) -> frozenset[str]:
        try:
            result = subprocess.run(
                [
                    *self._base_args(network=False),
                    self.image,
                    "ffmpeg",
                    "-hide_banner",
                    "-encoders",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return frozenset()

        encoders: set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] and parts[0][0] in {"V", "A", "S"}:
                encoders.add(parts[1])
        return frozenset(encoders)

    def has_gif2webp(self) -> bool:
        """Assumes the image was built from `docker/Dockerfile.capture`.

        The shipped Dockerfile installs the `webp` apt package, which
        provides `gif2webp`. Users who point `capture_docker_image` at
        a custom image without that package will see this method lie
        and hit a ``gif2webp: command not found`` at capture time.
        If that becomes a real operational issue, probe the binary
        inside the container (``docker run ... which gif2webp``) on
        first use and cache the result.
        """
        return True
