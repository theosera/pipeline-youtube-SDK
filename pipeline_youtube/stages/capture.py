"""Stage 03: animated capture at stage-02 theme-range centers.

Flow:
    1. Parse `[MM:SS ~ MM:SS]` headings from stage 02's 02_Summary md
    2. Download the video via yt-dlp to pipeline-youtube/tmp/ (480p mp4)
    3. For each range, run ffmpeg to extract an animated clip centered
       on `(start + end) / 2`, duration = `window_seconds`, fps = 5
    4. Save files into a per-playlist subfolder of
       `Permanent Note/_assets/2026/pipeline-youtube/` as
       `pyt_{video_id}_{idx}.{ext}` — a naming scheme Obsidian's
       Attachment Management leaves alone, nested under a subfolder whose
       name matches the playlist's 01~05 note folders so the captures can
       be deleted alongside them.
    5. Append `[MM:SS ~ MM:SS]\\n![[{playlist_folder}/filename.ext]]`
       blocks to the 03_Capture placeholder md (the embed is path-qualified
       so duplicate basenames across playlists stay unambiguous in Obsidian)
    6. Delete the temp video file

Capture format (WebP vs GIF)
----------------------------
WebP is preferred (~3x smaller than GIF for the same content). Three
extraction strategies are supported, picked automatically in this
priority order:

    1. `direct`: ffmpeg `-c:v libwebp` (1-pass, fastest)
        → needs ffmpeg built with `--enable-libwebp`.
    2. `via_gif2webp`: ffmpeg 2-pass GIF then `gif2webp` conversion
        → works when ffmpeg lacks libwebp but `gif2webp` binary exists
          (e.g. `brew install webp`). Still produces `.webp` output.
    3. `native_gif`: ffmpeg 2-pass GIF only
        → universal fallback, outputs `.gif` files.

The caller can force a strategy via `capture_format="webp"|"gif"|"auto"`.
Default is `"auto"`, which walks the priority order above.

Failures per-range are isolated: if one ffmpeg call fails, the rest
still run and the md records the failure as an HTML comment.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config import get_vault_root
from ..path_safety import ensure_safe_path
from ..playlist import VideoMeta
from .capture_backend import CaptureBackend, HostCaptureBackend

# Resource-class semaphore (Phase 3 A): bounds concurrent video downloads
# (yt-dlp network I/O) independently of the per-video --concurrency. ``None`` =
# unbounded (default, preserves prior behavior). Covers both the Stage 03
# download and the parallel prefetch thread since both go through
# ``_download_video``.
_download_semaphore: threading.BoundedSemaphore | None = None


def configure_download_concurrency(limit: int | None) -> None:
    """Cap concurrent video downloads. ``None`` / <=0 removes the cap."""
    global _download_semaphore
    _download_semaphore = threading.BoundedSemaphore(limit) if limit and limit > 0 else None


# Pipeline-managed subfolder separate from Obsidian's default _assets/2026/img/.
# Reason: Obsidian's Attachment Management plugin treats img/ as an auto-managed
# attachment folder and will rename files to match note titles (`${notename}`
# template) whenever it detects a wiki-link. By using a distinct folder +
# a `pyt_<video_id>_<idx>.webp` naming scheme that doesn't match any note
# title, Attachment Management leaves our files alone.
ASSETS_REL_PATH = "Permanent Note/_assets/2026/pipeline-youtube"
DEFAULT_WINDOW_SECONDS = 3.5
DEFAULT_FPS = 5
DEFAULT_SCALE_HEIGHT = 480
DEFAULT_RESOLUTION = "480"

CaptureFormat = Literal["auto", "webp", "gif"]

_TMP_SWEEP_EXTENSIONS = (".mp4", ".webm", ".m4a", ".mkv")


def _tmp_video_path(video: VideoMeta) -> Path:
    """Canonical temp path for a video's downloaded mp4.

    Directory permissions are tightened to 0o700 (owner-only) at
    creation and on every call so the video binary never becomes
    world-readable on shared hosts.
    """
    import os

    project_root = Path(__file__).resolve().parent.parent.parent
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        # Non-POSIX filesystems (e.g. FAT) silently ignore — best-effort.
        os.chmod(tmp_dir, 0o700)
    return tmp_dir / f"{video.video_id}.mp4"


@dataclass
class VideoPrefetch:
    """Handle for a background video download started before Stage 02."""

    path: Path
    future: Any  # concurrent.futures.Future[None]

    def wait(self, timeout: float | None = 600.0) -> Exception | None:
        """Block until the download finishes. Returns the exception (if any).

        `timeout=None` blocks until the download thread completes (success
        or failure). Callers that own `path` must wait to completion before
        any fallback download to the same path, otherwise a queued prefetch
        can outlive the timeout and race the fallback on `path`.
        """
        try:
            self.future.result(timeout=timeout)
            return None
        except Exception as exc:  # noqa: BLE001 — propagate as return value
            return exc


def prefetch_video_download(
    video: VideoMeta,
    resolution: str = DEFAULT_RESOLUTION,
    *,
    backend: CaptureBackend | None = None,
) -> VideoPrefetch:
    """Kick off a video download on a daemon thread and return the handle.

    The caller should `wait()` on the handle before calling
    `run_stage_capture(..., prefetched_video_path=handle.path)`.

    `backend` defaults to `HostCaptureBackend()` for backward compat.
    Pass a `DockerCaptureBackend` instance to download inside the
    hardened container (slower per-call due to docker start overhead,
    but eliminates the R1 residual risk).
    """
    from concurrent.futures import ThreadPoolExecutor

    path = _tmp_video_path(video)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"pyt-dl-{video.video_id}")
    future = executor.submit(_download_video, video.watch_url, path, resolution, backend=backend)
    executor.shutdown(wait=False)  # thread keeps running until task completes
    return VideoPrefetch(path=path, future=future)


def _assert_not_flaglike(path: Path) -> None:
    """Guard against path arguments whose *string form* starts with `-`.

    ffmpeg (and other CLIs) would interpret a leading dash as a flag.
    Our paths are always absolute (built from `project_root / ...`) so
    the string form starts with `/` — this check catches regressions if
    a relative path ever slips in.
    """
    arg = str(path)
    if arg.startswith("-"):
        raise ValueError(f"path argument starts with '-' (flag-like): {arg!r}")


def sweep_stale_tmp(tmp_dir: Path, *, older_than_hours: float = 24.0) -> int:
    """Remove stale video tempfiles left behind by OOM-killed runs.

    Called at CLI startup. Returns the number of files removed.
    Missing dir is a no-op (returns 0).
    """
    if not tmp_dir.exists():
        return 0
    import time

    cutoff = time.time() - older_than_hours * 3600.0
    removed = 0
    for entry in tmp_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in _TMP_SWEEP_EXTENSIONS:
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


# Match '### [MM:SS ~ MM:SS] 見出し' — tolerates full-width ~ and ASCII ~.
_RANGE_PATTERN = re.compile(
    r"^###\s*\[\s*(\d{1,2}):(\d{2})\s*[~〜～]\s*(\d{1,2}):(\d{2})\s*\]\s*(.+?)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SummaryRange:
    start_sec: int
    end_sec: int
    heading: str

    @property
    def center_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0

    @property
    def start_mmss(self) -> str:
        mm, ss = divmod(self.start_sec, 60)
        return f"{mm:02d}:{ss:02d}"

    @property
    def end_mmss(self) -> str:
        mm, ss = divmod(self.end_sec, 60)
        return f"{mm:02d}:{ss:02d}"


@dataclass(frozen=True)
class CaptureOutcome:
    range: SummaryRange
    image_path: Path | None  # None if extraction failed
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.image_path is not None


@dataclass(frozen=True)
class CaptureResult:
    ranges: list[SummaryRange]
    outcomes: list[CaptureOutcome] = field(default_factory=list)
    video_downloaded: bool = False
    capture_format: str | None = None  # resolved format actually used
    error: str | None = None

    @property
    def success_count(self) -> int:
        return sum(1 for o in self.outcomes if o.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for o in self.outcomes if not o.success)

    @property
    def image_paths(self) -> list[Path]:
        return [o.image_path for o in self.outcomes if o.image_path is not None]


# =====================================================
# ffmpeg capability detection
# =====================================================


ExtractStrategy = Literal["direct", "via_gif2webp", "native_gif"]


@dataclass(frozen=True)
class _FormatChoice:
    ext: Literal["webp", "gif"]
    strategy: ExtractStrategy


def _resolve_capture_format(requested: CaptureFormat, backend: CaptureBackend) -> _FormatChoice:
    """Decide the output extension + extraction strategy.

    Priority for `auto`: libwebp encoder > gif2webp > native GIF.
    Capability probes are delegated to the backend (host or docker) so
    the decision reflects what will actually execute the capture.
    """
    encoders = backend.ffmpeg_encoders()
    has_libwebp = "libwebp" in encoders or "libwebp_anim" in encoders
    has_gif2webp = backend.has_gif2webp()

    if requested == "webp":
        if has_libwebp:
            return _FormatChoice(ext="webp", strategy="direct")
        if has_gif2webp:
            return _FormatChoice(ext="webp", strategy="via_gif2webp")
        raise RuntimeError(
            "capture_format='webp' requested but neither ffmpeg libwebp encoder "
            "nor `gif2webp` binary is available. Install via `brew install webp` "
            "or switch capture_format to 'gif'/'auto'."
        )

    if requested == "gif":
        return _FormatChoice(ext="gif", strategy="native_gif")

    # auto
    if has_libwebp:
        return _FormatChoice(ext="webp", strategy="direct")
    if has_gif2webp:
        return _FormatChoice(ext="webp", strategy="via_gif2webp")
    return _FormatChoice(ext="gif", strategy="native_gif")


# =====================================================
# Public API
# =====================================================


def parse_summary_ranges(summary_md: str) -> list[SummaryRange]:
    """Extract semantic ranges from stage 02's markdown output.

    Only matches the h3 heading format produced by stage 02:
        `### [MM:SS ~ MM:SS] heading text`

    Ignores malformed ranges where end <= start.
    """
    ranges: list[SummaryRange] = []
    for m in _RANGE_PATTERN.finditer(summary_md):
        start = int(m.group(1)) * 60 + int(m.group(2))
        end = int(m.group(3)) * 60 + int(m.group(4))
        heading = m.group(5).strip()
        if end > start:
            ranges.append(SummaryRange(start, end, heading))
    return ranges


def run_stage_capture(
    video: VideoMeta,
    summary_md_path: Path,
    capture_md_path: Path,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    fps: int = DEFAULT_FPS,
    scale_height: int = DEFAULT_SCALE_HEIGHT,
    resolution: str = DEFAULT_RESOLUTION,
    capture_format: CaptureFormat = "auto",
    dry_run: bool = False,
    prefetched_video_path: Path | None = None,
    backend: CaptureBackend | None = None,
) -> CaptureResult:
    """Download the video, extract animated frames, update the 03 md.

    Image files are named `pyt_{video_id}_{idx}.{ext}` and placed in
    `Permanent Note/_assets/2026/pipeline-youtube/{playlist_folder}/` — a
    per-playlist subfolder (named to match the 01~05 unit folders) inside a
    dedicated assets dir outside Obsidian's Attachment Management
    `${notename}` auto-rename scope. The per-playlist nesting means
    deleting a playlist's 01~05 note folders has a matching assets
    subfolder to drop, so captures never linger as orphans.

    Parameters
    ----------
    capture_format:
        `"auto"` (default): WebP if ffmpeg has libwebp, else GIF.
        `"webp"`: force WebP, error if missing encoder.
        `"gif"`: force GIF (2-pass palette).
    prefetched_video_path:
        Optional existing mp4 file prepared by a background thread
        (see `prefetch_video_download`). When supplied and present,
        the internal yt-dlp download is skipped.
    """
    if not summary_md_path.exists():
        return CaptureResult(ranges=[], error="summary_md_not_found")

    summary_md = summary_md_path.read_text(encoding="utf-8")
    ranges = parse_summary_ranges(summary_md)

    if not ranges:
        return CaptureResult(ranges=[], error="no_ranges_parsed")

    if dry_run:
        return CaptureResult(ranges=ranges, capture_format=capture_format)

    if backend is None:
        active_backend: CaptureBackend = HostCaptureBackend()
    else:
        active_backend = backend

    try:
        choice = _resolve_capture_format(capture_format, active_backend)
    except RuntimeError as e:
        return CaptureResult(ranges=ranges, error=f"format_unavailable: {e}")

    ext = choice.ext

    # Resolve the assets dir (subject to path safety). Captures are nested
    # under a per-playlist subfolder whose name is taken from the 03_Capture
    # md's parent — i.e. the same `{playlist_folder}` used by the 01~05 unit
    # dirs. Deleting a playlist's 01~05 note folders then maps to one obvious
    # assets subfolder, instead of leaving orphaned webp/gif files behind.
    vault_root = get_vault_root()
    playlist_folder = capture_md_path.parent.name
    assets_rel = ensure_safe_path(f"{ASSETS_REL_PATH}/{playlist_folder}")
    assets_dir = vault_root / assets_rel
    assets_dir.mkdir(parents=True, exist_ok=True)

    from ..cache import get_cache

    cache = get_cache()
    # `cleanup_path` is the working copy to delete in `finally`. A cache HIT
    # points extraction at the persistent copy, which must NOT be deleted.
    cleanup_path: Path | None = None

    if prefetched_video_path is not None and prefetched_video_path.exists():
        tmp_video_path = prefetched_video_path
        cleanup_path = prefetched_video_path
        cache.put_video(video.video_id, resolution, prefetched_video_path)
    else:
        cached_video = cache.get_video(video.video_id, resolution)
        if cached_video is not None:
            tmp_video_path = cached_video  # reuse persistent copy; do not delete
        else:
            tmp_video_path = _tmp_video_path(video)
            cleanup_path = tmp_video_path
            try:
                _download_video(
                    video.watch_url,
                    tmp_video_path,
                    resolution=resolution,
                    backend=active_backend,
                )
            except Exception as e:
                return CaptureResult(
                    ranges=ranges,
                    capture_format=ext,
                    error=f"download_failed: {type(e).__name__}: {e}",
                )
            cache.put_video(video.video_id, resolution, tmp_video_path)

    extractor = _dispatch_extractor(choice.strategy)

    outcomes: list[CaptureOutcome] = []
    success_counter = 0
    try:
        for rng in ranges:
            image_name = _capture_image_name(video.video_id, success_counter, ext)
            image_path = assets_dir / image_name

            start = max(0.0, rng.center_sec - window_seconds / 2.0)
            try:
                extractor(
                    tmp_video_path,
                    image_path,
                    start_sec=start,
                    duration=window_seconds,
                    fps=fps,
                    scale_height=scale_height,
                    backend=active_backend,
                )
            except subprocess.CalledProcessError as e:
                stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-200:]
                outcomes.append(
                    CaptureOutcome(
                        range=rng,
                        image_path=None,
                        error=f"ffmpeg_exit_{e.returncode}: {stderr}",
                    )
                )
                continue
            except Exception as e:
                outcomes.append(
                    CaptureOutcome(
                        range=rng,
                        image_path=None,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
                continue

            outcomes.append(CaptureOutcome(range=rng, image_path=image_path))
            success_counter += 1

        if outcomes:
            body = _render_body(outcomes)
            _append_body(capture_md_path, body)

        return CaptureResult(
            ranges=ranges,
            outcomes=outcomes,
            video_downloaded=True,
            capture_format=ext,
        )
    finally:
        # Only delete the working copy. When `--no-cache` is set the download
        # path above is the working copy, so delete-after-use is preserved.
        if cleanup_path is not None:
            with contextlib.suppress(OSError):
                cleanup_path.unlink(missing_ok=True)


# =====================================================
# Internals
# =====================================================


def _capture_image_name(video_id: str, idx: int, ext: str = "webp") -> str:
    """Generate a stable image filename that Attachment Management ignores.

    Format: `pyt_{video_id}_{idx:02d}.{ext}` — deliberately does NOT
    match the `${notename}` template Obsidian uses, so the plugin
    treats these files as "not our attachments" and leaves them
    alone. Contiguous zero-padded indices starting from 00.
    """
    return f"pyt_{video_id}_{idx:02d}.{ext}"


def _download_video(
    url: str,
    dest: Path,
    resolution: str = "480",
    *,
    backend: CaptureBackend | None = None,
) -> None:
    """Download a single video at <= `resolution` height to `dest`.

    Delegates the actual yt-dlp call to the configured backend (host
    subprocess or hardened container). `backend=None` preserves the
    prefetch-thread signature that existed before R1 hardening and
    defaults to the host backend.
    """
    if dest.exists():
        dest.unlink()

    active = backend or HostCaptureBackend()
    sem = _download_semaphore
    if sem is not None:
        with sem:
            active.download_video(url, dest, resolution=resolution)
    else:
        active.download_video(url, dest, resolution=resolution)


ExtractorFn = Callable[..., None]


def _dispatch_extractor(strategy: ExtractStrategy) -> ExtractorFn:
    if strategy == "direct":
        return _extract_webp_direct
    if strategy == "via_gif2webp":
        return _extract_webp_via_gif2webp
    if strategy == "native_gif":
        return _extract_gif
    raise ValueError(f"unknown strategy: {strategy}")


def _extract_webp_direct(
    video_path: Path,
    output_path: Path,
    *,
    start_sec: float,
    duration: float,
    fps: int,
    scale_height: int,
    backend: CaptureBackend,
) -> None:
    """Strategy 1: ffmpeg with -c:v libwebp (1-pass, requires libwebp encoder).

    `scale=-2:H` preserves aspect ratio and ensures even width.
    """
    _assert_not_flaglike(video_path)
    _assert_not_flaglike(output_path)
    args = [
        "-ss",
        f"{start_sec:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale=-2:{scale_height}:flags=lanczos",
        "-c:v",
        "libwebp",
        "-loop",
        "0",
        "-an",
        "-y",
        str(output_path),
    ]
    backend.ffmpeg(args, timeout=180)


def _extract_webp_via_gif2webp(
    video_path: Path,
    output_path: Path,
    *,
    start_sec: float,
    duration: float,
    fps: int,
    scale_height: int,
    backend: CaptureBackend,
) -> None:
    """Strategy 2: ffmpeg 2-pass GIF then gif2webp conversion.

    Used when ffmpeg lacks libwebp but gif2webp binary is installed
    (typical Homebrew `brew install webp` setup). Produces a `.webp`
    file roughly 3x smaller than the intermediate GIF.
    """
    # Step A: write 2-pass GIF to a sibling temp file
    tmp_gif = output_path.with_suffix(".tmp.gif")
    try:
        _extract_gif(
            video_path,
            tmp_gif,
            start_sec=start_sec,
            duration=duration,
            fps=fps,
            scale_height=scale_height,
            backend=backend,
        )
        # Step B: gif2webp converts animated GIF -> animated WebP
        #   -q 75: quality (lossy) — good balance for UI/code captures
        #   -mt: multi-threaded (faster)
        #   -m 6: compression method 6 = max quality/size tradeoff
        args = [
            "-q",
            "75",
            "-m",
            "6",
            "-mt",
            "-quiet",
            str(tmp_gif),
            "-o",
            str(output_path),
        ]
        backend.gif2webp(args, timeout=120)
    finally:
        with contextlib.suppress(OSError):
            tmp_gif.unlink(missing_ok=True)


def _extract_gif(
    video_path: Path,
    output_path: Path,
    *,
    start_sec: float,
    duration: float,
    fps: int,
    scale_height: int,
    backend: CaptureBackend,
) -> None:
    """Extract an animated GIF via 2-pass palette (palettegen + paletteuse).

    2-pass gives significantly better color quality than naive 1-pass
    since GIF is limited to 256 colors and needs a curated palette.
    Always available on any ffmpeg build (no external dependencies).
    """
    vf = f"fps={fps},scale=-2:{scale_height}:flags=lanczos"

    _assert_not_flaglike(video_path)
    _assert_not_flaglike(output_path)
    # Pass 1: generate optimized palette
    palette_path = output_path.with_suffix(".palette.png")
    palettegen_args = [
        "-ss",
        f"{start_sec:.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        str(video_path),
        "-vf",
        f"{vf},palettegen",
        "-y",
        str(palette_path),
    ]
    backend.ffmpeg(palettegen_args, timeout=120)

    try:
        # Pass 2: apply palette to produce the final GIF
        paletteuse_args = [
            "-ss",
            f"{start_sec:.2f}",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(video_path),
            "-i",
            str(palette_path),
            "-lavfi",
            f"{vf} [x]; [x][1:v] paletteuse",
            "-loop",
            "0",
            "-an",
            "-y",
            str(output_path),
        ]
        backend.ffmpeg(paletteuse_args, timeout=180)
    finally:
        with contextlib.suppress(OSError):
            palette_path.unlink(missing_ok=True)


def _render_body(outcomes: list[CaptureOutcome]) -> str:
    """Render capture md body: one range per block."""
    blocks: list[str] = []
    for outcome in outcomes:
        rng = outcome.range
        header = f"[{rng.start_mmss} ~ {rng.end_mmss}]"
        if outcome.image_path is not None:
            # Qualify the embed with the per-playlist subfolder. Because the
            # same video can be captured into multiple playlist folders (e.g.
            # a rerun creates a new dated folder), bare `![[pyt_<id>_NN.ext]]`
            # links would collide on basename and Obsidian would resolve them
            # ambiguously. `{playlist_folder}/{name}` is a unique path suffix.
            embed = f"{outcome.image_path.parent.name}/{outcome.image_path.name}"
            blocks.append(f"{header}\n![[{embed}]]")
        else:
            blocks.append(f"{header}\n<!-- capture failed: {outcome.error} -->")
    return "\n\n".join(blocks) + "\n"


def _append_body(path: Path, body: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"placeholder md not found: {path}")
    existing = path.read_text(encoding="utf-8")
    if existing.endswith("\n\n"):
        sep = ""
    elif existing.endswith("\n"):
        sep = "\n"
    else:
        sep = "\n\n"
    path.write_text(existing + sep + body, encoding="utf-8")
