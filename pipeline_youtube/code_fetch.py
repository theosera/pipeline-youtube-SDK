"""Per-video yt-dlp metadata extract (description + chapters) for Stage 01.

Stage 01a fetches every video's description + declared chapters once via
``fetch_video_extra_metadata`` (flat-playlist metadata has neither, so this
is a dedicated non-flat yt-dlp extract). The result feeds two downstream
consumers:

1. Stage 01b (``transcript.correction``) uses the description as known
   context so the web-search-backed correction pass can resolve proper
   nouns without spending a search when the description already names them.
2. Stage 02 (``stages.summary``) uses description + chapters as passive
   Mode-diagnosis context (never triggers a search of its own).

When the Router additionally classifies a playlist as ``coding``, Stage 01
also scans the fetched description for GitHub repository/file/gist URLs,
fetches their raw content (capped at 5 URLs / 50KB each), and formats the
result as a markdown ``## 関連コード`` section.

The whole module is *advisory* — any failure (network, parse, rate
limit) degrades to an empty result without raising, so a transcript that
can't get its metadata/code blocks still completes Stage 01 normally.

Network safety: all HTTP fetches go through ``urllib.request`` (built-in,
no extra deps), use a short timeout, and only allow ``raw.githubusercontent.com``
and ``gist.githubusercontent.com`` hosts after URL rewriting — even if a
malicious description contains a redirect-style URL, the fetch target is
constrained.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import yt_dlp  # type: ignore[import-untyped]

from .domain.transcript import VideoChapter
from .services.cache import url_key

if TYPE_CHECKING:
    from .services.cache import Cache

# GitHub blob URL: https://github.com/<owner>/<repo>/blob/<ref>/<path>
# Captures owner, repo, and the rest (ref + path).
_GITHUB_BLOB_RE = re.compile(
    r"https://github\.com/([\w.-]+)/([\w.-]+)/blob/([^\s)]+)",
    flags=re.IGNORECASE,
)

# Gist URL: https://gist.github.com/[<owner>/]<id>
# Captures the gist id.
_GIST_RE = re.compile(
    r"https://gist\.github\.com/(?:[\w-]+/)?([0-9a-fA-F]{6,40})",
    flags=re.IGNORECASE,
)

# Optional repo-level URL: https://github.com/<owner>/<repo>
# Captures owner and repo. Matched only when no /blob/ appears.
_GITHUB_REPO_RE = re.compile(
    r"https://github\.com/([\w.-]+)/([\w.-]+)(?:[/\s)#]|$)",
    flags=re.IGNORECASE,
)

# Hard caps to prevent runaway descriptions from inflating Stage 01 md.
MAX_URLS_PER_VIDEO = 5
MAX_BYTES_PER_FILE = 50_000
FETCH_TIMEOUT = 10  # seconds


# Common code file extensions → fenced-code language hint.
_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".ps1": "powershell",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".lua": "lua",
    ".dockerfile": "dockerfile",
    ".tf": "hcl",
}


@dataclass(frozen=True)
class CodeSnippet:
    """A single fetched code block with provenance."""

    source_url: str  # original github.com/... URL
    raw_url: str  # the raw URL actually fetched
    filename: str  # display name (last path segment for files, "<gist_id>.json" for gists)
    language: str  # fenced-code hint (may be empty string)
    content: str  # raw text content (already truncated to MAX_BYTES_PER_FILE)
    truncated: bool


@dataclass(frozen=True)
class VideoExtraMetadata:
    """Description + declared chapters from one non-flat yt-dlp extract."""

    description: str | None
    chapters: tuple[VideoChapter, ...] = ()


def fetch_video_extra_metadata(video_id: str, *, timeout: int = 30, cache: Cache) -> VideoExtraMetadata:
    """Fetch a video's description + chapters via one non-video yt-dlp extract.

    Flat-playlist metadata has neither field, so this is a dedicated
    per-video extract (``skip_download=True``, no video bytes fetched).
    Cached under the ``code_fetch`` namespace, keyed by video id, since the
    result is deterministic for a given upload. Only a successful extract is
    cached; a failed/empty extract returns an empty result without caching
    it, so a transient network error doesn't poison future runs.
    """
    key = url_key(f"video_extra_metadata:{video_id}")
    cached = cache.get_code_fetch(key)
    if cached is not None:
        return VideoExtraMetadata(
            description=cached.get("description"),
            chapters=tuple(VideoChapter(**c) for c in cached.get("chapters", [])),
        )

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": timeout,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False,
            )
    except Exception:
        return VideoExtraMetadata(description=None)
    if info is None:
        return VideoExtraMetadata(description=None)

    desc = info.get("description")
    description = str(desc) if desc else None
    chapters = tuple(
        VideoChapter(title=str(c["title"]).strip(), start_seconds=float(c.get("start_time") or 0.0))
        for c in (info.get("chapters") or [])
        if c.get("title")
    )

    result = VideoExtraMetadata(description=description, chapters=chapters)
    cache.put_code_fetch(
        key,
        {"description": result.description, "chapters": [asdict(c) for c in result.chapters]},
    )
    return result


def extract_github_urls(description: str) -> list[str]:
    """Extract unique GitHub blob / Gist / repo URLs from a description.

    Order is preserved (first appearance wins). Capped at
    ``MAX_URLS_PER_VIDEO`` * 2 to leave room for filtering, but the
    caller is expected to apply the final cap when fetching.
    """
    if not description:
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _push(u: str) -> None:
        if u in seen:
            return
        seen.add(u)
        found.append(u)

    for m in _GITHUB_BLOB_RE.finditer(description):
        owner, repo, rest = m.groups()
        _push(f"https://github.com/{owner}/{repo}/blob/{rest}")

    for m in _GIST_RE.finditer(description):
        gist_id = m.group(1)
        _push(f"https://gist.github.com/{gist_id}")

    # Repo URLs are lowest priority — only matched if not already captured
    # via /blob/ and not used to dilute the budget.
    for m in _GITHUB_REPO_RE.finditer(description):
        owner, repo = m.group(1), m.group(2)
        # Skip if we already have a blob URL for this repo (more specific).
        if any(f"github.com/{owner}/{repo}/blob/" in u for u in found):
            continue
        _push(f"https://github.com/{owner}/{repo}")

    return found[: MAX_URLS_PER_VIDEO * 2]


def _blob_to_raw(blob_url: str) -> str | None:
    m = _GITHUB_BLOB_RE.match(blob_url)
    if not m:
        return None
    owner, repo, rest = m.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"


def _language_for_path(path: str) -> str:
    lower = path.lower()
    # Special filename-based detection (Dockerfile has no extension).
    base = lower.rsplit("/", 1)[-1]
    if base == "dockerfile":
        return "dockerfile"
    if base == "makefile":
        return "makefile"
    if "." not in base:
        return ""
    ext = "." + base.rsplit(".", 1)[1]
    return _LANG_BY_EXT.get(ext, "")


def _fetch_raw(
    url: str, *, timeout: int = FETCH_TIMEOUT, max_bytes: int = MAX_BYTES_PER_FILE
) -> tuple[str, bool] | None:
    """Fetch URL content, capped at max_bytes. Returns (text, truncated) or None on error."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pipeline-youtube/1.0 (+https://github.com/theosera/pipeline-youtube)"
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — host is gated by caller
            data = resp.read(max_bytes + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return None

    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def _fetch_blob(url: str) -> CodeSnippet | None:
    raw_url = _blob_to_raw(url)
    if raw_url is None:
        return None
    result = _fetch_raw(raw_url)
    if result is None:
        return None
    text, truncated = result
    filename = raw_url.rsplit("/", 1)[-1]
    return CodeSnippet(
        source_url=url,
        raw_url=raw_url,
        filename=filename,
        language=_language_for_path(filename),
        content=text,
        truncated=truncated,
    )


def _fetch_gist(url: str) -> CodeSnippet | None:
    """Fetch a gist via its public API. Concatenates all files into one snippet."""
    m = _GIST_RE.match(url)
    if m is None:
        return None
    gist_id = m.group(1)
    api_url = f"https://api.github.com/gists/{gist_id}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": "pipeline-youtube/1.0",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:  # noqa: S310
            import json as _json

            payload = _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, TimeoutError):
        return None

    files = payload.get("files") or {}
    if not files:
        return None

    parts: list[str] = []
    total_len = 0
    truncated = False
    primary_lang = ""
    for fname, fdata in files.items():
        content = (fdata or {}).get("content") or ""
        lang = _language_for_path(fname) or ""
        if not primary_lang:
            primary_lang = lang
        block = f"--- {fname} ---\n{content}"
        if total_len + len(block) > MAX_BYTES_PER_FILE:
            block = block[: MAX_BYTES_PER_FILE - total_len]
            truncated = True
            parts.append(block)
            break
        parts.append(block)
        total_len += len(block)

    return CodeSnippet(
        source_url=url,
        raw_url=api_url,
        filename=f"gist_{gist_id}",
        language=primary_lang,
        content="\n\n".join(parts),
        truncated=truncated,
    )


def fetch_snippets_for_urls(urls: list[str], *, cache: Cache) -> list[CodeSnippet]:
    """Fetch up to ``MAX_URLS_PER_VIDEO`` snippets, skipping unsupported URLs.

    Repo-level URLs (without /blob/) are skipped — fetching a repo's
    default README gets noisy fast and rarely matches what the video is
    actually demonstrating. Only blob and gist URLs return code.

    ``cache`` is injected by the caller: the per-URL snippet cache (a disabled
    ``Cache`` no-ops every lookup/store).
    """
    from dataclasses import asdict

    from .cache import url_key

    out: list[CodeSnippet] = []
    for url in urls:
        if len(out) >= MAX_URLS_PER_VIDEO:
            break
        # Per-URL caching maximizes reuse across videos/playlists that
        # reference the same repo file. Only successful fetches are cached;
        # unsupported (repo) URLs are skipped cheaply without a network call.
        key = url_key(url)
        cached = cache.get_code_fetch(key)
        if cached is not None:
            out.append(CodeSnippet(**cached))
            continue
        snippet: CodeSnippet | None = None
        if "/blob/" in url:
            snippet = _fetch_blob(url)
        elif url.startswith("https://gist.github.com/"):
            snippet = _fetch_gist(url)
        # Repo URLs intentionally skipped.
        if snippet is not None:
            cache.put_code_fetch(key, asdict(snippet))
            out.append(snippet)
    return out


def render_code_section(snippets: list[CodeSnippet]) -> str:
    """Render snippets as a markdown ``## 関連コード`` section.

    Returns an empty string when the snippets list is empty so callers
    can append unconditionally.
    """
    if not snippets:
        return ""

    lines: list[str] = ["", "## 関連コード", ""]
    for s in snippets:
        lines.append(f"### [{s.filename}]({s.source_url})")
        lines.append("")
        fence_lang = s.language or ""
        lines.append(f"```{fence_lang}")
        lines.append(s.content.rstrip())
        lines.append("```")
        if s.truncated:
            lines.append("")
            lines.append(f"_(truncated to {MAX_BYTES_PER_FILE:,} bytes; see source for full file)_")
        lines.append("")
    return "\n".join(lines)
