"""入力解決 (input resolution)。

YouTube URL からメタデータを取得するか、``--local-media`` でローカル動画を
走査して動画リストを作り、Stage 00.5 ルーターで code_bearing を分類する。
「材料を揃える係」。取得・走査・分類の HOW は ``playlist`` / ``local_media`` /
``genres`` が持ち、ここは配線のみ。
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .cli_types import CliRequest, ResolvedInput, Runtime
from .genres import CODE_BEARING_GENRES, classify_playlist_genre
from .playlist import fetch_metadata


def resolve_input(request: CliRequest, runtime: Runtime) -> ResolvedInput:
    """Build the video list (YouTube or local) and classify its genre."""
    # --local-media: build the video list from a local folder (no YouTube).
    # media_map (video_id → file path) is threaded into stages 01/03 so they
    # transcribe/capture the local file instead of downloading. Empty otherwise.
    media_map: dict[str, Path] = {}
    if request.local_media:
        from .local_media import build_local_videos

        videos, media_map = build_local_videos(request.local_media)
        if not videos:
            click.echo(f"No media files found in {request.local_media}")
            sys.exit(1)
        click.echo(f"local-media: {len(videos)} file(s) from {request.local_media}")
    else:
        if request.url is None:
            raise click.UsageError(
                "A playlist/video URL is required unless --local-media is given."
            )
        click.echo("fetching metadata...")
        videos = fetch_metadata(request.url)
        if not videos:
            click.echo("No videos found.")
            sys.exit(1)

    playlist_title = videos[0].playlist_title or videos[0].title or "single video"
    click.echo(f"playlist: {playlist_title!r}")
    click.echo(f"videos: {len(videos)}")

    # Stage 00.5: Router. One cheap haiku call decides whether downstream
    # code-bearing features (GitHub URL extraction, concept/practice split)
    # apply. Errors collapse to Genre.OTHER → default behavior. The parent
    # classifies once and pins the result for every sub-agent shard (internal
    # --code-bearing/--no-code-bearing), so a transient router error on one
    # worker can't leave shards disagreeing on code_bearing.
    if request.code_bearing_override is not None:
        code_bearing = request.code_bearing_override
        click.echo(f"genre: (inherited from parent) code_bearing={code_bearing}")
    else:
        genre, genre_rationale = classify_playlist_genre(
            playlist_title, videos, model=runtime.models["router"], cache=runtime.cache
        )
        code_bearing = genre in CODE_BEARING_GENRES
        click.echo(f"genre: {genre.value} (code_bearing={code_bearing}) — {genre_rationale[:120]}")

    return ResolvedInput(
        videos=videos,
        media_map=media_map,
        playlist_title=playlist_title,
        code_bearing=code_bearing,
    )
