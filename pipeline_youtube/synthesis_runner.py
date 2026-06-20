"""Stage 05 周辺制御 (synthesis runner)。

固有名詞シートのユーザー訂正を反映する用語集を組み立て、Stage 05 統合
(``run_stage_synthesis``) を実行して結果を返す。「Stage 05 の工程管理者」。
統合本体の HOW は ``stages/synthesis`` が持ち、ここは入力準備と起動のみ。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from .cli_types import CliRequest, ResolvedInput, Runtime
from .glossary import Glossary, correction_glossary, load_sheet
from .playlist import VideoMeta
from .stages.synthesis import (
    SynthesisStageResult,
    log_synthesis_preflight,
    run_stage_synthesis,
)
from .synthesis.agents import compute_synthesis_timeouts


def run_synthesis(
    request: CliRequest,
    runtime: Runtime,
    resolved: ResolvedInput,
    run_time: datetime,
    synthesis_videos: list[VideoMeta],
    synthesis_bodies: list[str],
    folder_override: str | None,
    proper_noun_sheet_path: Path | None,
) -> SynthesisStageResult:
    """Build the proper-noun glossary and run Stage 05 synthesis."""
    # Apply the user's proper-noun corrections to the Stage 05 output: build a
    # glossary from the sheet's user-corrected rows (correction = canonical,
    # system spelling = alias) and rewrite the MOC + chapters with it.
    proper_noun_glossary: Glossary | None = None
    if proper_noun_sheet_path is not None:
        sheet_glossary = correction_glossary(load_sheet(proper_noun_sheet_path))
        if sheet_glossary.entries:
            proper_noun_glossary = sheet_glossary

    click.echo("\n=== Stage 05 Synthesis (Agent Teams) ===")
    synth_timeouts = compute_synthesis_timeouts(
        len(synthesis_videos), override=runtime.synthesis_timeout
    )
    click.echo(log_synthesis_preflight(len(synthesis_videos), synthesis_bodies, synth_timeouts))
    return run_stage_synthesis(
        synthesis_videos,
        synthesis_bodies,
        run_time=run_time,
        playlist_title=resolved.playlist_title,
        model=request.model,
        agent_models={k: runtime.models[k] for k in ("alpha", "beta", "leader", "reviewer")},
        min_playlist_size=request.min_playlist_size,
        max_chapters=request.max_chapters,
        dry_run=request.dry_run,
        folder_name_override=folder_override,
        synthesis_timeout=runtime.synthesis_timeout,
        profile=runtime.synthesis_profile,
        proper_noun_glossary=proper_noun_glossary,
        cache=runtime.cache,
        vault_root=runtime.vault_root,
    )
