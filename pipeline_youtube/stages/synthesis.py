"""Stage 05: Playlist-level synthesis via Agent Teams.

Runs after all per-video stages (01-04) complete for a playlist with
≥3 successful videos. Reads every 04_Learning_Material md in the
playlist folder and orchestrates the α→β→Leader agent chain to
produce:

    {vault}/Permanent Note/08_YouTube学習/05_Synthesis/
        {YYYY-MM-DD <playlist_title>}/
            00_MOC.md
            01_<chapter>.md
            02_<chapter>.md
            ...
            _meta/
                duplicate_score.json

Execution is **sequential** (α→β→Leader) because the roles depend on
each other's output. Coverage (α topics vs β chapter topic_ids) is
computed deterministically in Python via `compute_coverage()` — no
LLM call. Claude's server-side cache shares context across consecutive
calls within ~5 minutes, so the cumulative cache-creation overhead is
paid only once in practice.

Skipping rules
--------------
- Playlists with fewer than `MIN_PLAYLIST_SIZE` (default 3) videos: skip.
- Playlists where stage 04 failed for all videos: skip.
- Single-video URLs: caller should not invoke this stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from ..config import get_vault_root
from ..obsidian import format_playlist_folder_name
from ..path_safety import ensure_safe_path
from ..playlist import VideoMeta
from ..synthesis.agents import (
    _MAX_INPUT_CHARS,
    AgentCallResult,
    call_alpha,
    call_alpha_batched,
    call_beta,
    call_leader,
    call_reviewer,
    compute_coverage,
    compute_synthesis_timeouts,
    rerun_leader_with_feedback,
)
from ..synthesis.body_validator import extract_allowed_embeds
from ..synthesis.chapter import write_chapter
from ..synthesis.moc import write_moc
from ..synthesis.scoring import (
    ChapterPlan,
    CoverageReport,
    LeaderOutput,
    ReviewerFeedback,
    SynthesisParseError,
    Topic,
)

SYNTHESIS_BASE = "Permanent Note/08_YouTube学習/05_Synthesis"
META_SUBDIR = "_meta"
DUPLICATE_SCORE_FILENAME = "duplicate_score.json"
MIN_PLAYLIST_SIZE = 3

# Maximum number of β reflexion retries when coverage has missing topics.
# Empirically: attempt #1 fixes the common "β forgot t015" case; attempt
# #2 catches structural misunderstandings that surface only after seeing
# the first retry feedback; attempt #3 is insurance. Beyond 3, β tends to
# be genuinely stuck (topic is too ambiguous to place) and more retries
# burn tokens without improving coverage.
MAX_BETA_REFLEXION_RETRIES = 3


class SynthesisProfile(StrEnum):
    """Agent Teams composition variants.

    - ``standard``: α → β → Leader (legacy / default for small playlists)
    - ``parallel``: α batched in parallel → merge → β → Leader
    - ``full``: α → β → Leader → Reviewer (quality-gated)
    - ``parallel+full``: parallel α + Reviewer
    """

    STANDARD = "standard"
    PARALLEL = "parallel"
    FULL = "full"
    PARALLEL_FULL = "parallel+full"

    @property
    def uses_parallel_alpha(self) -> bool:
        return "parallel" in self.value

    @property
    def uses_reviewer(self) -> bool:
        return "full" in self.value


# Auto-selection thresholds. The upper boundary of ``standard`` is
# inclusive: 15 videos still runs the legacy single-α path so cache
# reuse dominates the cost budget. Above 15 we switch to batched α.
_AUTO_STANDARD_MAX_VIDEOS = 15
_AUTO_PARALLEL_MAX_VIDEOS = 30


def _select_profile(
    n_videos: int,
    override: str | None,
) -> SynthesisProfile:
    """Choose the profile from an explicit override or the video count.

    ``override`` accepts profile names, ``"auto"``, or ``None`` (treated
    as auto). Invalid names raise ``ValueError``.
    """
    if override and override != "auto":
        return SynthesisProfile(override)
    if n_videos <= _AUTO_STANDARD_MAX_VIDEOS:
        return SynthesisProfile.STANDARD
    if n_videos <= _AUTO_PARALLEL_MAX_VIDEOS:
        return SynthesisProfile.PARALLEL
    return SynthesisProfile.PARALLEL_FULL


@dataclass(frozen=True)
class SynthesisStageResult:
    topics: list[Topic] = field(default_factory=list)
    chapters: list[ChapterPlan] = field(default_factory=list)
    coverage: CoverageReport | None = None
    leader_output: LeaderOutput | None = None
    moc_path: Path | None = None
    chapter_paths: list[Path] = field(default_factory=list)
    meta_path: Path | None = None
    agent_results: list[AgentCallResult] = field(default_factory=list)
    profile: SynthesisProfile | None = None
    reviewer_feedback: ReviewerFeedback | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens or 0 for r in self.agent_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens or 0 for r in self.agent_results)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(r.cache_creation_tokens or 0 for r in self.agent_results)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens or 0 for r in self.agent_results)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.total_cost_usd or 0.0 for r in self.agent_results)

    @property
    def total_duration_ms(self) -> int:
        return sum(r.duration_ms or 0 for r in self.agent_results)


def log_synthesis_preflight(
    n_videos: int,
    learning_md_bodies: list[str],
    timeouts: dict[str, int],
) -> str:
    """Build a preflight summary string for Stage 05.

    Compares actual input sizes against ``_MAX_INPUT_CHARS`` and reports
    whether any per-video truncation will occur.
    """
    per_video_limit = _MAX_INPUT_CHARS // max(n_videos, 1)
    total_chars = sum(len(b) for b in learning_md_bodies)
    truncated = sum(1 for b in learning_md_bodies if len(b) > per_video_limit)
    fill_pct = total_chars / _MAX_INPUT_CHARS * 100

    lines = [
        f"  videos: {n_videos}",
        f"  timeout: α={timeouts['alpha']}s β={timeouts['beta']}s leader={timeouts['leader']}s",
        f"  input: {total_chars:,}/{_MAX_INPUT_CHARS:,} chars ({fill_pct:.0f}%)",
        f"  per_video_limit: {per_video_limit:,} chars",
    ]
    if truncated:
        lines.append(f"  truncated: {truncated}/{n_videos} videos exceed per-video limit")
    else:
        lines.append("  truncation: none")
    return "\n".join(lines)


def run_stage_synthesis(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    *,
    run_time: datetime,
    playlist_title: str,
    model: str = "sonnet",
    agent_models: dict[str, str] | None = None,
    min_playlist_size: int = MIN_PLAYLIST_SIZE,
    max_chapters: int | None = None,
    dry_run: bool = False,
    folder_name_override: str | None = None,
    synthesis_timeout: int | None = None,
    profile: str | None = None,
) -> SynthesisStageResult:
    """Orchestrate α→β→Leader and write MOC + chapter md files.

    Parameters
    ----------
    videos:
        Per-video metadata. Must align 1:1 with `learning_md_bodies`.
        Videos whose stage 04 failed should be filtered out BEFORE
        calling this function.
    learning_md_bodies:
        Frontmatter-stripped 04 md bodies for each video.
    run_time:
        Shared datetime for the synthesis folder and all frontmatter.
    playlist_title:
        Used for the output folder name and MOC title.
    model:
        Default model used for any agent not explicitly overridden.
    agent_models:
        Optional `{"alpha", "beta", "leader", "reviewer"}` override map.
        Missing keys fall back to `model`. (`gamma` accepted for
        config backward-compat but ignored — coverage is now a Python
        set diff, no LLM.)
    synthesis_timeout:
        Per-agent timeout override in seconds. ``None`` = auto-compute
        from video count (``300 + 60 × n_videos``).
    profile:
        Agent Teams profile name (``"standard"``, ``"parallel"``,
        ``"full"``, ``"parallel+full"``) or ``"auto"`` / ``None`` to
        auto-pick from video count. See ``_select_profile``.
    """
    am = agent_models or {}
    alpha_model = am.get("alpha", model)
    beta_model = am.get("beta", model)
    leader_model = am.get("leader", model)
    reviewer_model = am.get("reviewer", model)

    timeouts = compute_synthesis_timeouts(len(videos), override=synthesis_timeout)

    if len(videos) != len(learning_md_bodies):
        return SynthesisStageResult(
            error=f"length mismatch: {len(videos)} videos vs {len(learning_md_bodies)} bodies"
        )

    if len(videos) < min_playlist_size:
        return SynthesisStageResult(
            skipped=True,
            skip_reason=f"playlist has {len(videos)} videos (< {min_playlist_size})",
        )

    try:
        resolved_profile = _select_profile(len(videos), profile)
    except ValueError as e:
        return SynthesisStageResult(error=f"invalid profile: {e}")

    vault_root = get_vault_root()
    playlist_folder_name = folder_name_override or format_playlist_folder_name(
        run_time, playlist_title
    )
    rel_path = f"{SYNTHESIS_BASE}/{playlist_folder_name}"
    safe_rel = ensure_safe_path(rel_path)
    playlist_dir = vault_root / safe_rel

    agent_results: list[AgentCallResult] = []

    try:
        if resolved_profile.uses_parallel_alpha:
            topics, alpha_results = call_alpha_batched(
                videos,
                learning_md_bodies,
                model=alpha_model,
                playlist_title=playlist_title,
                timeout=timeouts["alpha"],
            )
            agent_results.extend(alpha_results)
        else:
            topics, alpha_res = call_alpha(
                videos,
                learning_md_bodies,
                model=alpha_model,
                playlist_title=playlist_title,
                timeout=timeouts["alpha"],
            )
            agent_results.append(alpha_res)
    except SynthesisParseError as e:
        return SynthesisStageResult(
            error=f"alpha_parse_failed: {e}",
            profile=resolved_profile,
            agent_results=agent_results,
        )

    try:
        chapters, beta_res = call_beta(
            topics, model=beta_model, max_chapters=max_chapters, timeout=timeouts["beta"]
        )
    except SynthesisParseError as e:
        return SynthesisStageResult(
            topics=topics,
            agent_results=agent_results,
            error=f"beta_parse_failed: {e}",
            profile=resolved_profile,
        )
    agent_results.append(beta_res)

    coverage = compute_coverage(topics, chapters)

    # Reflexion loop: if β missed any α-extracted topics, re-run β with
    # the missing IDs fed back as an error instruction. Each iteration
    # narrows the gap. We cap at MAX_BETA_REFLEXION_RETRIES to bound the
    # worst-case token spend. If the final iteration still has misses,
    # Leader applies the residual-miss policy documented in its system
    # prompt (insert uncovered topics into the most-related existing
    # chapter as a trailing 補足).
    for _attempt in range(MAX_BETA_REFLEXION_RETRIES):
        if not coverage.missing_topic_ids:
            break
        try:
            chapters, retry_res = call_beta(
                topics,
                model=beta_model,
                max_chapters=max_chapters,
                missing_topic_ids=coverage.missing_topic_ids,
                timeout=timeouts["beta"],
            )
        except SynthesisParseError:
            # Parse fail on retry: keep the last good chapters and let
            # Leader absorb the residual. Don't abort the stage.
            break
        agent_results.append(retry_res)
        coverage = compute_coverage(topics, chapters)

    try:
        leader_output, leader_res = call_leader(
            videos,
            learning_md_bodies,
            topics,
            chapters,
            coverage,
            model=leader_model,
            playlist_title=playlist_title,
            timeout=timeouts["leader"],
        )
    except SynthesisParseError as e:
        return SynthesisStageResult(
            topics=topics,
            chapters=chapters,
            coverage=coverage,
            agent_results=agent_results,
            error=f"leader_parse_failed: {e}",
            profile=resolved_profile,
        )
    agent_results.append(leader_res)

    # Reviewer pass (profile = full / parallel+full). One-shot: if fixes
    # are non-empty, re-render Leader once with the feedback appended.
    # We do not loop further — the quality gate is bounded to avoid
    # unbounded re-renders and keep costs predictable.
    #
    # The Reviewer is advisory: any failure (parse error, claude CLI
    # timeout, subprocess crash, transient network error) must fall back
    # to the original Leader output rather than aborting a stage that
    # has already produced valid α/β/Leader results. We catch a broad
    # Exception for that reason.
    reviewer_feedback: ReviewerFeedback | None = None
    reviewer_status: str = "skipped"
    if resolved_profile.uses_reviewer:
        reviewer_status = "failed"
        try:
            reviewer_feedback, reviewer_res = call_reviewer(
                leader_output,
                topics,
                chapters,
                coverage,
                model=reviewer_model,
                timeout=timeouts["leader"],
            )
            agent_results.append(reviewer_res)
            reviewer_status = "ok"
        except Exception:
            reviewer_feedback = None

        if reviewer_feedback and reviewer_feedback.needs_revision:
            try:
                leader_output, leader_retry_res = rerun_leader_with_feedback(
                    videos,
                    learning_md_bodies,
                    topics,
                    chapters,
                    coverage,
                    reviewer_feedback,
                    model=leader_model,
                    playlist_title=playlist_title,
                    timeout=timeouts["leader"],
                )
                agent_results.append(leader_retry_res)
            except Exception:
                # Revision re-run failed: keep the original leader output.
                pass

    if dry_run:
        return SynthesisStageResult(
            topics=topics,
            chapters=chapters,
            coverage=coverage,
            leader_output=leader_output,
            agent_results=agent_results,
            profile=resolved_profile,
            reviewer_feedback=reviewer_feedback,
        )

    # Write files
    playlist_dir.mkdir(parents=True, exist_ok=True)

    allowed_assets = extract_allowed_embeds(learning_md_bodies)

    moc_path = playlist_dir / "00_MOC.md"
    write_moc(
        leader_output.moc,
        moc_path,
        run_time=run_time,
        playlist_title=playlist_title,
        allowed_assets=allowed_assets,
    )

    chapter_paths: list[Path] = []
    for chapter_body in leader_output.chapters:
        path = write_chapter(
            chapter_body,
            playlist_dir,
            run_time=run_time,
            playlist_title=playlist_title,
            allowed_assets=allowed_assets,
        )
        chapter_paths.append(path)

    meta_dir = playlist_dir / META_SUBDIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / DUPLICATE_SCORE_FILENAME
    meta_payload: dict[str, object] = {
        "profile": resolved_profile.value,
        "topics": [
            {
                "topic_id": t.topic_id,
                "label": t.label,
                "aliases": t.aliases,
                "source_videos": t.source_videos,
                "duplication_count": t.duplication_count,
                "category": t.category,
                "summary": t.summary,
            }
            for t in topics
        ],
        "chapters": [
            {
                "index": c.index,
                "label": c.label,
                "category": c.category,
                "topic_ids": c.topic_ids,
            }
            for c in chapters
        ],
        "coverage": {
            "covered_topic_ids": coverage.covered_topic_ids,
            "missing_topic_ids": coverage.missing_topic_ids,
        },
    }
    meta_payload["reviewer_status"] = reviewer_status
    if reviewer_feedback is not None:
        meta_payload["reviewer"] = {
            "needs_revision": reviewer_feedback.needs_revision,
            "summary": reviewer_feedback.summary,
            "fixes": [
                {"target": f.target, "reason": f.reason, "patch_hint": f.patch_hint}
                for f in reviewer_feedback.fixes
            ],
        }
    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return SynthesisStageResult(
        topics=topics,
        chapters=chapters,
        coverage=coverage,
        leader_output=leader_output,
        moc_path=moc_path,
        chapter_paths=chapter_paths,
        meta_path=meta_path,
        agent_results=agent_results,
        profile=resolved_profile,
        reviewer_feedback=reviewer_feedback,
    )
