"""Vault writes for hands-on mode (fold → filename → frontmatter → validate).

Every note follows the Stage 05 write-safety order (``synthesis/chapter.py``
/ ``moc.py``): LLM-generated label/body are homoglyph-folded FIRST (so a
Cyrillic-obfuscated ``<sсript>`` folds to ``<script>`` and is then stripped
by ``validate_chapter_body``, not written), filenames go through
``chapter_filename`` (invisible-char strip + OS-unsafe replace + byte-safe
truncation), frontmatter uses only allowlisted ``extra`` keys, and bodies
pass ``validate_chapter_body`` with an empty asset allowlist — any embed
the model hallucinated is dropped; the step's capture clip embed is
appended deterministically afterwards, from our own generated path.

Lossless guarantees live here too: an assigned insight whose timestamp
stamp is missing from a step body gets a deterministic callout appended,
and the final summary note appends any insight the model failed to list.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..obsidian import build_frontmatter
from ..playlist import VideoMeta
from ..services.confusables import (
    fold_markdown_mixed_script_confusables,
    fold_mixed_script_confusables,
)
from ..synthesis.body_validator import validate_chapter_body
from ..synthesis.chapter import chapter_filename
from .schemas import (
    HandsonMocOutput,
    Insight,
    SegmentLabel,
    StepBody,
    StepPlan,
    fmt_hms,
)

MOC_FILENAME = "00_MOC.md"
QA_TIPS_NOTE_BASENAME = "99_QA_Tipsまとめ"
QA_TIPS_NOTE_FILENAME = f"{QA_TIPS_NOTE_BASENAME}.md"
HANDSON_META_FILENAME = "handson_meta.json"

_HANDSON_TAGS = ["memo", "youtube", "handson"]


def step_note_filename(index: int, label: str) -> str:
    """The exact filename ``write_step_note`` will produce for this step.

    Deterministic (fold + ``chapter_filename``), so the MOC prompt can be
    given the real link stems before any file exists.
    """
    return chapter_filename(index, fold_mixed_script_confusables(label))


def write_step_note(
    step_body: StepBody,
    step: StepPlan,
    assigned_insights: list[Insight],
    capture_embed: str | None,
    handson_dir: Path,
    *,
    run_time: datetime,
    video: VideoMeta,
) -> Path:
    """Write one ``NN_<label>.md`` step note and return its path.

    Body layout: a deterministic time-range link line, the capture clip
    embed (success only), then the validated LLM body. Assigned insights
    missing their ``[H:MM:SS]`` stamp are appended as deterministic
    callouts before validation (lossless weaving).
    """
    label = fold_mixed_script_confusables(step_body.label or step.label)
    body = fold_markdown_mixed_script_confusables(step_body.body_markdown)

    missing_callouts = [
        _insight_callout(insight)
        for insight in assigned_insights
        if _stamp(insight.start_sec) not in body
    ]
    if missing_callouts:
        body = body.rstrip() + "\n\n" + "\n\n".join(missing_callouts)

    validated = validate_chapter_body(body, frozenset())

    range_line = (
        f"[{fmt_hms(step.start_sec)} 〜 {fmt_hms(step.end_sec)}]"
        f"({video.watch_url}&t={step.start_sec})"
    )
    parts = [range_line]
    if capture_embed:
        parts.append(f"![[{capture_embed}]]")
    parts.append(validated.strip())

    fm = build_frontmatter(
        dt=run_time,
        title=label,
        url=f"{video.watch_url}&t={step.start_sec}",
        tags=list(_HANDSON_TAGS),
        extra={
            "video_id": video.video_id,
            "chapter": str(step.index),
            "category": "handson-step",
        },
    )
    target = handson_dir / chapter_filename(step.index, label)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fm + "\n" + "\n\n".join(parts) + "\n", encoding="utf-8")
    return target


def write_handson_moc(
    moc: HandsonMocOutput,
    target_path: Path,
    *,
    run_time: datetime,
    video: VideoMeta,
    step_link_targets: frozenset[str] | set[str],
    has_insights: bool,
) -> None:
    """Write ``00_MOC.md``, guaranteeing every generated note stays reachable.

    The MOC is the hub: a step the model forgot to list would leave a written
    note orphaned in the graph. Any missing step link — and the final-summary
    link when insights exist — is therefore appended deterministically.
    """
    title = fold_mixed_script_confusables(moc.title) if moc.title else f"{video.title} ハンズオン"
    link_targets = frozenset(step_link_targets) | {QA_TIPS_NOTE_BASENAME}
    body = fold_markdown_mixed_script_confusables(
        moc.moc_markdown, fold_wikilink_targets=link_targets
    )
    orphaned = [t for t in sorted(step_link_targets) if f"[[{t}]]" not in body]
    if orphaned:
        body = (
            body.rstrip()
            + "\n\n## 未掲載のステップ (自動追記)\n\n"
            + "\n".join(f"- [[{target}]]" for target in orphaned)
        )
    if has_insights and f"[[{QA_TIPS_NOTE_BASENAME}]]" not in body:
        body = body.rstrip() + f"\n\n- 巻末まとめ: [[{QA_TIPS_NOTE_BASENAME}]]"

    validated = validate_chapter_body(body, frozenset())
    fm = build_frontmatter(
        dt=run_time,
        title=title,
        url=video.watch_url,
        tags=[*_HANDSON_TAGS, "moc"],
        extra={"video_id": video.video_id, "category": "handson-moc"},
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(fm + "\n" + validated.strip() + "\n", encoding="utf-8")


def write_qa_tips_summary(
    moc: HandsonMocOutput,
    insights: list[Insight],
    insight_step_links: dict[str, str],
    target_path: Path,
    *,
    run_time: datetime,
    video: VideoMeta,
) -> None:
    """Write ``99_QA_Tipsまとめ.md`` with a lossless-coverage appendix.

    Any insight whose ``[H:MM:SS]`` stamp the model omitted is appended
    deterministically (with its assigned-step link when known), so the
    final summary always lists every detected Q&A/Tips item.
    """
    link_targets = frozenset(insight_step_links.values())
    body = fold_markdown_mixed_script_confusables(
        moc.summary_markdown, fold_wikilink_targets=link_targets
    )

    appendix: list[str] = []
    for insight in insights:
        if _stamp(insight.start_sec) in body:
            continue
        stamp = fmt_hms(insight.start_sec)
        kind = "Q&A" if insight.label is SegmentLabel.QA else "Tips"
        summary = _fold_one_line(insight.summary or insight.quote or "(内容未取得)")
        line = f"- [{stamp}]({video.watch_url}&t={insight.start_sec}) [{kind}] {summary}"
        link = insight_step_links.get(insight.insight_id)
        if link:
            line += f" (→ [[{link}]])"
        appendix.append(line)
    if appendix:
        body = body.rstrip() + "\n\n## 補遺 (自動追記)\n\n" + "\n".join(appendix)

    validated = validate_chapter_body(body, frozenset())
    fm = build_frontmatter(
        dt=run_time,
        title="Q&A・Tips まとめ",
        url=video.watch_url,
        tags=list(_HANDSON_TAGS),
        extra={"video_id": video.video_id, "category": "handson-summary"},
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(fm + "\n" + validated.strip() + "\n", encoding="utf-8")


def write_handson_meta(meta: dict[str, Any], meta_dir: Path) -> Path:
    """Write ``_meta/handson_meta.json`` (the ``duplicate_score.json`` analogue)."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    target = meta_dir / HANDSON_META_FILENAME
    target.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _stamp(start_sec: int) -> str:
    """Bracketed timestamp — the exact form the prompts require.

    Matching on the bracketed token (``[1:05:00]``) instead of the bare
    ``fmt_hms`` output avoids substring false positives: ``1:00:00`` occurs
    inside ``11:00:00`` and ``05:00`` inside ``1:05:00``, either of which
    would make the lossless pass believe an insight was already woven in and
    silently skip it.
    """
    return f"[{fmt_hms(start_sec)}]"


def _insight_callout(insight: Insight) -> str:
    """Deterministic callout for an insight the LLM failed to weave in."""
    stamp = fmt_hms(insight.start_sec)
    summary = _fold_one_line(insight.summary or insight.quote or "(内容未取得)")
    if insight.label is SegmentLabel.QA:
        return f"> [!question] Q&Aより [{stamp}]: {summary}"
    return f"> [!tip] Tips [{stamp}]: {summary}"


def _fold_one_line(text: str) -> str:
    """Fold homoglyphs and collapse whitespace for single-line insertion."""
    return " ".join(fold_mixed_script_confusables(text).split())
