"""Data structures and JSON parsing for hands-on mode LLM outputs.

Follows the local canonical idiom (``synthesis/scoring.py`` /
``evaluation/schemas.py``): ``@dataclass(frozen=True)`` + ``extract_json``
+ hand-written ``parse_*`` validators with defensive caps.

Long-video invariant: all timestamps are **integer seconds** end to end;
``fmt_hms`` renders them as ``MM:SS`` / ``H:MM:SS`` for display only. No
parser in this module reads clock strings back — that keeps hands-on mode
free of the ``MM:SS ≤ 99:59`` regex limit in summary/capture/learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..domain.errors import HandsonParseError as HandsonParseError
from ..domain.errors import SynthesisParseError
from ..synthesis.scoring import extract_json


class SegmentLabel(StrEnum):
    """Transcript segment classification for a single talk video.

    - ``LECTURE``: the main slide-based presentation flow.
    - ``QA``: a Q&A session interleaved mid-video — carries the speaker's
      trial-and-error, failures, and the story behind the idea (core theme
      material, per the mode's design requirement).
    - ``TIPS``: tips / asides spoken outside Q&A sessions.
    """

    LECTURE = "lecture"
    QA = "qa"
    TIPS = "tips"


@dataclass(frozen=True)
class Segment:
    """One contiguous classified span of the video timeline (seconds)."""

    start_sec: int
    end_sec: int
    label: SegmentLabel
    summary: str = ""

    @property
    def duration_sec(self) -> int:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class Insight:
    """A distilled QA/TIPS segment, tracked by id through step planning.

    Ids are assigned deterministically by the segmenter (``q001…`` for QA,
    ``p001…`` for TIPS, in timeline order) so the planner's assignment
    contract — every id lands in a step or in the unassigned list — can be
    verified without trusting the LLM.
    """

    insight_id: str
    label: SegmentLabel  # QA or TIPS
    start_sec: int
    end_sec: int
    summary: str
    quote: str = ""  # head of the transcript slice (context anchor)


@dataclass(frozen=True)
class StepPlan:
    """One planned hands-on step over a lecture span."""

    index: int  # 1-based, renumbered deterministically by the planner
    label: str
    start_sec: int
    end_sec: int
    goal: str = ""
    insight_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class HandsonPlan:
    """The validated step plan plus insights that fit no step."""

    steps: tuple[StepPlan, ...]
    unassigned_insight_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepBody:
    """LLM-written tutorial body for a single step (no frontmatter)."""

    index: int
    label: str
    body_markdown: str


@dataclass(frozen=True)
class HandsonMocOutput:
    """LLM-written MOC body + the final QA/Tips summary body."""

    title: str
    moc_markdown: str
    summary_markdown: str


def fmt_hms(seconds: int) -> str:
    """Render integer seconds as ``MM:SS`` (< 1h) or ``H:MM:SS`` (>= 1h).

    The hour field is unpadded and unbounded, so a 2.5-hour talk renders
    as ``2:32:05`` — unlike the legacy ``MM:SS`` renderers this never
    wraps or overflows past 99 minutes.
    """
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# =====================================================
# JSON parsing (defensive caps mirror synthesis/scoring.py)
# =====================================================

_MAX_ITEMS = 500  # segments / steps per response
_MAX_LIST = 200  # insight_ids per step / unassigned ids
_MAX_FIELD_CHARS = 4_000  # label / summary / goal / id
_MAX_BODY_CHARS = 50_000  # step body / MOC / final summary markdown


def _cap_str(value: object, limit: int = _MAX_FIELD_CHARS) -> str:
    """Coerce to str and cap length (injection / runaway-padding guard)."""
    return str(value or "")[:limit]


def _cap_strs(values: object, limit: int = _MAX_LIST) -> tuple[str, ...]:
    """Coerce to a capped tuple of non-empty strings (non-list → empty)."""
    if not isinstance(values, list):
        return ()
    return tuple(_cap_str(v) for v in values[:limit] if v)


def _extract(raw: str) -> dict[str, Any]:
    """``extract_json`` re-raised under this package's error contract."""
    try:
        return extract_json(raw)
    except SynthesisParseError as exc:
        raise HandsonParseError(str(exc)) from exc


def _coerce_int(value: object) -> int | None:
    """Best-effort int coercion for LLM-emitted seconds (bool/None → None)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, OverflowError):
            return None
    return None


def _coerce_label(value: object) -> SegmentLabel:
    """Map an LLM-emitted label to ``SegmentLabel``; unknown → LECTURE.

    LECTURE is the safe default: misfiling a Q&A span as lecture keeps its
    content inside the step material, while an invented label would drop it.
    """
    raw = str(value or "").strip().lower()
    try:
        return SegmentLabel(raw)
    except ValueError:
        return SegmentLabel.LECTURE


def parse_segments(raw: str) -> list[Segment]:
    """Parse the segment classifier's output.

    Expected schema::

        {"segments": [
          {"start_sec": 0, "end_sec": 1800, "label": "lecture", "summary": "…"}
        ]}

    Items missing usable integer bounds are dropped; ordering, overlap,
    coverage, and boundary snapping are the segmenter's normalization job,
    not this parser's.
    """
    data = _extract(raw)
    segments_raw = data.get("segments") or []
    if not isinstance(segments_raw, list):
        raise HandsonParseError(f"segments must be a list, got {type(segments_raw).__name__}")

    segments: list[Segment] = []
    for item in segments_raw[:_MAX_ITEMS]:
        if not isinstance(item, dict):
            continue
        start = _coerce_int(item.get("start_sec"))
        end = _coerce_int(item.get("end_sec"))
        if start is None or end is None:
            continue
        segments.append(
            Segment(
                start_sec=start,
                end_sec=end,
                label=_coerce_label(item.get("label")),
                summary=_cap_str(item.get("summary")),
            )
        )
    return segments


def parse_step_plan(raw: str) -> HandsonPlan:
    """Parse the planner's output (renumbering/coverage checks live in planner).

    Expected schema::

        {"steps": [
           {"index": 1, "label": "…", "start_sec": 0, "end_sec": 900,
            "goal": "…", "insight_ids": ["q001"]}
         ],
         "unassigned_insight_ids": ["p002"]}
    """
    data = _extract(raw)
    steps_raw = data.get("steps") or []
    if not isinstance(steps_raw, list):
        raise HandsonParseError(f"steps must be a list, got {type(steps_raw).__name__}")

    steps: list[StepPlan] = []
    for i, item in enumerate(steps_raw[:_MAX_ITEMS]):
        if not isinstance(item, dict):
            continue
        start = _coerce_int(item.get("start_sec"))
        end = _coerce_int(item.get("end_sec"))
        if start is None or end is None:
            continue
        steps.append(
            StepPlan(
                index=_coerce_int(item.get("index")) or (i + 1),
                label=_cap_str(item.get("label")),
                start_sec=start,
                end_sec=end,
                goal=_cap_str(item.get("goal")),
                insight_ids=_cap_strs(item.get("insight_ids")),
            )
        )
    return HandsonPlan(
        steps=tuple(steps),
        unassigned_insight_ids=_cap_strs(data.get("unassigned_insight_ids")),
    )


def parse_step_body(raw: str) -> tuple[str, str]:
    """Parse a step-writer response into ``(label, body_markdown)``.

    Expected schema: ``{"label": "…", "body_markdown": "## ゴール…"}``.
    The step index is caller-owned (the plan is the source of truth).
    """
    data = _extract(raw)
    return _cap_str(data.get("label")), _cap_str(data.get("body_markdown"), _MAX_BODY_CHARS)


def parse_moc_output(raw: str) -> HandsonMocOutput:
    """Parse the MOC + final-summary response.

    Expected schema::

        {"title": "…", "moc_markdown": "…", "summary_markdown": "…"}
    """
    data = _extract(raw)
    return HandsonMocOutput(
        title=_cap_str(data.get("title")),
        moc_markdown=_cap_str(data.get("moc_markdown"), _MAX_BODY_CHARS),
        summary_markdown=_cap_str(data.get("summary_markdown"), _MAX_BODY_CHARS),
    )
