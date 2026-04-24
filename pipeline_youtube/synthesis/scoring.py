"""Data structures and JSON parsing for Stage 05 Agent Teams outputs.

Each agent role produces JSON with a well-defined shape:

- **α (TopicExtractor)** outputs `Topic[]`:
    Extracts concepts, calculates duplication scores, assigns category.
- **β (ChapterArchitect)** outputs `ChapterPlan[]`:
    Designs hand-on chapter structure from topics.
- **Leader** outputs `SynthesisOutput` with MOC + chapter bodies.

Coverage (`CoverageReport`) is computed deterministically in Python —
see `synthesis.agents.compute_coverage()`. No LLM parsing needed for
a simple set diff.

α / β / leader go through `claude -p` via providers.claude_cli. Strict
JSON parsing with a regex fallback (find the first `{...}` block) handles
occasional prose leaks around the JSON payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Category = Literal["core", "supporting", "unique"]


@dataclass(frozen=True)
class TopicExcerpt:
    video_id: str
    range_str: str  # e.g. "[01:56 ~ 03:32]"
    quote: str


@dataclass(frozen=True)
class Topic:
    topic_id: str  # e.g. "t001"
    label: str
    aliases: list[str] = field(default_factory=list)
    source_videos: list[str] = field(default_factory=list)
    duplication_count: int = 0
    category: Category = "unique"
    summary: str = ""
    excerpts: list[TopicExcerpt] = field(default_factory=list)


@dataclass(frozen=True)
class ChapterPlan:
    index: int  # 1-based
    label: str  # chapter title (used in filename)
    category: Category  # drives ordering + callout style
    topic_ids: list[str]
    source_videos: list[str]  # contributing video IDs
    rationale: str = ""


@dataclass(frozen=True)
class CoverageReport:
    covered_topic_ids: list[str] = field(default_factory=list)
    missing_topic_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SynthesisChapterBody:
    """Leader-produced body for a single chapter (no frontmatter)."""

    chapter_index: int
    label: str
    category: Category
    source_video_ids: list[str]
    body_markdown: str


@dataclass(frozen=True)
class SynthesisMoc:
    """Leader-produced Map of Content body (no frontmatter)."""

    title: str
    body_markdown: str


@dataclass(frozen=True)
class LeaderOutput:
    moc: SynthesisMoc
    chapters: list[SynthesisChapterBody]


@dataclass(frozen=True)
class ReviewerFix:
    """A single revision instruction from the Reviewer agent."""

    target: str  # "moc" or "chapter:<index>"
    reason: str
    patch_hint: str


@dataclass(frozen=True)
class ReviewerFeedback:
    """Output of the optional Reviewer (ε) agent in the `full` profile.

    The Reviewer does not rewrite the leader output directly. Instead it
    emits a list of ``fixes`` that the orchestrator forwards back to
    Leader for one re-render pass. This keeps body generation centralized
    in Leader (single source of rendering truth) while letting Reviewer
    focus on policy-level checks (citation presence, arrow-compression,
    missing-topic reconciliation).
    """

    needs_revision: bool
    fixes: list[ReviewerFix] = field(default_factory=list)
    summary: str = ""


# =====================================================
# Category derivation rule (per plan decision)
# =====================================================


def derive_category(duplication_count: int) -> Category:
    if duplication_count >= 3:
        return "core"
    if duplication_count == 2:
        return "supporting"
    return "unique"


# =====================================================
# JSON extraction (robust against prose around the payload)
# =====================================================


class SynthesisParseError(RuntimeError):
    """Raised when an agent's JSON output cannot be parsed."""


def extract_json(raw: str) -> dict[str, Any]:
    """Parse JSON from agent output, tolerating surrounding prose.

    Strategy chain:
      1. Strict ``json.loads`` on the full response.
      2. Strip markdown code fences and retry.
      3. Walk every ``{`` in the text, try ``json.JSONDecoder().raw_decode``
         at each position, and keep the **largest** successfully parsed
         object. This handles responses that contain multiple JSON blobs
         (e.g. a small metadata object followed by the real payload),
         avoiding the "Extra data" failure that a greedy regex causes.
    """
    if not raw:
        raise SynthesisParseError("empty raw response")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fence if present
    stripped = raw.strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence) :].strip()
            break
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Walk every '{' and try raw_decode; keep the largest parsed dict.
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    best_len = 0
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (end - i) > best_len:
            best = obj
            best_len = end - i

    if best is not None:
        return best

    raise SynthesisParseError(f"no JSON object found in response; first 200 chars: {raw[:200]!r}")


# =====================================================
# Typed parsers for each agent's output
# =====================================================


def parse_alpha_topics(raw: str) -> list[Topic]:
    """Parse α's topic extraction output.

    Expected schema:
        {
          "topics": [
            {
              "topic_id": "t001",
              "label": "...",
              "aliases": ["..."],
              "source_videos": ["vid1", "vid2"],
              "duplication_count": 3,
              "category": "core",
              "summary": "...",
              "excerpts": [
                {"video_id": "vid1", "range": "[01:56 ~ 03:32]", "quote": "..."}
              ]
            }
          ]
        }
    """
    data = extract_json(raw)
    topics_raw = data.get("topics") or []
    if not isinstance(topics_raw, list):
        raise SynthesisParseError(f"topics must be a list, got {type(topics_raw).__name__}")

    topics: list[Topic] = []
    for i, t in enumerate(topics_raw):
        if not isinstance(t, dict):
            continue
        excerpts_raw = t.get("excerpts") or []
        excerpts = [
            TopicExcerpt(
                video_id=str(e.get("video_id", "")),
                range_str=str(e.get("range", "")),
                quote=str(e.get("quote", "")),
            )
            for e in excerpts_raw
            if isinstance(e, dict)
        ]
        dup = int(t.get("duplication_count") or len(t.get("source_videos") or []))
        # Trust the model's category if valid, else derive from count
        raw_cat = str(t.get("category") or "").lower()
        category: Category
        if raw_cat in ("core", "supporting", "unique"):
            category = raw_cat  # type: ignore[assignment]
        else:
            category = derive_category(dup)
        topics.append(
            Topic(
                topic_id=str(t.get("topic_id") or f"t{i + 1:03d}"),
                label=str(t.get("label") or ""),
                aliases=[str(a) for a in (t.get("aliases") or []) if a],
                source_videos=[str(v) for v in (t.get("source_videos") or []) if v],
                duplication_count=dup,
                category=category,
                summary=str(t.get("summary") or ""),
                excerpts=excerpts,
            )
        )
    return topics


def parse_beta_chapters(raw: str) -> list[ChapterPlan]:
    """Parse β's chapter plan output.

    Expected schema:
        {
          "chapters": [
            {
              "index": 1,
              "label": "ハーネスエンジニアリングの基礎概念",
              "category": "core",
              "topic_ids": ["t001", "t002"],
              "source_videos": ["vid1", "vid2"],
              "rationale": "..."
            }
          ]
        }
    """
    data = extract_json(raw)
    chapters_raw = data.get("chapters") or []
    if not isinstance(chapters_raw, list):
        raise SynthesisParseError(f"chapters must be a list, got {type(chapters_raw).__name__}")

    chapters: list[ChapterPlan] = []
    for i, c in enumerate(chapters_raw):
        if not isinstance(c, dict):
            continue
        raw_cat = str(c.get("category") or "unique").lower()
        category: Category = raw_cat if raw_cat in ("core", "supporting", "unique") else "unique"  # type: ignore[assignment]
        chapters.append(
            ChapterPlan(
                index=int(c.get("index") or (i + 1)),
                label=str(c.get("label") or ""),
                category=category,
                topic_ids=[str(t) for t in (c.get("topic_ids") or []) if t],
                source_videos=[str(v) for v in (c.get("source_videos") or []) if v],
                rationale=str(c.get("rationale") or ""),
            )
        )
    return chapters


def parse_leader_output(raw: str) -> LeaderOutput:
    """Parse leader's final synthesis output.

    Expected schema:
        {
          "moc": {"title": "...", "body_markdown": "..."},
          "chapters": [
            {
              "chapter_index": 1,
              "label": "...",
              "category": "core",
              "source_video_ids": ["vid1", "vid2"],
              "body_markdown": "## ...\n..."
            }
          ]
        }
    """
    data = extract_json(raw)

    moc_raw = data.get("moc") or {}
    if not isinstance(moc_raw, dict):
        raise SynthesisParseError("moc field must be an object")
    moc = SynthesisMoc(
        title=str(moc_raw.get("title") or ""),
        body_markdown=str(moc_raw.get("body_markdown") or ""),
    )

    chapters_raw = data.get("chapters") or []
    if not isinstance(chapters_raw, list):
        raise SynthesisParseError("chapters must be a list")

    chapters: list[SynthesisChapterBody] = []
    for i, c in enumerate(chapters_raw):
        if not isinstance(c, dict):
            continue
        raw_cat = str(c.get("category") or "unique").lower()
        category: Category = raw_cat if raw_cat in ("core", "supporting", "unique") else "unique"  # type: ignore[assignment]
        chapters.append(
            SynthesisChapterBody(
                chapter_index=int(c.get("chapter_index") or (i + 1)),
                label=str(c.get("label") or ""),
                category=category,
                source_video_ids=[str(v) for v in (c.get("source_video_ids") or []) if v],
                body_markdown=str(c.get("body_markdown") or ""),
            )
        )

    return LeaderOutput(moc=moc, chapters=chapters)


def parse_reviewer_output(raw: str) -> ReviewerFeedback:
    """Parse Reviewer's JSON feedback.

    Expected schema:
        {
          "needs_revision": true,
          "summary": "...",
          "fixes": [
            {"target": "chapter:2", "reason": "...", "patch_hint": "..."}
          ]
        }

    A non-dict response, or one missing ``needs_revision``, defaults to
    "no revision needed" rather than raising — the Reviewer pass is
    advisory, not gating.
    """
    try:
        data = extract_json(raw)
    except SynthesisParseError:
        return ReviewerFeedback(needs_revision=False)
    if not isinstance(data, dict):
        # extract_json may legitimately return a list or scalar for valid
        # JSON that isn't an object (e.g. ``[{"target":"moc"}]``). The
        # docstring promises a safe default in that case rather than an
        # AttributeError on ``.get``.
        return ReviewerFeedback(needs_revision=False)

    needs = bool(data.get("needs_revision"))
    fixes_raw = data.get("fixes") or []
    fixes: list[ReviewerFix] = []
    if isinstance(fixes_raw, list):
        for f in fixes_raw:
            if not isinstance(f, dict):
                continue
            fixes.append(
                ReviewerFix(
                    target=str(f.get("target") or ""),
                    reason=str(f.get("reason") or ""),
                    patch_hint=str(f.get("patch_hint") or ""),
                )
            )
    return ReviewerFeedback(
        needs_revision=needs and bool(fixes),
        fixes=fixes,
        summary=str(data.get("summary") or ""),
    )
