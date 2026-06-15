"""Deterministic proper-noun fidelity scan (no LLM).

The fidelity perspective answers one question: *does a Stage 04 learning
note still spell a proper noun the wrong way?* (the
``"ビブコーディング"`` instead of ``"Vibe Coding"`` problem). Because the
glossary (Phase A) is an exact oracle of known variant spellings, this
check is a pure, reproducible Python scan — kept out of the LLM path for
the same reason ``synthesis.agents.compute_coverage`` is. An LLM
fidelity evaluator (for *novel* mis-transcriptions not yet in the
glossary) can be layered on later; this deterministic core is what makes
the Phase B verification gate reproducible.

Each match produces a ``high`` / ``target_scope="04"`` ``Finding`` keyed
to the offending video. The evaluation stage is ADVISORY (it records
findings in a report; it does not regenerate 04/05), so the scope merely
tells a human where the defect lives. Matching reuses the Stage 02
rewriter's compiled pattern (``glossary.text.compile_variant_pattern``):
case-insensitive with ASCII word-boundary guards, resolved to canonical
via ``Normalizer``. It is therefore **width-literal** — half/full-width
divergent forms are not matched (the same limitation documented in
``glossary.text``). An alias whose folded form equals its own canonical
is excluded upstream so a redundantly-listed canonical never self-flags.
"""

from __future__ import annotations

from ..glossary.normalizer import Normalizer
from ..glossary.schema import Glossary
from ..glossary.text import compile_variant_pattern, variant_surfaces
from ..playlist import VideoMeta
from .schemas import EvaluatorReport, Finding

# Per-video cap: a single note matching dozens of entries should not flood
# the report. Beyond this, the note has a systemic transcription problem
# that one finding-per-variant would only obscure.
_DEFAULT_MAX_FINDINGS_PER_VIDEO = 20


def scan_fidelity(
    videos: list[VideoMeta],
    learning_md_bodies: list[str],
    glossary: Glossary,
    *,
    max_findings_per_video: int = _DEFAULT_MAX_FINDINGS_PER_VIDEO,
) -> EvaluatorReport:
    """Scan each 04 body for known mis-transcribed proper nouns.

    ``videos`` and ``learning_md_bodies`` MUST be index-aligned (the
    same 1:1 contract ``run_stage_synthesis`` enforces); a length
    mismatch is a caller bug and raises ``ValueError`` rather than
    silently scanning a misaligned pair.

    Returns an ``EvaluatorReport`` with ``perspective="fidelity"``. One
    ``high`` finding is emitted per (video, glossary entry) whose variant
    spelling appears in that video's body, capped at
    ``max_findings_per_video`` per note. An empty ``findings`` list means
    no known mis-transcription was found.
    """
    if len(videos) != len(learning_md_bodies):
        raise ValueError(
            f"length mismatch: {len(videos)} videos vs {len(learning_md_bodies)} bodies"
        )

    # Use the SAME matcher as the Stage 02 rewriter (word-boundary-guarded,
    # IGNORECASE) so detection and rewriting agree on what is a variant.
    pattern = compile_variant_pattern(variant_surfaces(glossary))
    normalizer = Normalizer(glossary)  # resolve match → canonical (+ conflict check)

    findings: list[Finding] = []
    counter = 0
    for video, body in zip(videos, learning_md_bodies, strict=True):
        if pattern is None:
            continue
        # Group matched surfaces by canonical, in first-seen (text) order.
        matched_by_canonical: dict[str, list[str]] = {}
        for match in pattern.finditer(body):
            surface = match.group(0)
            canonical = normalizer.canonical_for(surface)
            if canonical is None:  # defensive: pattern built from known surfaces
                continue
            seen = matched_by_canonical.setdefault(canonical, [])
            if surface not in seen:
                seen.append(surface)
        for per_video, (canonical, surfaces_hit) in enumerate(matched_by_canonical.items()):
            if per_video >= max_findings_per_video:
                break
            counter += 1
            findings.append(_make_finding(counter, video, canonical, surfaces_hit))

    summary = (
        f"{len(findings)} 件の固有名詞誤変換を {len(videos)} 動画から検出"
        if findings
        else "既知の固有名詞誤変換は検出されませんでした"
    )
    return EvaluatorReport(perspective="fidelity", findings=findings, summary=summary)


def _make_finding(
    counter: int,
    video: VideoMeta,
    canonical: str,
    matched_variants: list[str],
) -> Finding:
    """Build one ``high`` / ``04``-scoped fidelity finding."""
    joined = "、".join(matched_variants)
    return Finding(
        finding_id=f"f{counter:03d}",
        perspective="fidelity",
        severity="high",
        target_scope="04",
        target_video_id=video.video_id,
        description=(
            f"動画「{video.title}」の学習素材に固有名詞の誤変換 "
            f"{joined} が含まれます (正規表記「{canonical}」)。"
        ),
        suggested_fix=(
            f"Stage 04 再生成時に {joined} を「{canonical}」へ正規化する (glossary 照合)。"
        ),
    )


__all__ = ["scan_fidelity"]
