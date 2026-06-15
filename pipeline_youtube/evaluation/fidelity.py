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
to the offending video, so the orchestrator regenerates that single
video's Stage 04 note (then re-runs Stage 05). Matching is width/case
insensitive via ``glossary.normalizer.fold_term`` and uses substring
containment — appropriate for the distinctive katakana/long-form
spellings glossaries hold; an alias whose folded form equals its own
canonical is skipped so a redundantly-listed canonical never self-flags.
"""

from __future__ import annotations

from ..glossary.normalizer import fold_term
from ..glossary.schema import Glossary
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

    # Precompute (canonical, [variant_surfaces]) once. A variant whose
    # folded form equals the canonical's folded form is not a defect.
    variant_index = _build_variant_index(glossary)

    findings: list[Finding] = []
    counter = 0
    for video, body in zip(videos, learning_md_bodies, strict=True):
        folded_body = fold_term(body)
        per_video = 0
        for canonical, variants in variant_index:
            if per_video >= max_findings_per_video:
                break
            matched = [v for v, folded_v in variants if folded_v in folded_body]
            if not matched:
                continue
            counter += 1
            per_video += 1
            findings.append(_make_finding(counter, video, canonical, matched))

    summary = (
        f"{len(findings)} 件の固有名詞誤変換を {len(videos)} 動画から検出"
        if findings
        else "既知の固有名詞誤変換は検出されませんでした"
    )
    return EvaluatorReport(perspective="fidelity", findings=findings, summary=summary)


def _build_variant_index(
    glossary: Glossary,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return ``[(canonical, [(variant_surface, folded_variant), ...]), ...]``.

    Skips any variant whose folded form is empty or equals the
    canonical's folded form (a redundantly-listed canonical is not a
    mis-transcription). Entries left without usable variants are dropped.
    """
    index: list[tuple[str, list[tuple[str, str]]]] = []
    for entry in glossary.entries:
        canonical_fold = fold_term(entry.canonical)
        variants: list[tuple[str, str]] = []
        for alias in entry.aliases:
            folded = fold_term(alias)
            if not folded or folded == canonical_fold:
                continue
            variants.append((alias, folded))
        if variants:
            index.append((entry.canonical, variants))
    return index


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
