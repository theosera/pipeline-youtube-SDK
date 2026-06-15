"""Golden-set behavior for the deterministic fidelity scan (Phase B).

The verification gate: run the read-only detector against an existing,
un-normalized 04 body that contains ``"ビブコーディング"`` and prove it
fires a single ``high`` / ``target_scope="04"`` finding pointing at the
canonical ``"Vibe Coding"`` — using the Phase A glossary as oracle. No
LLM; fully reproducible.
"""

from __future__ import annotations

import pytest

from pipeline_youtube.evaluation.fidelity import scan_fidelity
from pipeline_youtube.glossary.schema import Glossary, GlossaryEntry
from pipeline_youtube.playlist import VideoMeta

_GLOSSARY = Glossary(
    entries=(
        GlossaryEntry(canonical="Vibe Coding", aliases=["ビブコーディング", "バイブコーディング"]),
        GlossaryEntry(canonical="Obsidian", aliases=["オブシディアン"]),
    )
)


def _video(i: int) -> VideoMeta:
    return VideoMeta(
        video_id=f"vid{i:03d}",
        title=f"Video {i}",
        url=f"https://www.youtube.com/watch?v=vid{i:03d}",
        duration=1000 + i,
        channel="Test",
        upload_date="20260415",
        playlist_title="Test Playlist",
    )


def test_detects_known_variant_as_high_04_finding() -> None:
    body = "本動画ではビブコーディングの基礎を解説します。"
    report = scan_fidelity([_video(0)], [body], _GLOSSARY)

    assert report.perspective == "fidelity"
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.severity == "high"
    assert finding.target_scope == "04"
    assert finding.target_video_id == "vid000"
    assert finding.perspective == "fidelity"
    assert "Vibe Coding" in finding.description
    assert "ビブコーディング" in finding.description
    assert "Vibe Coding" in finding.suggested_fix


def test_canonical_spelling_is_not_flagged() -> None:
    body = "本動画では Vibe Coding の基礎を解説します。"
    report = scan_fidelity([_video(0)], [body], _GLOSSARY)
    assert report.findings == []
    assert "検出されませんでした" in report.summary


def test_clean_body_yields_no_findings() -> None:
    report = scan_fidelity([_video(0)], ["まったく無関係な内容です。"], _GLOSSARY)
    assert report.findings == []


def test_multiple_variants_of_one_entry_collapse_to_one_finding() -> None:
    body = "ビブコーディングとバイブコーディングは同じものです。"
    report = scan_fidelity([_video(0)], [body], _GLOSSARY)
    assert len(report.findings) == 1
    # both matched surfaces are reported in the single finding
    assert "ビブコーディング" in report.findings[0].description
    assert "バイブコーディング" in report.findings[0].description


def test_findings_route_to_the_correct_video() -> None:
    videos = [_video(0), _video(1), _video(2)]
    bodies = [
        "クリーンな本文。",
        "ここにオブシディアンが登場する。",
        "別のクリーンな本文。",
    ]
    report = scan_fidelity(videos, bodies, _GLOSSARY)
    assert [f.target_video_id for f in report.findings] == ["vid001"]
    assert "Obsidian" in report.findings[0].description


def test_ascii_alias_uses_word_boundaries() -> None:
    # Short ASCII aliases must match whole tokens only, never inside a
    # larger word — otherwise a clean note is wrongly routed to 04 regen.
    glossary = Glossary(
        entries=(GlossaryEntry(canonical="Artificial Intelligence", aliases=["AI"]),)
    )
    # standalone token (any case) -> flagged
    assert len(scan_fidelity([_video(0)], ["He said AI loudly"], glossary).findings) == 1
    # "ai" inside "said"/"maintain" -> NOT flagged
    assert scan_fidelity([_video(0)], ["She said we maintain it"], glossary).findings == []


def test_case_or_width_only_alias_is_treated_as_canonical() -> None:
    # An alias differing from the canonical only by case/width folds to the
    # canonical key, so it is NOT a mis-transcription and is never flagged.
    glossary = Glossary(entries=(GlossaryEntry(canonical="Vibe Coding", aliases=["VIBE CODING"]),))
    report = scan_fidelity([_video(0)], ["I love ｖｉｂｅ　ｃｏｄｉｎｇ here"], glossary)
    assert report.findings == []


def test_finding_ids_are_sequential_and_deterministic() -> None:
    videos = [_video(0), _video(1)]
    bodies = ["ビブコーディング", "オブシディアン"]
    report = scan_fidelity(videos, bodies, _GLOSSARY)
    assert [f.finding_id for f in report.findings] == ["f001", "f002"]


def test_per_video_cap_limits_findings() -> None:
    entries = tuple(
        GlossaryEntry(canonical=f"Canon{i}", aliases=[f"へんかん{i}"]) for i in range(5)
    )
    body = "".join(f"へんかん{i}" for i in range(5))
    report = scan_fidelity([_video(0)], [body], Glossary(entries=entries), max_findings_per_video=2)
    assert len(report.findings) == 2


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        scan_fidelity([_video(0), _video(1)], ["only one body"], _GLOSSARY)


def test_redundant_canonical_in_aliases_is_not_flagged() -> None:
    glossary = Glossary(entries=(GlossaryEntry(canonical="Vibe Coding", aliases=["Vibe Coding"]),))
    report = scan_fidelity([_video(0)], ["We use Vibe Coding daily."], glossary)
    assert report.findings == []
