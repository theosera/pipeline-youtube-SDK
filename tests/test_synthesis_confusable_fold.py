"""Regression: Stage 05 synthesis chapter / MOC writes fold homoglyphs.

Homoglyph code points are built with ``chr()`` so this test source embeds no
literal confusable glyph (mirrors ``services.confusables``' own rule). Japanese
is legitimate content and is asserted to survive the fold unchanged.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline_youtube.synthesis.chapter import write_chapter
from pipeline_youtube.synthesis.moc import write_moc
from pipeline_youtube.synthesis.scoring import SynthesisChapterBody, SynthesisMoc

CYR_E = chr(0x435)  # CYRILLIC SMALL LETTER IE (looks like Latin 'e')
CYR_ES = chr(0x441)  # CYRILLIC SMALL LETTER ES (looks like Latin 'c')
GRK_O = chr(0x3BF)  # GREEK SMALL LETTER OMICRON (looks like Latin 'o')
RUN_TIME = datetime(2026, 1, 1, 12, 0)


def test_write_chapter_folds_homoglyphs(tmp_path: Path) -> None:
    chapter = SynthesisChapterBody(
        chapter_index=1,
        label="Intro",
        category="core",
        source_video_ids=["vid1"],
        body_markdown=f"## Vib{CYR_E}\nlearn c{GRK_O}de here",
    )
    target = write_chapter(chapter, tmp_path, run_time=RUN_TIME, playlist_title="pl")
    written = target.read_text(encoding="utf-8")
    assert "Vibe" in written
    assert "code" in written
    assert CYR_E not in written
    assert GRK_O not in written


def test_write_moc_folds_homoglyphs(tmp_path: Path) -> None:
    moc = SynthesisMoc(title="Map", body_markdown=f"see c{GRK_O}de and Vib{CYR_E}")
    target = tmp_path / "00_MOC.md"
    write_moc(moc, target, run_time=RUN_TIME, playlist_title="pl")
    written = target.read_text(encoding="utf-8")
    assert "code" in written and "Vibe" in written
    assert GRK_O not in written and CYR_E not in written


def test_write_chapter_folds_label_into_filename_and_title(tmp_path: Path) -> None:
    # Leader-generated label is folded before the filename and frontmatter
    # title are built, so concealment never reaches the most visible
    # identifiers.
    chapter = SynthesisChapterBody(
        chapter_index=3,
        label=f"C{GRK_O}re Concepts",
        category="core",
        source_video_ids=["vid3"],
        body_markdown="plain body",
    )
    target = write_chapter(chapter, tmp_path, run_time=RUN_TIME, playlist_title="pl")
    assert target.name == "03_Core Concepts.md"
    written = target.read_text(encoding="utf-8")
    assert 'title: "Core Concepts"' in written
    assert GRK_O not in written


def test_write_moc_folds_title(tmp_path: Path) -> None:
    moc = SynthesisMoc(title=f"Vib{CYR_E} Handbook", body_markdown="plain body")
    target = tmp_path / "00_MOC.md"
    write_moc(moc, target, run_time=RUN_TIME, playlist_title="pl")
    written = target.read_text(encoding="utf-8")
    assert 'title: "Vibe Handbook"' in written
    assert CYR_E not in written


def test_write_chapter_preserves_external_links_and_embeds(tmp_path: Path) -> None:
    citation = f"Vib{CYR_E} Coding#^00-30"
    embed = f"2026 Vib{CYR_E}/pyt_vid1_00.webp"
    chapter = SynthesisChapterBody(
        chapter_index=4,
        label="Sources",
        category="core",
        source_video_ids=["vid1"],
        body_markdown=f"Vib{CYR_E} prose [[{citation}]] ![[{embed}]]",
    )

    target = write_chapter(
        chapter,
        tmp_path,
        run_time=RUN_TIME,
        playlist_title="pl",
        allowed_assets={embed},
    )
    written = target.read_text(encoding="utf-8")

    assert "Vibe prose" in written
    assert f"[[{citation}]]" in written
    assert f"![[{embed}]]" in written
    assert "dropped embed" not in written


def test_write_moc_only_folds_generated_chapter_link_targets(tmp_path: Path) -> None:
    chapter_target = f"01_C{GRK_O}re"
    source_target = f"Vib{CYR_E} Coding"
    moc = SynthesisMoc(
        title="Map",
        body_markdown=f"[[{chapter_target}]] [[{source_target}]]",
    )
    target = tmp_path / "00_MOC.md"

    write_moc(
        moc,
        target,
        run_time=RUN_TIME,
        playlist_title="pl",
        generated_chapter_link_targets={chapter_target},
    )
    written = target.read_text(encoding="utf-8")

    assert "[[01_Core]]" in written
    assert f"[[{source_target}]]" in written


def test_write_chapter_folds_before_html_strip(tmp_path: Path) -> None:
    # Security regression: a Cyrillic-obfuscated `<sсript>` must be folded to
    # `<script>` and then stripped by validate_chapter_body, NOT written as an
    # active tag. Guards against folding after sanitization.
    chapter = SynthesisChapterBody(
        chapter_index=4,
        label="Sec",
        category="core",
        source_video_ids=["vid4"],
        body_markdown=f"intro\n<s{CYR_ES}ript>alert(1)</s{CYR_ES}ript>\nmore",
    )
    target = write_chapter(chapter, tmp_path, run_time=RUN_TIME, playlist_title="pl")
    written = target.read_text(encoding="utf-8")
    assert "<script>" not in written
    assert CYR_ES not in written


def test_write_moc_folds_before_html_strip(tmp_path: Path) -> None:
    moc = SynthesisMoc(title="Sec", body_markdown=f"<s{CYR_ES}ript>alert(1)</s{CYR_ES}ript>")
    target = tmp_path / "00_MOC.md"
    write_moc(moc, target, run_time=RUN_TIME, playlist_title="pl")
    written = target.read_text(encoding="utf-8")
    assert "<script>" not in written
    assert CYR_ES not in written


def test_write_chapter_leaves_japanese_intact(tmp_path: Path) -> None:
    # THE regression guard: legitimate Japanese content must never be folded.
    body = "## 設計\nAnthropicが公開したハーネス設計"
    chapter = SynthesisChapterBody(
        chapter_index=2,
        label="設計",
        category="core",
        source_video_ids=["vid2"],
        body_markdown=body,
    )
    target = write_chapter(chapter, tmp_path, run_time=RUN_TIME, playlist_title="pl")
    written = target.read_text(encoding="utf-8")
    assert "Anthropicが公開したハーネス設計" in written
