"""Tests for the per-playlist proper-noun TSV sheet (glossary/proper_nouns.py)."""

from __future__ import annotations

from pathlib import Path

from pipeline_youtube.glossary.normalizer import Normalizer
from pipeline_youtube.glossary.proper_nouns import (
    ProperNounRow,
    ProperNounSheet,
    VideoSection,
    correction_entries,
    correction_glossary,
    known_pairs,
    load_sheet,
    parse_sheet,
    render_sheet,
    upsert_video_terms,
    write_sheet,
)


class TestProperNounRow:
    def test_resolved_prefers_user_correction(self) -> None:
        assert ProperNounRow("ぐぐる", "Google").resolved == "Google"

    def test_resolved_falls_back_to_system(self) -> None:
        assert ProperNounRow("Anthropic", "").resolved == "Anthropic"

    def test_is_user_corrected(self) -> None:
        assert ProperNounRow("ぐぐる", "Google").is_user_corrected
        assert not ProperNounRow("Anthropic", "").is_user_corrected
        # A correction that only restates the system spelling is not a real fix.
        assert not ProperNounRow("Anthropic", "Anthropic").is_user_corrected


class TestParseRender:
    def test_roundtrip(self) -> None:
        sheet = ProperNounSheet(
            sections=(
                VideoSection(
                    "vid1",
                    "Title One",
                    (ProperNounRow("Anthropic", ""), ProperNounRow("ぐぐる", "Google")),
                ),
                VideoSection("vid2", "Title Two", (ProperNounRow("Claude", ""),)),
            )
        )
        text = render_sheet(sheet)
        assert parse_sheet(text) == sheet

    def test_parse_skips_comments_and_blanks(self) -> None:
        text = (
            "# a header comment\n"
            "\n"
            "## [vid1] Hello\n"
            "Anthropic\t\n"
            "   \n"
            "# inline comment\n"
            "ぐぐる\tGoogle\n"
        )
        sheet = parse_sheet(text)
        assert len(sheet.sections) == 1
        section = sheet.sections[0]
        assert section.video_id == "vid1"
        assert section.title == "Hello"
        assert section.rows == (ProperNounRow("Anthropic", ""), ProperNounRow("ぐぐる", "Google"))

    def test_orphan_rows_before_section_are_dropped(self) -> None:
        sheet = parse_sheet("orphan\tx\n## [vid] T\nreal\t\n")
        assert len(sheet.sections) == 1
        assert sheet.sections[0].rows == (ProperNounRow("real", ""),)

    def test_row_without_tab_has_empty_correction(self) -> None:
        sheet = parse_sheet("## [v] T\nAnthropic\n")
        assert sheet.sections[0].rows == (ProperNounRow("Anthropic", ""),)

    def test_heading_without_title(self) -> None:
        sheet = parse_sheet("## [v]\nA\t\n")
        assert sheet.sections[0].video_id == "v"
        assert sheet.sections[0].title == ""


class TestUpsert:
    def test_creates_section_for_new_video(self) -> None:
        sheet = upsert_video_terms(
            ProperNounSheet(), video_id="v", title="T", terms=["Anthropic", "Claude"]
        )
        assert sheet.section_for("v") is not None
        assert [r.system_term for r in sheet.section_for("v").rows] == ["Anthropic", "Claude"]

    def test_preserves_existing_user_corrections(self) -> None:
        base = ProperNounSheet(
            sections=(VideoSection("v", "T", (ProperNounRow("ぐぐる", "Google"),)),)
        )
        # Re-run confirms the same term plus a new one; the override survives.
        merged = upsert_video_terms(base, video_id="v", title="T", terms=["ぐぐる", "Anthropic"])
        rows = merged.section_for("v").rows
        assert rows[0] == ProperNounRow("ぐぐる", "Google")
        assert rows[1] == ProperNounRow("Anthropic", "")

    def test_dedup_is_fold_insensitive(self) -> None:
        merged = upsert_video_terms(
            ProperNounSheet(),
            video_id="v",
            title="T",
            terms=["Anthropic", "ＡＮＴＨＲＯＰＩＣ", "  "],
        )
        assert len(merged.section_for("v").rows) == 1

    def test_does_not_disturb_other_sections(self) -> None:
        base = ProperNounSheet(sections=(VideoSection("a", "A", (ProperNounRow("X", ""),)),))
        merged = upsert_video_terms(base, video_id="b", title="B", terms=["Y"])
        assert [s.video_id for s in merged.sections] == ["a", "b"]


class TestDerived:
    def test_known_pairs_uses_resolved(self) -> None:
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v", "T", (ProperNounRow("ぐぐる", "Google"), ProperNounRow("A", ""))),
            )
        )
        assert known_pairs(sheet) == [("ぐぐる", "Google"), ("A", "A")]

    def test_known_pairs_dedup_across_sections(self) -> None:
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v1", "", (ProperNounRow("A", ""),)),
                VideoSection("v2", "", (ProperNounRow("A", ""),)),
            )
        )
        assert known_pairs(sheet) == [("A", "A")]

    def test_known_pairs_prefers_later_user_correction(self) -> None:
        # A later user-corrected row must win over an earlier system-only row.
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v1", "", (ProperNounRow("ぐぐる", ""),)),
                VideoSection("v2", "", (ProperNounRow("ぐぐる", "Google"),)),
            )
        )
        assert known_pairs(sheet) == [("ぐぐる", "Google")]

    def test_correction_entries_maps_correction_to_canonical(self) -> None:
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v", "T", (ProperNounRow("ぐぐる", "Google"), ProperNounRow("A", ""))),
            )
        )
        entries = correction_entries(sheet)
        assert len(entries) == 1
        assert entries[0].canonical == "Google"
        assert entries[0].aliases == ["ぐぐる"]

    def test_correction_entries_merges_aliases_by_canonical(self) -> None:
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v1", "", (ProperNounRow("ぐぐる", "Google"),)),
                VideoSection("v2", "", (ProperNounRow("グーグル", "Google"),)),
            )
        )
        entries = correction_entries(sheet)
        assert len(entries) == 1
        assert entries[0].canonical == "Google"
        assert entries[0].aliases == ["ぐぐる", "グーグル"]

    def test_correction_entries_fold_merges_collding_canonicals(self) -> None:
        # Two hand-edited corrections differ only by fold rules (case +
        # full-width). They must collapse into one entry, not fold-collide.
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v1", "", (ProperNounRow("おーぷんえーあい", "OpenAI"),)),
                VideoSection("v2", "", (ProperNounRow("オープンエーアイ", "openai"),)),
                VideoSection("v3", "", (ProperNounRow("ＯｐｅｎＡＩ", "ＯｐｅｎＡＩ"),)),
            )
        )
        entries = correction_entries(sheet)
        assert len(entries) == 1
        assert entries[0].canonical == "OpenAI"  # first spelling wins
        # The full-width row folds to the canonical key, so it is not a redundant alias.
        assert entries[0].aliases == ["おーぷんえーあい", "オープンエーアイ"]

    def test_correction_glossary_builds_without_conflict(self) -> None:
        # The lenient sheet must always yield a buildable Normalizer (Stage 05
        # never aborts on GlossaryConflictError).
        sheet = ProperNounSheet(
            sections=(
                VideoSection("v1", "", (ProperNounRow("ぐぐる", "Google"),)),
                VideoSection("v2", "", (ProperNounRow("ぐーぐる", "google"),)),
            )
        )
        glossary = correction_glossary(sheet)
        norm = Normalizer(glossary)  # would raise GlossaryConflictError if folding collided
        assert norm.normalize("ぐぐる") == "Google"


class TestLoadWrite:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        assert load_sheet(tmp_path / "nope.tsv") == ProperNounSheet()

    def test_write_then_load_roundtrip(self, tmp_path: Path) -> None:
        sheet = upsert_video_terms(
            ProperNounSheet(), video_id="v", title="Café 日本語", terms=["Anthropic"]
        )
        path = tmp_path / "__proper_nouns.tsv"
        write_sheet(path, sheet)
        assert load_sheet(path) == sheet
