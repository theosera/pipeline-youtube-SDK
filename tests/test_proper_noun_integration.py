"""Integration tests for the per-playlist proper-noun flow wiring.

Covers the glue that ties the proper-noun sheet to glossary.json (main.py
helpers) and to the Stage 05 output (synthesis._apply_proper_nouns).
"""

from __future__ import annotations

from pathlib import Path

from pipeline_youtube import main as main_mod
from pipeline_youtube.glossary import (
    Glossary,
    GlossaryEntry,
    ProperNounRow,
    ProperNounSheet,
    VideoSection,
    load_glossary,
    load_sheet,
    write_glossary,
)
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.stages.synthesis import _apply_proper_nouns
from pipeline_youtube.synthesis.scoring import (
    LeaderOutput,
    SynthesisChapterBody,
    SynthesisMoc,
)


def _video(video_id: str = "v1") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title="T",
        url=f"https://youtu.be/{video_id}",
        duration=60,
        channel="c",
        upload_date="20260101",
        playlist_title="PL",
    )


class TestPromoteCorrectionsToGlossary:
    def test_promotes_only_user_corrected_rows(self, tmp_path: Path) -> None:
        glossary_path = tmp_path / "glossary.json"
        write_glossary(glossary_path, Glossary())
        sheet = ProperNounSheet(
            sections=(
                VideoSection(
                    "v1",
                    "T",
                    (ProperNounRow("ぐぐる", "Google"), ProperNounRow("Anthropic", "")),
                ),
            )
        )
        added = main_mod._promote_corrections_to_glossary(sheet, glossary_path)
        assert added == 1
        merged = load_glossary(glossary_path)
        assert [e.canonical for e in merged.entries] == ["Google"]
        assert merged.entries[0].aliases == ["ぐぐる"]
        # The non-corrected "Anthropic" row is not promoted.

    def test_no_corrections_leaves_file_untouched(self, tmp_path: Path) -> None:
        glossary_path = tmp_path / "glossary.json"
        write_glossary(glossary_path, Glossary(entries=(GlossaryEntry(canonical="X"),)))
        before = glossary_path.read_text(encoding="utf-8")
        sheet = ProperNounSheet(sections=(VideoSection("v1", "T", (ProperNounRow("A", ""),)),))
        assert main_mod._promote_corrections_to_glossary(sheet, glossary_path) == 0
        assert glossary_path.read_text(encoding="utf-8") == before

    def test_merges_into_existing_glossary(self, tmp_path: Path) -> None:
        glossary_path = tmp_path / "glossary.json"
        write_glossary(glossary_path, Glossary(entries=(GlossaryEntry(canonical="Existing"),)))
        sheet = ProperNounSheet(
            sections=(VideoSection("v1", "T", (ProperNounRow("ぐぐる", "Google"),)),)
        )
        main_mod._promote_corrections_to_glossary(sheet, glossary_path)
        merged = load_glossary(glossary_path)
        assert {e.canonical for e in merged.entries} == {"Existing", "Google"}


class TestUpdateProperNounSheet:
    def test_writes_confirmed_terms(self, tmp_path: Path) -> None:
        sheet_path = tmp_path / "__proper_nouns.tsv"
        results = [
            main_mod.VideoRunResult(video=_video("v1"), confirmed_terms=("Anthropic", "Claude")),
            main_mod.VideoRunResult(video=_video("v2"), confirmed_terms=()),
        ]
        main_mod._update_proper_noun_sheet(sheet_path, results)
        sheet = load_sheet(sheet_path)
        assert sheet.section_for("v1") is not None
        assert [r.system_term for r in sheet.section_for("v1").rows] == ["Anthropic", "Claude"]
        # v2 contributed nothing, so it gets no section.
        assert sheet.section_for("v2") is None

    def test_preserves_user_corrections_on_rewrite(self, tmp_path: Path) -> None:
        sheet_path = tmp_path / "__proper_nouns.tsv"
        from pipeline_youtube.glossary import write_sheet

        write_sheet(
            sheet_path,
            ProperNounSheet(
                sections=(VideoSection("v1", "T", (ProperNounRow("ぐぐる", "Google"),)),)
            ),
        )
        results = [main_mod.VideoRunResult(video=_video("v1"), confirmed_terms=("ぐぐる", "New"))]
        main_mod._update_proper_noun_sheet(sheet_path, results)
        rows = load_sheet(sheet_path).section_for("v1").rows
        assert rows[0] == ProperNounRow("ぐぐる", "Google")  # override survived
        assert rows[1] == ProperNounRow("New", "")

    def test_no_terms_does_not_create_file(self, tmp_path: Path) -> None:
        sheet_path = tmp_path / "__proper_nouns.tsv"
        results = [main_mod.VideoRunResult(video=_video("v1"), confirmed_terms=())]
        main_mod._update_proper_noun_sheet(sheet_path, results)
        assert not sheet_path.exists()

    def test_separate_shards_accumulate_not_clobber(self, tmp_path: Path) -> None:
        # Two --sub-agents shards each write their own videos to the shared
        # sheet. The re-read-under-lock merge must keep both, not overwrite.
        sheet_path = tmp_path / "__proper_nouns.tsv"
        main_mod._update_proper_noun_sheet(
            sheet_path,
            [main_mod.VideoRunResult(video=_video("v1"), confirmed_terms=("Anthropic",))],
        )
        main_mod._update_proper_noun_sheet(
            sheet_path,
            [main_mod.VideoRunResult(video=_video("v2"), confirmed_terms=("Claude",))],
        )
        sheet = load_sheet(sheet_path)
        assert sheet.section_for("v1") is not None
        assert sheet.section_for("v2") is not None


class TestApplyProperNouns:
    def test_rewrites_moc_and_chapters(self) -> None:
        glossary = Glossary(
            entries=(GlossaryEntry(canonical="Vibe Coding", aliases=["ビブコーディング"]),)
        )
        output = LeaderOutput(
            moc=SynthesisMoc(
                title="ビブコーディング入門", body_markdown="本章はビブコーディングを扱う"
            ),
            chapters=[
                SynthesisChapterBody(
                    chapter_index=1,
                    label="ビブコーディングの基礎",
                    category="core",
                    source_video_ids=["v1"],
                    body_markdown="ビブコーディングとは何か",
                )
            ],
        )
        rewritten = _apply_proper_nouns(output, glossary)
        assert rewritten.moc.title == "Vibe Coding入門"
        assert rewritten.moc.body_markdown == "本章はVibe Codingを扱う"
        assert rewritten.chapters[0].label == "Vibe Codingの基礎"
        assert rewritten.chapters[0].body_markdown == "Vibe Codingとは何か"
        # Untouched metadata is preserved.
        assert rewritten.chapters[0].source_video_ids == ["v1"]
        assert rewritten.chapters[0].category == "core"
