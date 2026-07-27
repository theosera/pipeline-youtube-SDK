"""Verifies handson/writer.py: the fold → filename → frontmatter → validate
write order, lossless insight/coverage appendices, hostile-label filename
hardening, hallucinated-embed stripping, and H:MM:SS range rendering.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pipeline_youtube.handson.schemas import (
    HandsonMocOutput,
    Insight,
    SegmentLabel,
    StepBody,
    StepPlan,
)
from pipeline_youtube.handson.writer import (
    HANDSON_META_FILENAME,
    QA_TIPS_NOTE_FILENAME,
    step_note_filename,
    write_handson_meta,
    write_handson_moc,
    write_qa_tips_summary,
    write_step_note,
)
from pipeline_youtube.playlist import VideoMeta

_RUN_TIME = datetime(2026, 7, 27, 12, 0)


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="vid001",
        title="Long Talk",
        url="https://www.youtube.com/watch?v=vid001",
        duration=9000,
        channel="Test",
        upload_date="20260601",
        playlist_title=None,
    )


def _step(index: int = 1, start: int = 7200, end: int = 9000) -> StepPlan:
    return StepPlan(index=index, label="実装", start_sec=start, end_sec=end, goal="g")


def _qa_insight(start: int = 3600) -> Insight:
    return Insight("q001", SegmentLabel.QA, start, start + 300, "失敗談の共有", "")


@pytest.fixture
def handson_dir(tmp_path: Path) -> Path:
    return tmp_path / "05_Synthesis" / "2026-07-27-1200 Long Talk"


class TestWriteStepNote:
    def test_frontmatter_and_range_line(self, handson_dir: Path):
        body = StepBody(1, "実装", "## ゴール\n\nx\n\n## 手順\n\n1. y")
        path = write_step_note(
            body, _step(), [], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        text = path.read_text(encoding="utf-8")
        assert path.name == "01_実装.md"
        assert 'video_id: "vid001"' in text
        assert 'chapter: "1"' in text
        assert 'category: "handson-step"' in text
        assert "tags: [memo, youtube, handson]" in text
        # Range line renders H:MM:SS past the 99:59 ceiling, with a t= link.
        assert "[2:00:00 〜 2:30:00]" in text
        assert "&t=7200" in text

    def test_mixed_script_label_is_folded_everywhere(self, handson_dir: Path):
        # "Сlaude" starts with CYRILLIC ES — must fold to Latin "Claude".
        body = StepBody(1, "Сlaude 入門", "## ゴール\n\nx\n\n## 手順\n\n1. y")
        path = write_step_note(
            body, _step(), [], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        assert "Claude" in path.name
        assert "Сlaude" not in path.name
        text = path.read_text(encoding="utf-8")
        assert "Сlaude" not in text

    def test_obfuscated_script_tag_folds_then_strips(self, handson_dir: Path):
        # Cyrillic-obfuscated <sсript> folds to <script>, which the body
        # validator then strips — the fold-before-validate order matters.
        body = StepBody(1, "実装", "## ゴール\n\nx\n\n## 手順\n\n1. y\n\n<sсript>alert(1)</sсript>")
        path = write_step_note(
            body, _step(), [], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        text = path.read_text(encoding="utf-8")
        assert "<script" not in text
        assert "<sсript" not in text

    def test_hallucinated_embed_is_dropped_but_ours_survives(self, handson_dir: Path):
        body = StepBody(1, "実装", "## ゴール\n\n![[evil.png]]\n\n## 手順\n\n1. y")
        path = write_step_note(
            body,
            _step(),
            [],
            "2026-07-27-1200 Long Talk/pyt_vid001_h00.webp",
            handson_dir,
            run_time=_RUN_TIME,
            video=_video(),
        )
        text = path.read_text(encoding="utf-8")
        assert "dropped embed" in text
        assert "![[evil.png]]" not in text
        assert "![[2026-07-27-1200 Long Talk/pyt_vid001_h00.webp]]" in text

    def test_missing_insight_stamp_appends_callout(self, handson_dir: Path):
        body = StepBody(1, "実装", "## ゴール\n\nx\n\n## 手順\n\n1. y")
        path = write_step_note(
            body, _step(), [_qa_insight()], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        text = path.read_text(encoding="utf-8")
        assert "> [!question] Q&Aより [1:00:00]: 失敗談の共有" in text

    def test_present_insight_stamp_is_not_duplicated(self, handson_dir: Path):
        body = StepBody(
            1,
            "実装",
            "## ゴール\n\nx\n\n## 手順\n\n1. y\n\n> [!question] Q&Aより [1:00:00]: 失敗談の共有",
        )
        path = write_step_note(
            body, _step(), [_qa_insight()], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        assert path.read_text(encoding="utf-8").count("[1:00:00]") == 1

    def test_similar_timestamp_elsewhere_still_appends_callout(self, handson_dir: Path):
        # Regression: a bare-substring check let "1:00:00" match inside
        # "11:00:00", so the lossless pass skipped a genuinely missing insight.
        body = StepBody(1, "実装", "## ゴール\n\n11:00:00 に開始\n\n## 手順\n\n1. y")
        path = write_step_note(
            body, _step(), [_qa_insight()], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        assert "> [!question] Q&Aより [1:00:00]: 失敗談の共有" in path.read_text(encoding="utf-8")

    def test_hostile_label_cannot_escape_the_folder(self, handson_dir: Path):
        body = StepBody(1, "../../etc/passwd‮", "## ゴール\n\nx\n\n## 手順\n\n1. y")
        path = write_step_note(
            body, _step(), [], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        assert path.parent == handson_dir
        assert "/" not in path.name and "\\" not in path.name
        assert "‮" not in path.name

    def test_step_note_filename_matches_written_file(self, handson_dir: Path):
        body = StepBody(3, "Сlaude 実践", "## ゴール\n\nx\n\n## 手順\n\n1. y")
        predicted = step_note_filename(3, body.label)
        path = write_step_note(
            body, _step(index=3), [], None, handson_dir, run_time=_RUN_TIME, video=_video()
        )
        assert path.name == predicted


class TestWriteMocAndSummary:
    def test_moc_appends_summary_link_when_missing(self, handson_dir: Path):
        moc = HandsonMocOutput(title="T", moc_markdown="# T\n- [[01_実装]]", summary_markdown="")
        target = handson_dir / "00_MOC.md"
        write_handson_moc(
            moc,
            target,
            run_time=_RUN_TIME,
            video=_video(),
            step_link_targets={"01_実装"},
            has_insights=True,
        )
        text = target.read_text(encoding="utf-8")
        assert "[[99_QA_Tipsまとめ]]" in text
        assert 'category: "handson-moc"' in text

    def test_moc_appends_step_links_the_model_omitted(self, handson_dir: Path):
        # Regression: a step the model forgot to list left its written note
        # orphaned in the graph (the MOC is the only hub).
        moc = HandsonMocOutput(title="T", moc_markdown="# T\n- [[01_実装]]", summary_markdown="")
        target = handson_dir / "00_MOC.md"
        write_handson_moc(
            moc,
            target,
            run_time=_RUN_TIME,
            video=_video(),
            step_link_targets={"01_実装", "02_検証"},
            has_insights=False,
        )
        text = target.read_text(encoding="utf-8")
        assert "## 未掲載のステップ (自動追記)" in text
        assert "[[02_検証]]" in text

    def test_moc_listing_every_step_needs_no_appendix(self, handson_dir: Path):
        moc = HandsonMocOutput(
            title="T", moc_markdown="# T\n- [[01_実装]]\n- [[02_検証]]", summary_markdown=""
        )
        target = handson_dir / "00_MOC.md"
        write_handson_moc(
            moc,
            target,
            run_time=_RUN_TIME,
            video=_video(),
            step_link_targets={"01_実装", "02_検証"},
            has_insights=False,
        )
        assert "未掲載のステップ" not in target.read_text(encoding="utf-8")

    def test_moc_without_insights_does_not_force_link(self, handson_dir: Path):
        moc = HandsonMocOutput(title="T", moc_markdown="# T", summary_markdown="")
        target = handson_dir / "00_MOC.md"
        write_handson_moc(
            moc,
            target,
            run_time=_RUN_TIME,
            video=_video(),
            step_link_targets=set(),
            has_insights=False,
        )
        assert "99_QA_Tipsまとめ" not in target.read_text(encoding="utf-8")

    def test_summary_appends_missing_insights(self, handson_dir: Path):
        # The model listed q001 but forgot p001 → appendix restores it.
        moc = HandsonMocOutput(
            title="T",
            moc_markdown="# T",
            summary_markdown="## Q&A から\n- [1:00:00] 失敗談の共有 (→ [[01_実装]])",
        )
        insights = [
            _qa_insight(),
            Insight("p001", SegmentLabel.TIPS, 7500, 7620, "ショートカット術", ""),
        ]
        target = handson_dir / QA_TIPS_NOTE_FILENAME
        write_qa_tips_summary(
            moc,
            insights,
            {"q001": "01_実装"},
            target,
            run_time=_RUN_TIME,
            video=_video(),
        )
        text = target.read_text(encoding="utf-8")
        assert "## 補遺 (自動追記)" in text
        assert "[2:05:00]" in text
        assert "ショートカット術" in text
        assert "&t=7500" in text
        assert 'category: "handson-summary"' in text

    def test_summary_complete_listing_needs_no_appendix(self, handson_dir: Path):
        moc = HandsonMocOutput(
            title="T",
            moc_markdown="# T",
            summary_markdown="## Q&A から\n- [1:00:00] 失敗談の共有",
        )
        target = handson_dir / QA_TIPS_NOTE_FILENAME
        write_qa_tips_summary(moc, [_qa_insight()], {}, target, run_time=_RUN_TIME, video=_video())
        assert "補遺" not in target.read_text(encoding="utf-8")


class TestWriteMeta:
    def test_meta_json_round_trips(self, handson_dir: Path):
        meta = {"video_id": "vid001", "steps": [{"index": 1}]}
        path = write_handson_meta(meta, handson_dir / "_meta")
        assert path.name == HANDSON_META_FILENAME
        assert json.loads(path.read_text(encoding="utf-8")) == meta
