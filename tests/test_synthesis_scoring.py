"""Tests for synthesis.scoring (pure functions, no claude calls)."""

from __future__ import annotations

import json

import pytest

from pipeline_youtube.synthesis.scoring import (
    LeaderOutput,
    ReviewerFeedback,
    SynthesisParseError,
    derive_category,
    extract_json,
    parse_alpha_topics,
    parse_beta_chapters,
    parse_leader_output,
    parse_reviewer_output,
)


class TestDeriveCategory:
    def test_core_three_or_more(self):
        assert derive_category(3) == "core"
        assert derive_category(5) == "core"

    def test_supporting_exactly_two(self):
        assert derive_category(2) == "supporting"

    def test_unique_one_or_zero(self):
        assert derive_category(1) == "unique"
        assert derive_category(0) == "unique"


class TestExtractJson:
    def test_strict_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_json_with_prose_prefix(self):
        raw = 'ここに JSON を返します:\n{"a": 1, "b": [2, 3]}'
        assert extract_json(raw) == {"a": 1, "b": [2, 3]}

    def test_json_with_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert extract_json(raw) == {"a": 1}

    def test_json_with_trailing_prose(self):
        raw = '{"a": 1}\n\n以上です。'
        assert extract_json(raw) == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(SynthesisParseError, match="empty"):
            extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(SynthesisParseError, match="no JSON"):
            extract_json("just prose, no object at all")


SAMPLE_ALPHA_JSON = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "コンテキスト不安",
                "aliases": ["context anxiety"],
                "source_videos": ["vid1", "vid2", "vid3"],
                "duplication_count": 3,
                "category": "core",
                "summary": "AI が焦ってタスクを強引にまとめる現象。",
                "excerpts": [
                    {"video_id": "vid1", "range": "[01:56 ~ 03:32]", "quote": "..."},
                ],
            },
            {
                "topic_id": "t002",
                "label": "GAN 方式",
                "source_videos": ["vid1", "vid4"],
                "duplication_count": 2,
                "category": "supporting",
                "summary": "生成と評価を分離する。",
            },
            {
                "topic_id": "t003",
                "label": "個別実験",
                "source_videos": ["vid4"],
                "duplication_count": 1,
                "category": "unique",
                "summary": "特定動画のみの話題。",
            },
        ]
    },
    ensure_ascii=False,
)


class TestParseAlphaTopics:
    def test_parses_three_topics(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        assert len(topics) == 3

    def test_core_topic_fields(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        t = topics[0]
        assert t.topic_id == "t001"
        assert t.label == "コンテキスト不安"
        assert t.aliases == ["context anxiety"]
        assert t.source_videos == ["vid1", "vid2", "vid3"]
        assert t.duplication_count == 3
        assert t.category == "core"
        assert len(t.excerpts) == 1
        assert t.excerpts[0].range_str == "[01:56 ~ 03:32]"

    def test_supporting_topic(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        t = topics[1]
        assert t.category == "supporting"
        assert t.duplication_count == 2

    def test_unique_topic(self):
        topics = parse_alpha_topics(SAMPLE_ALPHA_JSON)
        t = topics[2]
        assert t.category == "unique"

    def test_missing_category_derived_from_count(self):
        raw = json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "t001",
                        "label": "no category",
                        "source_videos": ["v1", "v2"],
                        # category omitted
                    }
                ]
            }
        )
        topics = parse_alpha_topics(raw)
        assert topics[0].category == "supporting"
        assert topics[0].duplication_count == 2

    def test_invalid_category_falls_back(self):
        raw = json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "t001",
                        "label": "bad cat",
                        "category": "nonsense",
                        "source_videos": ["v1"],
                        "duplication_count": 1,
                    }
                ]
            }
        )
        topics = parse_alpha_topics(raw)
        assert topics[0].category == "unique"

    def test_malformed_topics_field(self):
        raw = json.dumps({"topics": "not a list"})
        with pytest.raises(SynthesisParseError, match="topics must be a list"):
            parse_alpha_topics(raw)

    def test_empty_topics_list(self):
        assert parse_alpha_topics('{"topics": []}') == []


class TestParseBetaChapters:
    def test_parses_chapters(self):
        raw = json.dumps(
            {
                "chapters": [
                    {
                        "index": 1,
                        "label": "Chapter One",
                        "category": "core",
                        "topic_ids": ["t001", "t002"],
                        "source_videos": ["vid1"],
                        "rationale": "because",
                    }
                ]
            }
        )
        chapters = parse_beta_chapters(raw)
        assert len(chapters) == 1
        c = chapters[0]
        assert c.index == 1
        assert c.label == "Chapter One"
        assert c.category == "core"
        assert c.topic_ids == ["t001", "t002"]
        assert c.rationale == "because"

    def test_index_defaults_to_position(self):
        raw = json.dumps({"chapters": [{"label": "A"}, {"label": "B"}]})
        chapters = parse_beta_chapters(raw)
        assert chapters[0].index == 1
        assert chapters[1].index == 2

    def test_invalid_category_falls_back_to_unique(self):
        raw = json.dumps({"chapters": [{"label": "x", "category": "bogus"}]})
        chapters = parse_beta_chapters(raw)
        assert chapters[0].category == "unique"


class TestParseLeaderOutput:
    def test_parses_moc_and_chapters(self):
        raw = json.dumps(
            {
                "moc": {
                    "title": "Test Playlist ハンズオン",
                    "body_markdown": "# Test Playlist ハンズオン\n\n## 章構成",
                },
                "chapters": [
                    {
                        "chapter_index": 1,
                        "label": "基礎概念",
                        "category": "core",
                        "source_video_ids": ["vid1", "vid2"],
                        "body_markdown": "> [!important]\n## 概念定義\n\n...",
                    },
                    {
                        "chapter_index": 2,
                        "label": "応用編",
                        "category": "supporting",
                        "source_video_ids": ["vid3"],
                        "body_markdown": "## 応用\n\n...",
                    },
                ],
            },
            ensure_ascii=False,
        )
        out = parse_leader_output(raw)
        assert isinstance(out, LeaderOutput)
        assert out.moc.title == "Test Playlist ハンズオン"
        assert len(out.chapters) == 2
        assert out.chapters[0].category == "core"
        assert out.chapters[0].body_markdown.startswith("> [!important]")

    def test_malformed_moc_raises(self):
        raw = json.dumps({"moc": "not a dict", "chapters": []})
        with pytest.raises(SynthesisParseError, match="moc"):
            parse_leader_output(raw)

    def test_malformed_chapters_raises(self):
        raw = json.dumps({"moc": {"title": "x", "body_markdown": "y"}, "chapters": "bogus"})
        with pytest.raises(SynthesisParseError, match="chapters"):
            parse_leader_output(raw)


class TestParseReviewerOutput:
    def test_no_revision_needed(self):
        raw = json.dumps({"needs_revision": False, "fixes": []})
        out = parse_reviewer_output(raw)
        assert isinstance(out, ReviewerFeedback)
        assert out.needs_revision is False
        assert out.fixes == []

    def test_with_fixes(self):
        raw = json.dumps(
            {
                "needs_revision": True,
                "summary": "missing cites",
                "fixes": [
                    {
                        "target": "chapter:2",
                        "reason": "citation missing",
                        "patch_hint": "add ref",
                    }
                ],
            }
        )
        out = parse_reviewer_output(raw)
        assert out.needs_revision is True
        assert out.summary == "missing cites"
        assert len(out.fixes) == 1
        assert out.fixes[0].target == "chapter:2"

    def test_needs_revision_requires_fixes_list(self):
        # needs_revision=True without fixes collapses to False to avoid
        # triggering an unnecessary re-render.
        raw = json.dumps({"needs_revision": True, "fixes": []})
        out = parse_reviewer_output(raw)
        assert out.needs_revision is False

    def test_malformed_json_returns_no_revision(self):
        out = parse_reviewer_output("garbage")
        assert out.needs_revision is False
        assert out.fixes == []

    def test_non_dict_top_level_returns_no_revision(self):
        # Valid JSON but a list at the top level — the docstring promises
        # a safe default rather than an AttributeError on ``.get``.
        out = parse_reviewer_output('[{"target": "moc"}]')
        assert out.needs_revision is False
        assert out.fixes == []
        assert out.summary == ""

    def test_scalar_top_level_returns_no_revision(self):
        out = parse_reviewer_output('"just a string"')
        assert out.needs_revision is False

    def test_missing_fields_defaults(self):
        raw = json.dumps({})
        out = parse_reviewer_output(raw)
        assert out.needs_revision is False
        assert out.fixes == []
        assert out.summary == ""

    def test_ignores_non_dict_fixes(self):
        raw = json.dumps(
            {
                "needs_revision": True,
                "fixes": [
                    "not a dict",
                    {"target": "moc", "reason": "r", "patch_hint": "p"},
                ],
            }
        )
        out = parse_reviewer_output(raw)
        assert len(out.fixes) == 1
        assert out.fixes[0].target == "moc"
