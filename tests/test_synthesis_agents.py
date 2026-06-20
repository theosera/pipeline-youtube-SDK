"""Tests for synthesis.agents with LLM provider mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse as ClaudeResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.synthesis import agents as agents_mod
from pipeline_youtube.synthesis.agents import (
    call_alpha,
    call_alpha_batched,
    call_beta,
    call_leader,
    call_reviewer,
    compute_coverage,
    format_learning_materials,
    merge_topics,
    render_reviewer_feedback,
)
from pipeline_youtube.synthesis.scoring import (
    ChapterPlan,
    CoverageReport,
    LeaderOutput,
    ReviewerFeedback,
    ReviewerFix,
    SynthesisChapterBody,
    SynthesisMoc,
    SynthesisParseError,
    Topic,
)

_NO_CACHE = Cache(None, enabled=False)


def _video(video_id: str, title: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title=title,
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=900,
        channel="Test Channel",
        upload_date="20260415",
        playlist_title="Test Playlist",
    )


def _fake_response(text: str) -> ClaudeResponse:
    return ClaudeResponse(
        text=text,
        model="sonnet",
        input_tokens=3,
        output_tokens=500,
        cache_creation_tokens=24000,
        cache_read_tokens=15000,
        total_cost_usd=0.10,
        duration_ms=20000,
        session_id="fake",
        stop_reason="end_turn",
    )


# =====================================================
# format_learning_materials
# =====================================================


class TestFormatLearningMaterials:
    def test_delimits_by_video_header(self):
        videos = [_video("vid1", "First Video"), _video("vid2", "Second Video")]
        bodies = ["## 概念: A\n\n要点", "## 概念: B\n\n別の要点"]
        formatted = format_learning_materials(videos, bodies)
        assert "## VIDEO: vid1: First Video" in formatted
        assert "## VIDEO: vid2: Second Video" in formatted
        assert "## 概念: A" in formatted
        assert "## 概念: B" in formatted
        # Videos separated by --- delimiter
        assert "\n---\n" in formatted

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            format_learning_materials([_video("v1", "t1")], ["body 1", "body 2"])

    def test_sanitizes_control_chars(self):
        videos = [_video("v1", "title\x01with\x08control")]
        bodies = ["body\u200bwith\x0czero-width"]
        formatted = format_learning_materials(videos, bodies)
        assert "\x01" not in formatted
        assert "\x08" not in formatted
        assert "\u200b" not in formatted


# =====================================================
# call_alpha
# =====================================================


SAMPLE_ALPHA_OUTPUT = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "コンテキスト管理",
                "source_videos": ["vid1", "vid2", "vid3"],
                "duplication_count": 3,
                "category": "core",
                "summary": "コンテキストウィンドウの管理。",
                "excerpts": [],
            },
            {
                "topic_id": "t002",
                "label": "Agent Teams 構成",
                "source_videos": ["vid1", "vid2"],
                "duplication_count": 2,
                "category": "supporting",
                "summary": "複数エージェントの分業。",
            },
        ]
    },
    ensure_ascii=False,
)


class TestCallAlpha:
    def test_happy_path(self, monkeypatch):
        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_response(SAMPLE_ALPHA_OUTPUT)

        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        videos = [
            _video("vid1", "First"),
            _video("vid2", "Second"),
            _video("vid3", "Third"),
        ]
        bodies = ["body1", "body2", "body3"]

        topics, result = call_alpha(videos, bodies, playlist_title="Test Playlist", cache=_NO_CACHE)

        assert len(topics) == 2
        assert topics[0].topic_id == "t001"
        assert topics[0].category == "core"
        assert topics[1].category == "supporting"

        # System prompt is append mode
        assert "append_system_prompt" in captured
        assert (
            "TopicExtractor" in captured["append_system_prompt"]
            or "topic" in captured["append_system_prompt"].lower()
        )

        # Prompt wraps materials in untrusted_content
        prompt = captured["prompt"]
        assert "<untrusted_content>" in prompt
        assert "Test Playlist" in prompt
        assert "## VIDEO: vid1: First" in prompt

        # Usage metadata propagated
        assert result.output_tokens == 500
        assert result.cache_creation_tokens == 24000

    def test_injected_cache_is_forwarded(self, monkeypatch):
        """DI: the cache passed to call_alpha reaches invoke_llm."""
        from pipeline_youtube.services.cache import Cache

        captured: dict = {}

        def fake_invoke(**kw):
            captured.update(kw)
            return _fake_response(SAMPLE_ALPHA_OUTPUT)

        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)
        sentinel = Cache(Path("/tmp/alpha-cache-di"), enabled=False)
        call_alpha([_video("vid1", "First")], ["body1"], cache=sentinel)
        assert captured["cache"] is sentinel

    def test_parse_error_propagates(self, monkeypatch):
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: _fake_response("not valid json at all"),
        )

        with pytest.raises(SynthesisParseError):
            call_alpha([_video("v1", "t1")], ["body"], playlist_title="x", cache=_NO_CACHE)


# =====================================================
# call_beta
# =====================================================


SAMPLE_BETA_OUTPUT = json.dumps(
    {
        "chapters": [
            {
                "index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "topic_ids": ["t001"],
                "source_videos": ["vid1", "vid2", "vid3"],
                "rationale": "全動画で言及される最重要概念",
            },
            {
                "index": 2,
                "label": "Agent Teams 実装",
                "category": "supporting",
                "topic_ids": ["t002"],
                "source_videos": ["vid1", "vid2"],
                "rationale": "2 本で取り上げられる実装手法",
            },
        ]
    },
    ensure_ascii=False,
)


class TestCallBeta:
    def test_happy_path(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [
            Topic(
                topic_id="t001",
                label="コンテキスト管理",
                source_videos=["vid1", "vid2", "vid3"],
                duplication_count=3,
                category="core",
            ),
            Topic(
                topic_id="t002",
                label="Agent Teams 構成",
                source_videos=["vid1", "vid2"],
                duplication_count=2,
                category="supporting",
            ),
        ]
        chapters, result = call_beta(topics, cache=_NO_CACHE)

        assert len(chapters) == 2
        assert chapters[0].index == 1
        assert chapters[0].category == "core"
        assert chapters[1].category == "supporting"

        # Prompt includes serialized topics
        prompt = captured["prompt"]
        assert "t001" in prompt
        assert "t002" in prompt
        assert (
            "ChapterArchitect" in captured["append_system_prompt"]
            or "章" in captured["append_system_prompt"]
        )

    def test_max_chapters_injects_prompt_constraint(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [
            Topic(topic_id="t001", label="A", duplication_count=3, category="core"),
        ]
        call_beta(topics, max_chapters=5, cache=_NO_CACHE)

        prompt = captured["prompt"]
        assert "最大 5 章" in prompt

    def test_unset_max_chapters_omits_constraint(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [
            Topic(topic_id="t001", label="A", duplication_count=3, category="core"),
        ]
        call_beta(topics, cache=_NO_CACHE)

        prompt = captured["prompt"]
        assert "最大" not in prompt
        assert "追加制約" not in prompt

    def test_missing_topic_ids_appends_reflexion_block(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [Topic(topic_id="t001", label="A", duplication_count=1, category="unique")]
        call_beta(topics, missing_topic_ids=["t005", "t009"], cache=_NO_CACHE)

        prompt = captured["prompt"]
        assert "エラー: 前回の章立てに漏れがあります" in prompt
        assert "t005" in prompt
        assert "t009" in prompt
        assert "全トピックを必ずどこかの章がカバーする" in prompt

    def test_empty_missing_topic_ids_omits_reflexion(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_BETA_OUTPUT))[1],
        )

        topics = [Topic(topic_id="t001", label="A", duplication_count=1, category="unique")]
        call_beta(topics, missing_topic_ids=[], cache=_NO_CACHE)

        prompt = captured["prompt"]
        assert "エラー" not in prompt


# =====================================================
# compute_coverage (replaces call_gamma)
# =====================================================


class TestComputeCoverage:
    def test_all_covered(self):
        topics = [
            Topic(topic_id="t001", label="x", duplication_count=3, category="core"),
            Topic(topic_id="t002", label="y", duplication_count=2, category="supporting"),
        ]
        chapters = [
            ChapterPlan(
                index=1, label="ch1", category="core", topic_ids=["t001", "t002"], source_videos=[]
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == []

    def test_missing_topic(self):
        topics = [
            Topic(topic_id="t001", label="a", duplication_count=1, category="unique"),
            Topic(topic_id="t002", label="b", duplication_count=1, category="unique"),
            Topic(topic_id="t003", label="c", duplication_count=1, category="unique"),
        ]
        chapters = [
            ChapterPlan(
                index=1, label="ch1", category="unique", topic_ids=["t001"], source_videos=[]
            ),
            ChapterPlan(
                index=2, label="ch2", category="unique", topic_ids=["t002"], source_videos=[]
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == ["t003"]

    def test_chapter_references_unknown_topic_is_not_covered(self):
        """Chapter topic_ids not in α topics must not appear in covered_topic_ids."""
        topics = [
            Topic(topic_id="t001", label="x", duplication_count=1, category="unique"),
        ]
        chapters = [
            ChapterPlan(
                index=1,
                label="ch1",
                category="unique",
                topic_ids=["t001", "t999"],  # t999 is a hallucinated id
                source_videos=[],
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001"]
        assert report.missing_topic_ids == []

    def test_empty_inputs(self):
        report = compute_coverage([], [])
        assert report.covered_topic_ids == []
        assert report.missing_topic_ids == []

    def test_sorted_output(self):
        """Output lists are sorted for deterministic downstream diffs."""
        topics = [
            Topic(topic_id="t003", label="c", duplication_count=1, category="unique"),
            Topic(topic_id="t001", label="a", duplication_count=1, category="unique"),
            Topic(topic_id="t002", label="b", duplication_count=1, category="unique"),
        ]
        chapters = [
            ChapterPlan(
                index=1,
                label="ch1",
                category="unique",
                topic_ids=["t002", "t001"],
                source_videos=[],
            ),
        ]
        report = compute_coverage(topics, chapters)
        assert report.covered_topic_ids == ["t001", "t002"]
        assert report.missing_topic_ids == ["t003"]


# =====================================================
# call_leader
# =====================================================


SAMPLE_LEADER_OUTPUT = json.dumps(
    {
        "moc": {
            "title": "Test Playlist ハンズオン",
            "body_markdown": "# Test Playlist ハンズオン\n\n## 章構成\n- [[01_基礎]]",
        },
        "chapters": [
            {
                "chapter_index": 1,
                "label": "コンテキスト管理の基礎",
                "category": "core",
                "source_video_ids": ["vid1", "vid2", "vid3"],
                "body_markdown": "> [!important]\n> コア概念\n\n## 概念定義\n\n本文。",
            }
        ],
    },
    ensure_ascii=False,
)


class TestCallLeader:
    def test_happy_path(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(SAMPLE_LEADER_OUTPUT))[1],
        )

        videos = [_video("vid1", "t1"), _video("vid2", "t2"), _video("vid3", "t3")]
        bodies = ["b1", "b2", "b3"]
        topics = [Topic(topic_id="t001", label="x", duplication_count=3, category="core")]
        chapters = [
            ChapterPlan(index=1, label="ch1", category="core", topic_ids=["t001"], source_videos=[])
        ]
        coverage = CoverageReport(covered_topic_ids=["t001"], missing_topic_ids=[])

        leader_out, result = call_leader(
            videos,
            bodies,
            topics,
            chapters,
            coverage,
            playlist_title="Test Playlist",
            cache=_NO_CACHE,
        )

        assert leader_out.moc.title == "Test Playlist ハンズオン"
        assert len(leader_out.chapters) == 1
        assert leader_out.chapters[0].body_markdown.startswith("> [!important]")

        prompt = captured["prompt"]
        # Leader receives all 4 inputs: topics, chapters, coverage, materials
        assert "t001" in prompt
        assert "ch1" in prompt
        assert "## VIDEO: vid1: t1" in prompt


# =====================================================
# merge_topics
# =====================================================


class TestMergeTopics:
    def test_dedup_by_label(self):
        batch_a = [
            Topic(
                topic_id="t001",
                label="Concept A",
                source_videos=["vid1"],
                duplication_count=1,
                category="unique",
                summary="first",
            )
        ]
        batch_b = [
            Topic(
                topic_id="t001",
                label="concept a",  # same label, different case
                source_videos=["vid2"],
                duplication_count=1,
                category="unique",
            )
        ]
        merged = merge_topics([batch_a, batch_b])
        assert len(merged) == 1
        assert merged[0].source_videos == ["vid1", "vid2"]
        # Label from the first batch is preserved
        assert merged[0].label == "Concept A"

    def test_sum_duplication_count(self):
        batch_a = [
            Topic(
                topic_id="t001",
                label="X",
                source_videos=["v1", "v2"],
                duplication_count=2,
                category="supporting",
            )
        ]
        batch_b = [
            Topic(
                topic_id="t001",
                label="X",
                source_videos=["v3"],
                duplication_count=1,
                category="unique",
            )
        ]
        merged = merge_topics([batch_a, batch_b])
        assert len(merged) == 1
        # merge_topics re-derives from unique source_videos — 3 distinct = core
        assert merged[0].duplication_count == 3
        assert merged[0].category == "core"

    def test_dedup_within_single_topic_source_videos(self):
        # α sometimes repeats a video_id within a single topic entry
        # (same concept discussed at multiple timestamps). These
        # intra-topic duplicates must not inflate duplication_count.
        batch = [
            Topic(
                topic_id="t001",
                label="X",
                source_videos=["v1", "v1", "v2"],
                duplication_count=3,
                category="core",
            )
        ]
        merged = merge_topics([batch])
        assert len(merged) == 1
        assert merged[0].source_videos == ["v1", "v2"]
        # 2 distinct sources → supporting, not core
        assert merged[0].duplication_count == 2
        assert merged[0].category == "supporting"

    def test_union_source_videos_dedup(self):
        batch_a = [
            Topic(
                topic_id="t001",
                label="X",
                source_videos=["v1", "v2"],
                duplication_count=2,
                category="supporting",
            )
        ]
        batch_b = [
            Topic(
                topic_id="t001",
                label="X",
                source_videos=["v2", "v3"],
                duplication_count=2,
                category="supporting",
            )
        ]
        merged = merge_topics([batch_a, batch_b])
        assert merged[0].source_videos == ["v1", "v2", "v3"]

    def test_distinct_labels_stay_separate(self):
        batch_a = [Topic(topic_id="t001", label="A", source_videos=["v1"], duplication_count=1)]
        batch_b = [Topic(topic_id="t001", label="B", source_videos=["v2"], duplication_count=1)]
        merged = merge_topics([batch_a, batch_b])
        assert len(merged) == 2
        labels = {t.label for t in merged}
        assert labels == {"A", "B"}

    def test_topic_ids_reissued_deterministically(self):
        batch_a = [
            Topic(topic_id="tXXX", label="B", source_videos=["v1"], duplication_count=1),
            Topic(topic_id="tYYY", label="A", source_videos=["v2"], duplication_count=1),
        ]
        merged = merge_topics([batch_a])
        assert [t.topic_id for t in merged] == ["t001", "t002"]

    def test_empty_inputs(self):
        assert merge_topics([]) == []
        assert merge_topics([[], []]) == []


# =====================================================
# call_alpha_batched
# =====================================================


BATCH_OUT_1 = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "shared",
                "source_videos": ["vid1", "vid2"],
                "duplication_count": 2,
                "category": "supporting",
                "summary": "a",
            }
        ]
    },
    ensure_ascii=False,
)

BATCH_OUT_2 = json.dumps(
    {
        "topics": [
            {
                "topic_id": "t001",
                "label": "shared",
                "source_videos": ["vid3"],
                "duplication_count": 1,
                "category": "unique",
                "summary": "",
            },
            {
                "topic_id": "t002",
                "label": "unique-in-b",
                "source_videos": ["vid4"],
                "duplication_count": 1,
                "category": "unique",
                "summary": "",
            },
        ]
    },
    ensure_ascii=False,
)


class TestCallAlphaBatched:
    def test_batches_and_merges(self, monkeypatch):
        queue = [BATCH_OUT_1, BATCH_OUT_2]

        def fake_invoke(**kw):
            return _fake_response(queue.pop(0))

        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        videos = [_video(f"vid{i}", f"t{i}") for i in range(1, 5)]
        bodies = [f"body{i}" for i in range(1, 5)]
        merged, results = call_alpha_batched(
            videos,
            bodies,
            batch_size=2,
            playlist_title="Playlist",
            cache=_NO_CACHE,
        )

        # 2 batches → 2 results
        assert len(results) == 2
        # "shared" appears in both batches and merges into one topic; "unique-in-b" adds a second
        labels = {t.label for t in merged}
        assert labels == {"shared", "unique-in-b"}
        shared_topic = next(t for t in merged if t.label == "shared")
        assert shared_topic.source_videos == ["vid1", "vid2", "vid3"]
        assert shared_topic.duplication_count == 3
        assert shared_topic.category == "core"

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            call_alpha_batched([_video("v1", "t1")], ["b1", "b2"], cache=_NO_CACHE)

    def test_empty_inputs(self):
        merged, results = call_alpha_batched([], [], cache=_NO_CACHE)
        assert merged == []
        assert results == []

    def test_single_batch_failure_preserves_others(self, monkeypatch):
        # Batch 1 returns valid α JSON; batch 2 returns garbage that
        # fails to parse. The function must return the topics from
        # batch 1 rather than aborting the whole stage.
        queue = [BATCH_OUT_1, "not valid json"]

        def fake_invoke(**kw):
            return _fake_response(queue.pop(0))

        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        videos = [_video(f"vid{i}", f"t{i}") for i in range(1, 5)]
        bodies = [f"body{i}" for i in range(1, 5)]
        merged, results = call_alpha_batched(
            videos,
            bodies,
            batch_size=2,
            playlist_title="Playlist",
            cache=_NO_CACHE,
        )

        # 1 batch succeeded, so 1 AgentCallResult and the topics from
        # BATCH_OUT_1 (label "shared") survive the merge.
        assert len(results) == 1
        assert len(merged) == 1
        assert merged[0].label == "shared"

    def test_all_batches_failing_raises(self, monkeypatch):
        queue = ["bad 1", "bad 2"]

        def fake_invoke(**kw):
            return _fake_response(queue.pop(0))

        monkeypatch.setattr(agents_mod, "invoke_claude", fake_invoke)

        videos = [_video(f"vid{i}", f"t{i}") for i in range(1, 5)]
        bodies = [f"body{i}" for i in range(1, 5)]
        with pytest.raises(SynthesisParseError, match="all .* α batches failed"):
            call_alpha_batched(
                videos, bodies, batch_size=2, playlist_title="Playlist", cache=_NO_CACHE
            )


# =====================================================
# call_reviewer + render_reviewer_feedback
# =====================================================


REVIEWER_NO_FIX = json.dumps({"needs_revision": False, "fixes": []}, ensure_ascii=False)

REVIEWER_WITH_FIX = json.dumps(
    {
        "needs_revision": True,
        "summary": "missing citations in chapter 2",
        "fixes": [
            {
                "target": "chapter:2",
                "reason": "citation missing on items 1-3",
                "patch_hint": "append [[video#^MM-SS]] to each numbered item",
            }
        ],
    },
    ensure_ascii=False,
)


def _simple_leader_output() -> LeaderOutput:
    return LeaderOutput(
        moc=SynthesisMoc(title="T", body_markdown="# T"),
        chapters=[
            SynthesisChapterBody(
                chapter_index=1,
                label="ch1",
                category="core",
                source_video_ids=["vid1"],
                body_markdown="body",
            )
        ],
    )


class TestCallReviewer:
    def test_happy_path_no_revision(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: (captured.update(kw), _fake_response(REVIEWER_NO_FIX))[1],
        )
        feedback, result = call_reviewer(
            _simple_leader_output(),
            [Topic(topic_id="t001", label="x", duplication_count=3, category="core")],
            [
                ChapterPlan(
                    index=1,
                    label="ch1",
                    category="core",
                    topic_ids=["t001"],
                    source_videos=["vid1"],
                )
            ],
            CoverageReport(covered_topic_ids=["t001"], missing_topic_ids=[]),
            cache=_NO_CACHE,
        )
        assert feedback.needs_revision is False
        assert feedback.fixes == []
        # Reviewer input should include leader output JSON
        prompt = captured["prompt"]
        assert "Leader 出力" in prompt
        assert "t001" in prompt

    def test_with_fixes(self, monkeypatch):
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: _fake_response(REVIEWER_WITH_FIX),
        )
        feedback, _ = call_reviewer(
            _simple_leader_output(),
            [],
            [],
            CoverageReport(),
            cache=_NO_CACHE,
        )
        assert feedback.needs_revision is True
        assert len(feedback.fixes) == 1
        assert feedback.fixes[0].target == "chapter:2"
        assert "citation" in feedback.fixes[0].reason.lower()

    def test_malformed_json_defaults_to_no_revision(self, monkeypatch):
        monkeypatch.setattr(
            agents_mod,
            "invoke_claude",
            lambda **kw: _fake_response("not json"),
        )
        feedback, _ = call_reviewer(
            _simple_leader_output(),
            [],
            [],
            CoverageReport(),
            cache=_NO_CACHE,
        )
        # Advisory parse failure falls back to no_revision so the stage
        # continues with the original leader output.
        assert feedback.needs_revision is False


class TestRenderReviewerFeedback:
    def test_empty_fixes_renders_empty_string(self):
        assert render_reviewer_feedback(ReviewerFeedback(needs_revision=False)) == ""

    def test_renders_fix_block(self):
        fb = ReviewerFeedback(
            needs_revision=True,
            summary="needs work",
            fixes=[
                ReviewerFix(target="chapter:2", reason="missing cite", patch_hint="add ref"),
            ],
        )
        out = render_reviewer_feedback(fb)
        assert "修正指示" in out
        assert "chapter:2" in out
        assert "missing cite" in out
        assert "add ref" in out
        assert "needs work" in out
