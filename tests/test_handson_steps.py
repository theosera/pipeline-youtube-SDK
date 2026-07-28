"""Verifies handson/steps.py: step-body repair/degradation and the MOC +
final-summary generation with its deterministic fallback.
"""

from __future__ import annotations

import json

from pipeline_youtube.handson import steps as steps_mod
from pipeline_youtube.handson.schemas import (
    HandsonPlan,
    Insight,
    SegmentLabel,
    StepBody,
    StepPlan,
)
from pipeline_youtube.handson.steps import generate_moc_summary, generate_step_body
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMError, LLMResponse
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.transcript.base import TranscriptSnippet

_NO_CACHE = Cache(None, enabled=False)


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


def _step(index: int = 1, start: int = 0, end: int = 600) -> StepPlan:
    return StepPlan(index=index, label=f"ステップ{index}", start_sec=start, end_sec=end, goal="g")


def _snippets(total_sec: int = 900) -> list[TranscriptSnippet]:
    return [TranscriptSnippet(f"テキスト{i}", float(i * 30), 30.0) for i in range(total_sec // 30)]


def _response(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="opus", input_tokens=10, output_tokens=10)


_GOOD_BODY = json.dumps(
    {"label": "実装する", "body_markdown": "## ゴール\n\nx\n\n## 手順\n\n1. yを行う。"},
    ensure_ascii=False,
)


class TestGenerateStepBody:
    def test_valid_body_first_try(self, monkeypatch):
        calls = {"n": 0}

        def fake_invoke(**kw):
            calls["n"] += 1
            return _response(_GOOD_BODY)

        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)
        body, responses = generate_step_body(
            _video(),
            _step(),
            _snippets(),
            [],
            prev_label=None,
            next_label="次",
            code_bearing=False,
            model="opus",
            cache=_NO_CACHE,
        )
        assert calls["n"] == 1
        assert body.label == "実装する"
        assert "## 手順" in body.body_markdown
        assert len(responses) == 1

    def test_missing_tejun_heading_repairs_once(self, monkeypatch):
        calls = {"n": 0}

        def fake_invoke(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _response(json.dumps({"label": "a", "body_markdown": "## ゴール\nのみ"}))
            return _response(_GOOD_BODY)

        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)
        body, _ = generate_step_body(
            _video(),
            _step(),
            _snippets(),
            [],
            prev_label=None,
            next_label=None,
            code_bearing=False,
            model="opus",
            cache=_NO_CACHE,
        )
        assert calls["n"] == 2
        assert "## 手順" in body.body_markdown

    def test_two_failures_yield_degraded_body(self, monkeypatch):
        def fake_invoke(**kw):
            return _response("not json")

        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)
        body, responses = generate_step_body(
            _video(),
            _step(start=7200, end=7800),
            _snippets(9000),
            [],
            prev_label=None,
            next_label=None,
            code_bearing=False,
            model="opus",
            cache=_NO_CACHE,
        )
        assert len(responses) == 2
        assert "## 手順" in body.body_markdown
        assert "自動生成に失敗した" in body.body_markdown
        # Long-video stamp appears in H:MM:SS (7200s = 2:00:00), never MM:SS-wrapped.
        assert "2:00:00" in body.body_markdown

    def test_llm_error_yields_degraded_body(self, monkeypatch):
        def fake_invoke(**kw):
            raise LLMError("down")

        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)
        body, responses = generate_step_body(
            _video(),
            _step(),
            _snippets(),
            [],
            prev_label=None,
            next_label=None,
            code_bearing=False,
            model="opus",
            cache=_NO_CACHE,
        )
        assert responses == []
        assert "## 手順" in body.body_markdown

    def test_code_bearing_extends_system_prompt(self, monkeypatch):
        seen: dict[str, str] = {}

        def fake_invoke(**kw):
            seen["system"] = kw["system_prompt"]
            return _response(_GOOD_BODY)

        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)
        generate_step_body(
            _video(),
            _step(),
            _snippets(),
            [],
            prev_label=None,
            next_label=None,
            code_bearing=True,
            model="opus",
            cache=_NO_CACHE,
        )
        assert "コード/開発ツール系" in seen["system"]


class TestGenerateMocSummary:
    def _plan(self) -> HandsonPlan:
        return HandsonPlan(
            steps=(
                StepPlan(1, "環境構築", 0, 3600, "整える", ("q001",)),
                StepPlan(2, "実装", 3900, 9000, "動かす"),
            ),
            unassigned_insight_ids=("p001",),
        )

    def _insights(self) -> list[Insight]:
        return [
            Insight("q001", SegmentLabel.QA, 3600, 3900, "失敗談", ""),
            Insight("p001", SegmentLabel.TIPS, 7500, 7620, "小ネタ", ""),
        ]

    def _bodies(self) -> list[StepBody]:
        return [StepBody(1, "環境構築", "b1"), StepBody(2, "実装", "b2")]

    def _links(self) -> dict[int, str]:
        return {1: "01_環境構築", 2: "02_実装"}

    def test_valid_response(self, monkeypatch):
        payload = json.dumps(
            {
                "title": "T",
                "moc_markdown": "# T\n- [[01_環境構築]]",
                "summary_markdown": "## Q&A から\n- [1:00:00] 失敗談",
            },
            ensure_ascii=False,
        )
        monkeypatch.setattr(steps_mod, "invoke_claude", lambda **kw: _response(payload))
        out, responses = generate_moc_summary(
            _video(),
            self._plan(),
            self._insights(),
            self._bodies(),
            self._links(),
            model="opus",
            cache=_NO_CACHE,
        )
        assert out.title == "T"
        assert len(responses) == 1

    def test_parse_failure_builds_deterministic_fallback(self, monkeypatch):
        monkeypatch.setattr(steps_mod, "invoke_claude", lambda **kw: _response("garbage"))
        out, _ = generate_moc_summary(
            _video(),
            self._plan(),
            self._insights(),
            self._bodies(),
            self._links(),
            model="opus",
            cache=_NO_CACHE,
        )
        assert "[[01_環境構築]]" in out.moc_markdown
        assert "[[02_実装]]" in out.moc_markdown
        assert "[[99_QA_Tipsまとめ]]" in out.moc_markdown
        # Assigned insight links its step; unassigned lands in the extra section.
        assert "1:00:00" in out.summary_markdown  # q001 at 3600s
        assert "2:05:00" in out.summary_markdown  # p001 at 7500s (H:MM:SS)
        assert "本編に紐づかない知見" in out.summary_markdown

    def test_llm_error_builds_deterministic_fallback(self, monkeypatch):
        def fake_invoke(**kw):
            raise LLMError("down")

        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)
        out, responses = generate_moc_summary(
            _video(),
            self._plan(),
            self._insights(),
            self._bodies(),
            self._links(),
            model="opus",
            cache=_NO_CACHE,
        )
        assert responses == []
        assert out.moc_markdown.startswith("# ")
