"""Verifies handson/planner.py: lossless insight assignment (retry →
auto-unassign), deterministic step repair, and the degraded fallback plan.
"""

from __future__ import annotations

import json

from pipeline_youtube.handson import planner as planner_mod
from pipeline_youtube.handson.planner import _validate_plan, plan_steps
from pipeline_youtube.handson.schemas import (
    HandsonPlan,
    Insight,
    Segment,
    SegmentLabel,
    StepPlan,
    parse_step_plan,
)
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMError, LLMResponse
from pipeline_youtube.services.cache import Cache

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


def _segments() -> list[Segment]:
    return [
        Segment(0, 3600, SegmentLabel.LECTURE, "前半"),
        Segment(3600, 3900, SegmentLabel.QA, "質疑"),
        Segment(3900, 9000, SegmentLabel.LECTURE, "後半"),
    ]


def _insights() -> list[Insight]:
    return [
        Insight("q001", SegmentLabel.QA, 3600, 3900, "失敗談", "抜粋"),
        Insight("p001", SegmentLabel.TIPS, 7500, 7620, "小ネタ", "抜粋"),
    ]


def _plan_payload(*, include: list[str], unassigned: list[str]) -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "index": 1,
                    "label": "前半を実践",
                    "start_sec": 0,
                    "end_sec": 3600,
                    "goal": "前半の内容を再現する",
                    "insight_ids": include,
                },
                {
                    "index": 2,
                    "label": "後半を実践",
                    "start_sec": 3900,
                    "end_sec": 9000,
                    "goal": "後半の内容を再現する",
                    "insight_ids": [],
                },
            ],
            "unassigned_insight_ids": unassigned,
        },
        ensure_ascii=False,
    )


def _response(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="sonnet", input_tokens=10, output_tokens=10)


class TestPlanSteps:
    def test_complete_assignment_needs_one_call(self, monkeypatch):
        calls = {"n": 0}

        def fake_invoke(**kw):
            calls["n"] += 1
            return _response(_plan_payload(include=["q001"], unassigned=["p001"]))

        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        plan, responses = plan_steps(
            _video(), _segments(), _insights(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert calls["n"] == 1
        assert plan.steps[0].insight_ids == ("q001",)
        assert plan.unassigned_insight_ids == ("p001",)
        assert len(responses) == 1

    def test_missing_id_triggers_retry_listing_it(self, monkeypatch):
        prompts: list[str] = []

        def fake_invoke(**kw):
            prompts.append(kw["prompt"])
            if len(prompts) == 1:
                # p001 is placed nowhere → coverage check must catch it.
                return _response(_plan_payload(include=["q001"], unassigned=[]))
            return _response(_plan_payload(include=["q001"], unassigned=["p001"]))

        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        plan, _ = plan_steps(
            _video(), _segments(), _insights(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert len(prompts) == 2
        assert "p001" in prompts[1]
        assert plan.unassigned_insight_ids == ("p001",)

    def test_still_missing_after_retry_is_auto_unassigned(self, monkeypatch):
        def fake_invoke(**kw):
            return _response(_plan_payload(include=["q001"], unassigned=[]))

        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        plan, responses = plan_steps(
            _video(), _segments(), _insights(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        # Lossless: p001 was never placed by the model, so it lands in
        # unassigned automatically after the single retry.
        assert len(responses) == 2
        assert plan.unassigned_insight_ids == ("p001",)

    def test_total_parse_failure_yields_fallback_plan(self, monkeypatch):
        calls = {"n": 0}

        def fake_invoke(**kw):
            calls["n"] += 1
            return _response("not json")

        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        plan, _ = plan_steps(
            _video(), _segments(), _insights(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert calls["n"] == 1  # no retry when there is no plan to fix
        # One step per lecture segment, everything unassigned (lossless).
        assert [s.label for s in plan.steps] == ["ステップ1", "ステップ2"]
        assert set(plan.unassigned_insight_ids) == {"q001", "p001"}

    def test_llm_error_yields_fallback_plan(self, monkeypatch):
        def fake_invoke(**kw):
            raise LLMError("down")

        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        plan, responses = plan_steps(
            _video(), _segments(), _insights(), duration=9000, model="sonnet", cache=_NO_CACHE
        )
        assert responses == []
        assert len(plan.steps) == 2
        assert set(plan.unassigned_insight_ids) == {"q001", "p001"}

    def test_no_lecture_segments_still_produces_a_step(self, monkeypatch):
        def fake_invoke(**kw):
            return _response("not json")

        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        qa_only = [Segment(0, 9000, SegmentLabel.QA, "全編質疑")]
        plan, _ = plan_steps(_video(), qa_only, [], duration=9000, model="sonnet", cache=_NO_CACHE)
        assert len(plan.steps) == 1
        assert (plan.steps[0].start_sec, plan.steps[0].end_sec) == (0, 9000)


class TestValidatePlan:
    def test_renumbers_by_start_and_fills_labels(self):
        raw = HandsonPlan(
            steps=(
                StepPlan(index=9, label="", start_sec=600, end_sec=900),
                StepPlan(index=1, label="A", start_sec=0, end_sec=600),
            )
        )
        plan, missing = _validate_plan(raw, set(), duration=900)
        assert [(s.index, s.start_sec) for s in plan.steps] == [(1, 0), (2, 600)]
        assert plan.steps[1].label == "ステップ2"
        assert missing == set()

    def test_clamps_and_drops_empty_ranges(self):
        raw = HandsonPlan(
            steps=(
                StepPlan(index=1, label="A", start_sec=-50, end_sec=100),
                StepPlan(index=2, label="B", start_sec=880, end_sec=2000),
                StepPlan(index=3, label="C", start_sec=950, end_sec=980),  # beyond duration
            )
        )
        plan, _ = _validate_plan(raw, set(), duration=900)
        assert [(s.start_sec, s.end_sec) for s in plan.steps] == [(0, 100), (880, 900)]

    def test_unknown_insight_ids_are_dropped(self):
        raw = parse_step_plan(
            json.dumps(
                {
                    "steps": [
                        {
                            "index": 1,
                            "label": "A",
                            "start_sec": 0,
                            "end_sec": 100,
                            "insight_ids": ["q001", "zz9"],
                        }
                    ],
                    "unassigned_insight_ids": ["p001", "hallucinated"],
                }
            )
        )
        plan, missing = _validate_plan(raw, {"q001", "p001"}, duration=100)
        assert plan.steps[0].insight_ids == ("q001",)
        assert plan.unassigned_insight_ids == ("p001",)
        assert missing == set()

    def test_duplicate_id_across_steps_keeps_only_the_first(self):
        # Regression: a repeated id used to satisfy the set-based coverage
        # check while producing the same callout in several notes and an
        # ambiguous step link in the final summary.
        raw = HandsonPlan(
            steps=(
                StepPlan(index=1, label="A", start_sec=0, end_sec=100, insight_ids=("q001",)),
                StepPlan(index=2, label="B", start_sec=100, end_sec=200, insight_ids=("q001",)),
            )
        )
        plan, missing = _validate_plan(raw, {"q001"}, duration=200)
        assert plan.steps[0].insight_ids == ("q001",)
        assert plan.steps[1].insight_ids == ()
        assert missing == set()

    def test_duplicate_id_within_one_step_is_collapsed(self):
        raw = HandsonPlan(
            steps=(
                StepPlan(
                    index=1, label="A", start_sec=0, end_sec=100, insight_ids=("q001", "q001")
                ),
            )
        )
        plan, _ = _validate_plan(raw, {"q001"}, duration=100)
        assert plan.steps[0].insight_ids == ("q001",)

    def test_assigned_wins_over_unassigned_duplicate(self):
        raw = HandsonPlan(
            steps=(StepPlan(index=1, label="A", start_sec=0, end_sec=100, insight_ids=("q001",)),),
            unassigned_insight_ids=("q001",),
        )
        plan, missing = _validate_plan(raw, {"q001"}, duration=100)
        assert plan.steps[0].insight_ids == ("q001",)
        assert plan.unassigned_insight_ids == ()
        assert missing == set()
