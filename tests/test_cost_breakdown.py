"""Tests for #6: per-stage / per-agent cost breakdown at end of CLI."""

from __future__ import annotations

from types import SimpleNamespace

from pipeline_youtube.main import VideoRunResult, _print_cost_breakdown
from pipeline_youtube.playlist import VideoMeta


def _video(video_id: str = "vid") -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        title="t",
        url=f"https://www.youtube.com/watch?v={video_id}",
        duration=60,
        channel=None,
        upload_date=None,
        playlist_title=None,
    )


def _result(video_id: str, s_cost: float, s_model: str, l_cost: float, l_model: str):
    return VideoRunResult(
        video=_video(video_id),
        learning_md_body="x",
        summary_cost_usd=s_cost,
        summary_model=s_model,
        learning_cost_usd=l_cost,
        learning_model=l_model,
    )


class TestCostBreakdown:
    def test_videos_only_sums_per_stage(self, capsys):
        results = [
            _result("a", 0.01, "haiku", 0.05, "sonnet"),
            _result("b", 0.01, "haiku", 0.05, "sonnet"),
        ]
        _print_cost_breakdown(results, synthesis_result=None)
        out = capsys.readouterr().out
        assert "stage_02" in out
        assert "haiku" in out
        assert "$  0.020" in out  # 0.01 + 0.01
        assert "stage_04" in out
        assert "$  0.100" in out  # 0.05 + 0.05
        assert "total" in out
        assert "$  0.120" in out

    def test_synthesis_adds_agents(self, capsys):
        results = [_result("a", 0.01, "haiku", 0.05, "sonnet")]
        # Profile-aware orchestration may emit a variable number of agent
        # calls (parallel α spawns N, reviewer adds one, etc.), so the
        # breakdown aggregates all Stage 05 LLM calls under a single
        # "synthesis" label instead of positional role names.
        synth = SimpleNamespace(
            agent_results=[
                SimpleNamespace(response=SimpleNamespace(model="haiku"), total_cost_usd=0.02),
                SimpleNamespace(response=SimpleNamespace(model="sonnet"), total_cost_usd=0.03),
                SimpleNamespace(response=SimpleNamespace(model="opus"), total_cost_usd=0.15),
            ]
        )
        _print_cost_breakdown(results, synthesis_result=synth)
        out = capsys.readouterr().out
        assert "synthesis" in out
        assert "gamma" not in out
        # total = 0.01 + 0.05 + 0.02 + 0.03 + 0.15 = 0.26
        assert "$  0.260" in out

    def test_empty_noop(self, capsys):
        _print_cost_breakdown([], synthesis_result=None)
        assert "Cost breakdown" not in capsys.readouterr().out

    def test_missing_cost_is_skipped(self, capsys):
        """A video with no cost info must not cause a divide or print error."""
        r = VideoRunResult(video=_video(), learning_md_body="x")  # no costs set
        _print_cost_breakdown([r], synthesis_result=None)
        assert "Cost breakdown" not in capsys.readouterr().out
