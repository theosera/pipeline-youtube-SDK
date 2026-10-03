"""Exercise all four terminal title paths without network or a real vault."""

from pathlib import Path
from unittest.mock import Mock

import click
import pytest

from pipeline_youtube import handson_runner, pipeline_runner, video_processing
from pipeline_youtube.cli_config import CliConfig
from pipeline_youtube.cli_types import CliRequest, ResolvedInput, Runtime
from pipeline_youtube.execution_plan import build_plan
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.run_result import VideoRunResult
from pipeline_youtube.services.cache import Cache
from pipeline_youtube.stages.handson import HandsonStageResult


@pytest.fixture(params=["sequential", "concurrent", "checkpoint", "handson"])
def render_title(request, tmp_path: Path, monkeypatch, capsys):
    route = request.param
    cli_request = CliRequest(
        url="https://www.youtube.com/watch?v=abcdefghijk",
        dry_run=False,
        concurrency=2 if route == "concurrent" else 1,
        sub_agents=1,
        video_range=None,
        run_timestamp="2026-10-03T12:00:00",
        code_bearing_override=None,
        transcript_concurrency=None,
        llm_concurrency=None,
        download_concurrency=None,
        cache_dir=tmp_path,
        no_cache=True,
        cache_llm_synthesis=False,
        skip_synthesis=route != "handson",
        synthesis_only=False,
        folder_name=None,
        eval_loop=0,
        force_video=(),
        capture_format="auto",
        model="sonnet",
        min_playlist_size=3,
        max_chapters=None,
        config_path=None,
        stop_after_capture=False,
        resume_reviewed=False,
        capture_backend=None,
        synthesis_timeout=None,
        synthesis_profile=None,
        provider=None,
        hybrid=False,
        handson=route == "handson",
        local_media=None,
    )
    runtime = Runtime(
        cfg=CliConfig(vault_root=tmp_path, models={}, filler_words=()),
        vault_root=tmp_path,
        models={},
        filler_words=(),
        project_root=tmp_path,
        logs_dir=tmp_path,
        cache=Cache(tmp_path / "cache", enabled=False),
        capture_backend=None,
        synthesis_timeout=None,
        synthesis_profile="auto",
    )

    def render(title: str) -> str:
        videos = [
            VideoMeta(
                video_id=video_id,
                title=title,
                url=f"https://www.youtube.com/watch?v={video_id}",
                duration=60,
                channel="Test",
                upload_date=None,
                playlist_title="Test playlist",
            )
            for video_id in (
                ["abcdefghijk", "lmnopqrstuv"] if route == "concurrent" else ["abcdefghijk"]
            )
        ]
        resolved = ResolvedInput(
            videos=videos, media_map={}, playlist_title="Test playlist", code_bearing=False
        )
        completed_ids = {v.video_id for v in videos} if route == "checkpoint" else set()
        monkeypatch.setattr(
            pipeline_runner, "get_completed_video_ids", lambda *a, **k: completed_ids
        )
        checkpoint_path = Mock(return_value=tmp_path / "learning.md")
        checkpoint_body = Mock(return_value="existing learning")
        monkeypatch.setattr(pipeline_runner, "_find_existing_04_md", checkpoint_path)
        monkeypatch.setattr(pipeline_runner, "_load_existing_04_body", checkpoint_body)
        sequential_worker = Mock(side_effect=lambda video, *a, **k: VideoRunResult(video=video))
        concurrent_worker = Mock(side_effect=lambda video, *a, **k: VideoRunResult(video=video))
        handson_worker = Mock(return_value=HandsonStageResult())
        monkeypatch.setattr(pipeline_runner, "_process_video", sequential_worker)
        monkeypatch.setattr(video_processing, "_process_video", concurrent_worker)
        monkeypatch.setattr(handson_runner, "run_stage_handson", handson_worker)

        # capsys is not a TTY: explicitly retain ANSI sequences as a real color
        # terminal would. Otherwise Click could hide a missing sanitizer.
        with click.Context(click.Command("test"), color=True):
            pipeline_runner.run_pipeline(
                cli_request, runtime, resolved, build_plan(cli_request, runtime, resolved)
            )
        output = capsys.readouterr().out

        for worker, expected_route in (
            (sequential_worker, "sequential"),
            (concurrent_worker, "concurrent"),
            (handson_worker, "handson"),
        ):
            assert worker.call_count == (len(videos) if route == expected_route else 0)
            assert {call.args[0].video_id for call in worker.call_args_list} == (
                {v.video_id for v in videos} if route == expected_route else set()
            )
            for call in worker.call_args_list:
                assert any(call.args[0] is video for video in videos)
        assert checkpoint_path.call_count == (1 if route == "checkpoint" else 0)
        assert checkpoint_body.call_count == (1 if route == "checkpoint" else 0)
        assert all(v.title == title for v in videos)  # Display-only sanitization.
        if route == "checkpoint":
            assert "[skip] checkpoint: stage 04 already exists" in output
        return output

    return render


def test_video_title_removes_terminal_controls(render_title):
    output = render_title("a\x1b[31mb\x00\x07\x08\x0b\x0c\x1f\x7f")

    assert "\x1b" not in output
    assert not any(ch in output for ch in "\x00\x07\x08\x0b\x0c\x1f\x7f")
    assert "abcdefghijk a[31mb\n" in output


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        pytest.param("日本語 title", "日本語 title", id="ordinary"),
        pytest.param("", "", id="empty"),
        pytest.param("a\tb\nc", "a\tb\nc", id="keep-tabs-and-newlines"),
        pytest.param("題" * 300 + "overflow", "題" * 300, id="limit-300"),
    ],
)
def test_video_title_preserves_display_contract(render_title, title, expected):
    output = render_title(title)

    assert f"abcdefghijk {expected}\n" in output
    assert "overflow" not in output
