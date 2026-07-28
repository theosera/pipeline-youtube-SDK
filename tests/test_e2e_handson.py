"""End-to-end --handson CLI test with yt-dlp / ffmpeg / LLM stubbed.

Drives a 2.5-hour (9000s) single-video run through the real entry layer
(cli → command → validation → runtime → input resolution → plan →
handson runner) and the real handson package logic (segmenter
normalization, planner coverage, writer safety), stubbing only the
external boundaries: metadata fetch, Stage 01 transcript, clip capture,
provider configuration, and the LLM calls (FIFO canned responses).

Asserts the 09_YouTube学習_Session_only output tree, H:MM:SS stamps past
the legacy 99:59 ceiling, Q&A callout weaving, and the playlist-URL /
flag-conflict rejections.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pipeline_youtube import input_resolver as ir_mod
from pipeline_youtube import main as main_mod
from pipeline_youtube import provider_runtime as provider_runtime_mod
from pipeline_youtube.handson import planner as planner_mod
from pipeline_youtube.handson import segmenter as segmenter_mod
from pipeline_youtube.handson import steps as steps_mod
from pipeline_youtube.playlist import VideoMeta
from pipeline_youtube.providers.base import LLMResponse
from pipeline_youtube.stages import handson as handson_stage_mod
from pipeline_youtube.stages.capture import CaptureOutcome, CaptureResult
from pipeline_youtube.transcript.base import (
    TranscriptSnippet,
    TranscriptSource,
    build_result,
)

_DURATION = 9000  # 2.5h — past the legacy MM:SS (99:59) ceiling


def _video() -> VideoMeta:
    return VideoMeta(
        video_id="vid001",
        title="Long Talk",
        url="https://www.youtube.com/watch?v=vid001",
        duration=_DURATION,
        channel="Test",
        upload_date="20260601",
        playlist_title=None,
    )


def _videos_playlist() -> list[VideoMeta]:
    return [
        VideoMeta(
            video_id=f"vid{i:03d}",
            title=f"Video {i}",
            url=f"https://www.youtube.com/watch?v=vid{i:03d}",
            duration=120,
            channel="Test",
            upload_date="20260601",
            playlist_title="PL",
        )
        for i in range(1, 4)
    ]


def _transcript_result():
    return build_result(
        video_id="vid001",
        source=TranscriptSource.OFFICIAL,
        language="ja",
        snippets=[
            TranscriptSnippet(f"テキスト{i}", float(i * 30), 30.0) for i in range(_DURATION // 30)
        ],
    )


SEGMENTS_OUT = json.dumps(
    {
        "segments": [
            {"start_sec": 0, "end_sec": 3600, "label": "lecture", "summary": "前半の講演"},
            {"start_sec": 3600, "end_sec": 3900, "label": "qa", "summary": "失敗談と構想の経緯"},
            {"start_sec": 3900, "end_sec": 7500, "label": "lecture", "summary": "後半の講演"},
            {
                "start_sec": 7500,
                "end_sec": 7620,
                "label": "tips",
                "summary": "ショートカット小ネタ",
            },
            {"start_sec": 7620, "end_sec": 9000, "label": "lecture", "summary": "まとめ"},
        ]
    },
    ensure_ascii=False,
)

PLAN_OUT = json.dumps(
    {
        "steps": [
            {
                "index": 1,
                "label": "環境構築",
                "start_sec": 0,
                "end_sec": 3600,
                "goal": "環境を整える",
                "insight_ids": [],
            },
            {
                "index": 2,
                "label": "実装",
                "start_sec": 3900,
                "end_sec": 7500,
                "goal": "実装する",
                "insight_ids": ["q001"],
            },
            {
                "index": 3,
                "label": "まとめ",
                "start_sec": 7620,
                "end_sec": 9000,
                "goal": "振り返る",
                "insight_ids": [],
            },
        ],
        "unassigned_insight_ids": ["p001"],
    },
    ensure_ascii=False,
)


def _step_body_out(label: str, extra: str = "") -> str:
    return json.dumps(
        {
            "label": label,
            "body_markdown": f"## ゴール\n\n{label}のゴール。\n\n## 手順\n\n1. 操作する。{extra}",
        },
        ensure_ascii=False,
    )


MOC_OUT = json.dumps(
    {
        "title": "Long Talk ハンズオン",
        "moc_markdown": (
            "# Long Talk ハンズオン\n\n## ステップ構成\n\n"
            "- [[01_環境構築]] — 環境を整える\n"
            "- [[02_実装]] — 実装する\n"
            "- [[03_まとめ]] — 振り返る\n\n"
            "## 進め方\n\n上から順に進める。\n\n- 巻末まとめ: [[99_QA_Tipsまとめ]]"
        ),
        "summary_markdown": (
            "## Q&A から\n\n- [1:00:00] 失敗談と構想の経緯 (→ [[02_実装]])\n\n"
            "## Tips\n\n### 本編に紐づかない知見\n\n- [2:05:00] ショートカット小ネタ"
        ),
    },
    ensure_ascii=False,
)


def _fake_response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="opus",
        input_tokens=100,
        output_tokens=100,
        total_cost_usd=0.05,
        duration_ms=800,
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / ".obsidian").mkdir()
    return tmp_path


def _wire_common_stubs(monkeypatch, videos: list[VideoMeta]):
    monkeypatch.setattr(ir_mod, "fetch_metadata", lambda url: videos)
    monkeypatch.setattr(provider_runtime_mod, "configure_providers", lambda *a, **kw: None)
    from pipeline_youtube.genres import Genre

    monkeypatch.setattr(
        ir_mod, "classify_playlist_genre", lambda *a, **kw: (Genre.OTHER, "stubbed")
    )


class TestE2EHandson:
    def test_full_cli_single_long_video(self, vault: Path, monkeypatch):
        _wire_common_stubs(monkeypatch, [_video()])

        monkeypatch.setattr(
            handson_stage_mod, "run_stage_scripts", lambda video, path, **kw: _transcript_result()
        )

        def fake_capture(video, ranges, *, assets_subfolder, **kw):
            outcomes = [
                CaptureOutcome(
                    range=r,
                    image_path=Path(
                        f"/assets/{assets_subfolder}/pyt_{video.video_id}_h{i:02d}.webp"
                    ),
                )
                for i, r in enumerate(ranges)
            ]
            return CaptureResult(ranges=ranges, outcomes=outcomes, capture_format="webp")

        monkeypatch.setattr(handson_stage_mod, "capture_step_clips", fake_capture)

        queue = [
            _fake_response(SEGMENTS_OUT),  # H1 segment classification
            _fake_response(PLAN_OUT),  # H2 step plan
            _fake_response(_step_body_out("環境構築")),
            _fake_response(
                _step_body_out(
                    "実装",
                    "\n\n> [!question] Q&Aより [1:00:00]: 失敗談と構想の経緯",
                )
            ),
            _fake_response(_step_body_out("まとめ")),
            _fake_response(MOC_OUT),  # H5 MOC + final summary
        ]

        def fake_invoke(**kw):
            if not queue:
                pytest.fail("invoke_claude called more times than canned responses")
            return queue.pop(0)

        monkeypatch.setattr(segmenter_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(planner_mod, "invoke_claude", fake_invoke)
        monkeypatch.setattr(steps_mod, "invoke_claude", fake_invoke)

        cfg = vault / "config.handson.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/watch?v=vid00000001",
                "--handson",
                "--config",
                str(cfg),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert queue == [], "all canned LLM responses must be consumed"
        assert "Hands-on mode" in result.output
        for marker in (
            "[01] scripts",
            "[H1] segments",
            "[H2] step plan",
            "[H3] step clips",
            "[H4] step 1/3",
            "[H5] MOC+summary",
        ):
            assert marker in result.output
        assert "lecture=3 qa=1 tips=1" in result.output

        base = vault / "Permanent Note/09_YouTube学習_Session_only"
        scripts_notes = list((base / "01_Scripts_Processing_Unit").rglob("*.md"))
        assert scripts_notes, "transcript note must land under the 09 tree's 01 unit"

        handson_dirs = list((base / "05_Synthesis").iterdir())
        assert len(handson_dirs) == 1
        folder = handson_dirs[0]
        assert (folder / "00_MOC.md").exists()
        assert (folder / "01_環境構築.md").exists()
        assert (folder / "02_実装.md").exists()
        assert (folder / "03_まとめ.md").exists()
        assert (folder / "99_QA_Tipsまとめ.md").exists()
        assert (folder / "_meta/handson_meta.json").exists()

        step2 = (folder / "02_実装.md").read_text(encoding="utf-8")
        assert "> [!question] Q&Aより [1:00:00]" in step2
        assert "[1:05:00 〜 2:05:00]" in step2  # 3900s〜7500s in H:MM:SS
        assert f"![[{folder.name}/pyt_vid001_h01.webp]]" in step2

        summary = (folder / "99_QA_Tipsまとめ.md").read_text(encoding="utf-8")
        assert "[2:05:00]" in summary  # tips at 7500s, past the 99:59 ceiling
        assert "[[02_実装]]" in summary

        meta = json.loads((folder / "_meta/handson_meta.json").read_text(encoding="utf-8"))
        assert meta["duration_sec"] == _DURATION
        assert meta["unassigned_insight_ids"] == ["p001"]
        assert [s["label"] for s in meta["segments"]].count("qa") == 1

    def test_playlist_url_is_rejected(self, vault: Path, monkeypatch):
        _wire_common_stubs(monkeypatch, _videos_playlist())
        cfg = vault / "config.handson.json"
        cfg.write_text(json.dumps({"vault_root": str(vault)}), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/playlist?list=PL_fake",
                "--handson",
                "--config",
                str(cfg),
            ],
        )
        assert result.exit_code != 0
        assert "exactly one video" in result.output

    def test_conflicting_flag_fails_before_any_work(self, vault: Path):
        runner = CliRunner()
        result = runner.invoke(
            main_mod.cli,
            [
                "https://www.youtube.com/watch?v=vid00000001",
                "--handson",
                "--synthesis-only",
            ],
        )
        assert result.exit_code != 0
        assert "--handson cannot be combined" in result.output
