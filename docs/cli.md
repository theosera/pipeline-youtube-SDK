# CLI リファレンス

`pipeline-youtube` コマンドの全フラグ一覧と挙動です。

## `--help` 出力

```
Usage: python -m pipeline_youtube.main [OPTIONS] [URL]

  Process a YouTube playlist or single-video URL end-to-end.

Options:
  --dry-run                       Do not write to vault; print to stdout only.
  --concurrency INTEGER RANGE     Videos in parallel (1-5, default 1).
                                  [1<=x<=5]
  --skip-synthesis                Skip stage 05 after 01-04 finish.
  --synthesis-only                Skip stages 01-04 and re-run only stage 05
                                  against existing 04 md files for today's
                                  date.
  --force-video TEXT              Force reprocess specific video IDs even if
                                  checkpoint shows complete. Repeatable.
  --capture-format [auto|webp|gif]
                                  Animated capture output format. Default auto
                                  picks WebP when possible.
  --model TEXT                    Claude model alias for stages 02/04/05
                                  (sonnet, haiku, opus, or full ID).
  --min-playlist-size INTEGER RANGE
                                  Skip stage 05 when fewer than N videos
                                  succeed (default 3).  [default: 3;
                                  1<=x<=100]
  --max-chapters INTEGER RANGE    Cap β's chapter count via prompt constraint.
                                  Unset = let β decide.  [1<=x<=30]
  --config PATH                   Override config.json path.
  --help                          Show this message and exit.
```

## フラグ詳細

### 入力関連

- **`URL`** (引数): YouTube プレイリスト URL (`.../playlist?list=...`) または単一動画 URL (`.../watch?v=...`)。省略時は usage を表示して exit 2。

### 実行モード

- **`--dry-run`**: Vault への書き込みを一切行わず、stdout に生成内容を出力。`config.json` の `vault_root` 存在チェックは依然必要 (実際の書き込みはしない)。
- **`--skip-synthesis`**: 動画単位の 01〜04 のみ実行。プレイリスト単位の Stage 05 はバイパス。
- **`--synthesis-only`**: 今日の日付のプレイリストフォルダ配下にある既存 `04_Learning_Material/*.md` を読み込み、Stage 05 のみ再実行。α/β/γ/leader の章立てを作り直したいときに使う。当日の 04 フォルダが無いと `click.UsageError` で exit。
- **`--stop-after-capture`**: Phase 1 実行。Stage 01〜03 を動画単位で完了させて停止する。続けてユーザーが Obsidian 上で 02_Summary.md を校閲し `reviewed: true` に書き換える運用向け。
- **`--resume-reviewed`**: Phase 3 実行。Stage 01〜03 をスキップし、02_Summary.md の frontmatter `reviewed: true` が付いた動画だけを対象に Stage 04〜05 を走らせる。`--synthesis-only` とは異なり Stage 04 を含めて再実行する。

### 並列処理

- **`--concurrency N`** (1〜5, default 1): 動画単位を `asyncio.Semaphore(N)` で並列に回す。Whisper (tier 3 文字起こし) は内部のファイルロックで常に 1 本ずつ、他段は並列。

### Stage 03 (Capture)

- **`--capture-format {auto,webp,gif}`** (default `auto`):
  - `auto`: ffmpeg に libwebp があれば `direct` WebP、なければ `gif2webp` 経由 WebP、両方無ければ native GIF にフォールバック。
  - `webp`: WebP を強制。エンコーダが無ければ `format_unavailable` で 03 は失敗。
  - `gif`: 2-pass palette の GIF を必ず出力。

### AI モデル

- **`--model`** (default `sonnet`): Stage 02 (要約) / 04 (学習材料) / 05 (synthesis) が使う `claude -p` モデル。`sonnet` / `haiku` / `opus` のエイリアス、または full ID (`claude-sonnet-4-6` 等)。

### Stage 05 (Synthesis) チューニング

- **`--min-playlist-size N`** (default 3, 1〜100): Stage 05 を起動する最低動画成功数。`--synthesis-only` でも適用。
- **`--max-chapters N`** (1〜30, default 未指定): β (ChapterArchitect) に章数上限をプロンプトで指示。未指定なら β が内容量に応じて自動決定 (3 章以上)。

### リトライ / 再実行

- **`--force-video VIDEO_ID`** (繰り返し可): checkpoint で完了扱いになっている video_id を無視して再処理。例: `--force-video abc123 --force-video xyz789`

### 設定

- **`--config PATH`**: デフォルトは repo root の `config.json`。別環境用 JSON を渡したいときに使う。

## よく使うパターン

```bash
# プレイリストを通常実行 (01〜05)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx"

# 動作確認 (API 課金なし・書き込みなし)
uv run python -m pipeline_youtube.main "<URL>" --dry-run

# CI / 試運転用に軽く
uv run python -m pipeline_youtube.main "<URL>" --model haiku --capture-format gif --skip-synthesis

# 5 本同時処理
uv run python -m pipeline_youtube.main "<URL>" --concurrency 5

# 04 を作り直さずに 05 だけ作り直す (同日)
uv run python -m pipeline_youtube.main "<URL>" --synthesis-only --max-chapters 6

# 単一動画 (Stage 05 は 3 本未満なので自動スキップ)
uv run python -m pipeline_youtube.main "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Exit コード

- `0`: 正常完了
- `1`: プレイリストに動画が 1 本も無かった / フェッチ失敗
- `2`: URL 未指定 (usage 表示)
- その他: Python 例外 — traceback が stderr に出力される
