# 実行サンプル

`pipeline-youtube` を 3 本の動画を含むプレイリストに対して実行したときのコンソール出力の読み方です。

## 入力

```bash
uv run python -m pipeline_youtube.main \
    "https://www.youtube.com/playlist?list=PLexample" \
    --concurrency 2 \
    --max-chapters 5
```

## 期待される出力

```
vault_root: /Users/you/Obsidian Vault
dry_run: False
model: sonnet
capture_format: auto
concurrency: 2
min_playlist_size: 3
max_chapters: 5
fetching metadata...
playlist: 'Agent Teams 実践'
videos: 3
run_time: 2026-04-16T19:32:07

[1/3] _h3decBW12Q Agent Teams とは何か
  [01] scripts... source=official snippets=241 lang=ja
  [02] summary... in=8421 out=3102 cost=$0.034
  [03] capture... 5/5 ranges fmt=webp
  [04] learning... in=12034 out=4211 cost=$0.045

[2/3] xyz456ABC01 Context Anxiety と Harness
  [01] scripts... source=auto-generated snippets=188 lang=en
  [02] summary... in=6290 out=2480 cost=$0.028
  [03] capture... 4/4 ranges fmt=webp
  [04] learning... in=9512 out=3800 cost=$0.041

[3/3] kJ9mXpqRsT2 α/β/γ/leader 設計パターン
  [01] scripts... source=official snippets=309 lang=ja
  [02] summary... in=10201 out=3602 cost=$0.039
  [03] capture... 6/6 ranges fmt=webp
  [04] learning... in=14800 out=4902 cost=$0.053

=== Video processing summary ===
succeeded: 3/3

=== Stage 05 Synthesis (Agent Teams) ===
MOC:       /Users/you/Obsidian Vault/Permanent Note/08_YouTube学習/05_Synthesis/2026-04-16 1932 Agent Teams 実践/00_MOC.md
chapters:  5
  - 01_Agent Teams の基礎概念.md
  - 02_ハーネスエンジニアリング.md
  - 03_コンテキスト管理戦略.md
  - 04_α_β_γ_leader の役割分担.md
  - 05_実践パターンと落とし穴.md
meta:      /Users/you/Obsidian Vault/Permanent Note/08_YouTube学習/05_Synthesis/2026-04-16 1932 Agent Teams 実践/_meta/duplicate_score.json
tokens:    in=47821 out=11302 cache_read=95200 cache_create=26100
cost:      $0.412
duration:  73.4s
```

## 読み方

### 先頭のメタ表示

- `vault_root` 〜 `max_chapters`: CLI/config の最終解決値
- `run_time`: その run の基準時刻。同一 run 内の全ファイル名の prefix (`YYYY-MM-DD-HHMM`) に使われる

### 動画ごとのブロック

```
[k/N] <video_id> <title>
  [01] scripts...  source={official|auto-generated|whisper} snippets=N lang=<code>
  [02] summary...  in=N out=N cost=$X
  [03] capture...  M/N ranges fmt={webp|gif}
  [04] learning... in=N out=N cost=$X
```

- `[01] source`: どの tier で字幕が取れたか (tier 1 official → tier 2 auto → tier 3 whisper)
- `[03] M/N ranges`: stage 02 が切り出したテーマ範囲の個数 N に対して M 個のキャプチャが成功
- `cost`: 各 `claude -p` 呼び出しの実課金額合計 (OAuth 経由 Pro/Max 定額でも内部集計される)

### 並列処理時の出力順

`--concurrency >= 2` のときは動画ブロックがインターリーブ (混ざって) 出力されます。ブロック先頭の `[k/N]` を頼りに読んでください。

### Stage 05 セクション

```
MOC:       <00_MOC.md の絶対パス>
chapters:  <章数>
  - 01_<章名>.md
  - ...
meta:      <_meta/duplicate_score.json の絶対パス>
tokens:    in=N out=N cache_read=N cache_create=N
cost:      $X.XXX
duration:  Y.Ys
```

`tokens` の 4 項目は α/β/γ/leader の合計。Claude の server-side cache が 5 分以内の連続呼び出しでは cache_read 主体になる (`cache_create` は α の初回のみ)。

## スキップケース

### プレイリストが小さいとき

```
succeeded: 2/2
[skip] only 2 videos succeeded (< 3), stage 05 skipped
```

`--min-playlist-size 2` で緩和可。

### checkpoint による 01〜04 スキップ

同日同プレイリストで再実行すると:

```
checkpoint: 3 videos already complete, will skip

[1/3] _h3decBW12Q Agent Teams とは何か
  [skip] checkpoint: stage 04 already exists
...
```

既存の 04 md 本文は Stage 05 に引き継がれるので、`--synthesis-only` 無しでも 05 は実行されます。強制再処理したいときは `--force-video <video_id>`。

### Stage 03 の format fallback

ffmpeg に libwebp も gif2webp も無い環境:

```
  [03] capture... 4/4 ranges fmt=gif
```

自動で GIF にフォールバック。`--capture-format webp` を明示していたら `format_unavailable` で 03 だけ失敗し、04 には進みます (画像埋め込みが抜けるだけ)。

## Phase 1 / Phase 3 分離実行 (`--stop-after-capture` / `--resume-reviewed`)

### Phase 1: 01〜03 までで停止

```bash
uv run python -m pipeline_youtube.main \
    "https://www.youtube.com/playlist?list=PLexample" \
    --stop-after-capture
```

```
[1/3] _h3decBW12Q Agent Teams とは何か
  [01] scripts... source=official snippets=241 lang=ja
  [02] summary (model=haiku)... in=8421 out=3102 cost=$0.008
  [03] capture... 5/5 ranges fmt=webp
  [stop-after-capture] review 02_Summary.md then re-run with --resume-reviewed

[2/3] xyz456ABC01 Context Anxiety と Harness
  [01] scripts... source=auto-generated snippets=188 lang=en
  [02] summary (model=haiku)... in=6290 out=2480 cost=$0.006
  [03] capture... 4/4 ranges fmt=webp
  [stop-after-capture] review 02_Summary.md then re-run with --resume-reviewed

[3/3] kJ9mXpqRsT2 α/β/γ/leader 設計パターン
  [01] scripts... source=official snippets=309 lang=ja
  [02] summary (model=haiku)... in=10201 out=3602 cost=$0.010
  [03] capture... 6/6 ranges fmt=webp
  [stop-after-capture] review 02_Summary.md then re-run with --resume-reviewed

[stop-after-capture] Phase 1 complete. Review 02_Summary.md, set `reviewed: true`, then re-run with --resume-reviewed.
```

この間に Obsidian で `02_Summary_Processing_Unit/.../.md` を開き、校閲済み動画の frontmatter に `reviewed: "true"` をセットする。

### Phase 3: reviewed:true だけ Stage 04〜05 を走らせる

```bash
uv run python -m pipeline_youtube.main \
    "https://www.youtube.com/playlist?list=PLexample" \
    --resume-reviewed
```

```
checkpoint: 0 videos already complete, will skip
  [skip] xyz456ABC01: reviewed='false'

[1/3] _h3decBW12Q Agent Teams とは何か
  [04] learning (model=sonnet)... in=12034 out=4211 cost=$0.045

[3/3] kJ9mXpqRsT2 α/β/γ/leader 設計パターン
  [04] learning (model=sonnet)... in=14800 out=4902 cost=$0.053

=== Video processing summary ===
succeeded: 2/3
  SKIP xyz456ABC01: reviewed='false'

=== Stage 05 Synthesis (Agent Teams) ===
[skip] only 2 videos succeeded (< 3), stage 05 skipped
```

## モデルカスケード (config.json の `models`)

`config.json` に `models` を設定していると、各 stage/agent の `claude -p` 呼び出しログがそのモデル名で出る。

```
  [02] summary (model=haiku)...
  [04] learning (model=sonnet)...
=== Stage 05 Synthesis (Agent Teams) ===
  (α=haiku / β=sonnet / γ=haiku / leader=opus でそれぞれ発話)
```

未設定のキーは CLI `--model` (デフォルト `sonnet`) にフォールバック。
