# Whisper バックエンド / モデル選択

Stage 01 の字幕が取れない動画（または `--local-media`）では Whisper で文字起こしする。
バックエンドとモデルを `config.json` で選べる。

## 設定 (`config.json`)

```jsonc
{
  "whisper_backend": "auto",   // "auto" | "mlx" | "openai"
  "whisper_model": null         // null=各backendの既定 / "small" / "medium" / "large-v3-turbo" ...
}
```

- **`whisper_backend`**
  - `auto`（既定）: **Apple Silicon で `mlx-whisper` が入っていれば MLX（GPU）**、それ以外は openai-whisper（CPU）。
  - `mlx`: 常に MLX（Apple Silicon 専用）。
  - `openai`: 常に openai-whisper（PyTorch/CPU）。
- **`whisper_model`**: 論理名。`null` なら backend 既定（MLX→`large-v3-turbo` / openai→`small`）。
  MLX は `mlx-community/whisper-<name>` に解決（フル repo id も可）。

## インストール

```bash
# Apple Silicon（推奨）: GPU バックエンド
uv sync --extra mlx
uv run --extra mlx python -m pipeline_youtube.main ...

# 互換/フォールバック（torch・CPU）
uv sync --extra whisper
uv run --extra whisper python -m pipeline_youtube.main ...
```

`mlx` extra は **プラットフォームマーカー付き**なので、Linux/CI で `--extra mlx` を付けても
no-op（インストールされない）。Apple Silicon でのみ実体が入る。

## モデルの目安（速度↔精度↔メモリ）

| モデル | メモリ | 速度 | 日本語精度 |
|---|---|---|---|
| tiny / base | ~1GB | 最速 | 低い（非推奨） |
| small（openai既定） | ~2GB | 速い | 実用ライン |
| medium | ~5–6GB(FP32) | 普通 | 良い |
| large-v3 | 量子化~2–6GB | 遅い | 最良 |
| **large-v3-turbo（MLX既定）** | ~2–3GB | 速い | large 近い |

> openai-whisper（CPU/FP32）で `large` は ~10–13GB に達しうるので、メモリ制約がある環境では
> 避ける。large 系を使うなら MLX（GPU・量子化）が前提。

## おすすめ（Apple Silicon / M-series Mac）

- **`whisper_backend: "auto"`（＝MLX）＋ `whisper_model: null`（＝large-v3-turbo）** が最適。
  GPU を使うので**ファンレス機でも CPU を焼かず**、メモリも軽く、日本語精度も高い。
- メモリをさらに絞りたい/速度優先なら `whisper_model: "small"` や `"medium"`。
- CPU しか無い環境（Linux サーバ等）は `openai` ＋ `small`〜`medium`。

詳細な背景は実装コメント（`pipeline_youtube/transcript/whisper_fallback.py`）参照。

## 既知の制限と今後の予定（Stage 01 文字起こし）

- **transcript キャッシュは backend/model で分離していない**（既知・意図的な見送り）。
  永続キャッシュのキーは `(video_id, tier, lang)` で、`whisper_backend`/`whisper_model` を
  含まない（`transcript/base.py`）。そのため一度文字起こし済みの動画を、後で
  `openai/small` → `mlx/large-v3-turbo` のようにモデルを変えて再実行しても、**古い
  キャッシュが再利用され新モデルの結果が反映されない**。今すぐ反映したい場合は当該
  動画の transcript キャッシュを削除する。
  - これは Codex のレビュー指摘（キャッシュキーに backend/model を含めるべき）と同じ
    論点で、**指摘は妥当**だと認識している。本 PR では**あえて修正を見送る**判断をした
    （トレードオフ）。理由は下記の Stage 01 刷新でキャッシュ層ごと再設計する見込みのため、
    今キーだけ直すと二度手間になり、刷新後に陳腐化する可能性が高いから。
- **Stage 01 の文字起こしは現状速度が遅い**（本パイプラインの 01–04 のボトルネック）。
  別 PR で 01 を刷新予定:
  1. まず**速度優先（精度度外視）で高速に文字起こし**（Obsidian の YTranscriptor 流。
     ただし時間軸の刻み方が現リポと異なる点は要正規化）。
  2. その後 **web search で誤変換の用語を訂正**し、用語を確定。
  3. （粒度を調整した上で）まとめ生成まで 01 で行うかは設計時に確定する。
  - この刷新でキャッシュ設計を見直すため、上記キャッシュキーの修正もそこで併せて対応する。
