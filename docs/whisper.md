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
