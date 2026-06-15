# pipeline-youtube-SDK

> **[pipeline-youtube](https://github.com/theosera/pipeline-youtube) のマルチ LLM SDK バージョン**

YouTube プレイリスト → Obsidian Vault 自学自習用学習レポート生成パイプライン。

元リポジトリの `claude -p` CLI サブプロセス依存を排除し、**Ollama / LM Studio / Anthropic / OpenAI / Gemini** の各 LLM バックエンドを直接 SDK 経由で呼び出す設計に変更。

## 元リポジトリとの主な違い

| 項目 | pipeline-youtube (CLI版) | pipeline-youtube-SDK |
|---|---|---|
| AI 呼び出し | `claude -p` subprocess (OAuth) | SDK 直接呼び出し (API キー / ローカル) |
| 認証 | `claude login` (OAuth セッション) | API キー or ローカル (認証不要) |
| 対応 LLM | Claude のみ | Ollama, LM Studio, Anthropic, OpenAI, Gemini |
| ステージ別モデル | モデル名のみ切替 | プロバイダー + モデルを個別指定可 |
| CI/サーバー実行 | 不可 (OAuth 依存) | 可能 (API キー設定のみ) |
| コスト | Claude Pro/Max 定額 | ローカル LLM: 無料 / クラウド API: 従量課金 |

## アーキテクチャ

```mermaid
graph TD
    subgraph "Pipeline Stages"
        R["Stage 00.5 Router"]
        S01["Stage 01 Scripts"]
        S02["Stage 02 Summary"]
        S03["Stage 03 Capture"]
        S04["Stage 04 Learning"]
        S05["Stage 05 Synthesis"]
    end

    subgraph "Provider Layer"
        BASE["LLMProvider ABC"]
        OAI["OpenAICompatibleProvider"]
        ANT["AnthropicProvider"]
    end

    subgraph "Backends"
        OLLAMA["Ollama<br/>Qwen, Kimi, ..."]
        LMSTUDIO["LM Studio"]
        OPENAI["OpenAI API"]
        ANTHROPIC["Anthropic API"]
        GEMINI["Gemini API"]
    end

    R & S02 & S04 & S05 --> BASE
    S01 --> |"YouTube字幕 / Whisper"| S01
    S03 --> |"yt-dlp / ffmpeg"| S03
    BASE --> OAI & ANT
    OAI --> OLLAMA & LMSTUDIO & OPENAI & GEMINI
    ANT --> ANTHROPIC

    style BASE fill:#4a9eff,color:#fff
    style OAI fill:#10b981,color:#fff
    style ANT fill:#d97706,color:#fff
    style OLLAMA fill:#6366f1,color:#fff
    style LMSTUDIO fill:#6366f1,color:#fff
    style OPENAI fill:#6366f1,color:#fff
    style ANTHROPIC fill:#6366f1,color:#fff
    style GEMINI fill:#6366f1,color:#fff
```

> **ポイント**: Ollama / LM Studio / OpenAI / Gemini はすべて OpenAI 互換 API (`/v1/chat/completions`) を提供するため、`OpenAICompatibleProvider` 一つで統一的に扱えます。Anthropic のみ独自 API のため専用プロバイダーを使用します。

## セットアップ

```bash
# 前提: Python 3.13, uv, yt-dlp, ffmpeg が PATH にあること
cd pipeline-youtube-SDK
uv sync

# Whisper フォールバック (3次) を使う場合は別途インストール:
#   uv sync --extra whisper
# ★重要: Whisper は optional extra なので、素の `uv run ...` は実行時に環境をロックへ
#   同期し直して Whisper を毎回アンインストールする。Whisper を効かせる実行は必ず
#   `uv run --extra whisper ...` を使うこと。字幕が IP ブロックされた動画は、この
#   Whisper フォールバック (音声から文字起こし) だけが命綱になる。

# 編集可能インストール
uv pip install -e .

# 設定ファイルを作成
cp config.example.json config.json
# config.json で vault_root と providers/models を設定
```

### Ollama を使う場合 (推奨: 無料)

```bash
# Ollama をインストール: https://ollama.com
# モデルをダウンロード
ollama pull qwen3:8b

# config.json はデフォルトで Ollama を使用するので、
# vault_root を設定するだけで動作します
```

### Anthropic API を使う場合

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# config.json の models セクションを変更:
# "stage_02": {"provider": "anthropic", "model": "sonnet"}
```

### LM Studio を使う場合

```bash
# LM Studio を起動し、モデルをロード
# デフォルトで http://localhost:1234/v1 でサーバーが起動

# config.json:
# "models": {"stage_02": {"provider": "lmstudio", "model": "loaded-model-name"}}
```

## config.json の構成

```json
{
  "vault_root": "/path/to/your/Obsidian Vault",
  "providers": {
    "ollama":    {"base_url": "http://localhost:11434/v1"},
    "lmstudio":  {"base_url": "http://localhost:1234/v1"},
    "anthropic": {"api_key": "${ANTHROPIC_API_KEY}"},
    "openai":    {"api_key": "${OPENAI_API_KEY}"},
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                  "api_key": "${GEMINI_API_KEY}"}
  },
  "models": {
    "router":   {"provider": "ollama", "model": "qwen3:8b"},
    "stage_02": {"provider": "ollama", "model": "qwen3:8b"},
    "stage_04": {"provider": "anthropic", "model": "sonnet"},
    "alpha":    {"provider": "ollama", "model": "qwen3:8b"},
    "beta":     {"provider": "ollama", "model": "qwen3:8b"},
    "leader":   {"provider": "anthropic", "model": "sonnet"},
    "reviewer": {"provider": "anthropic", "model": "haiku"}
  }
}
```

> **混合構成の例**: Router は Ollama のローカルモデル (高速・無料)、Leader は Anthropic Claude (高品質) のように、ステージごとに異なるプロバイダーを使い分けられます。

### ランタイムのモデル選択 (`--provider` / `--hybrid`)

config.json を編集せずに、**その実行だけ**バックエンドを切り替えられます。**config.json が真実**で、フラグはその上に乗る粗い上書きです。

- **無指定**: config.json の `models`（工程別指定）をそのまま使用。**Anthropic を既定にしたいなら `models` を Anthropic で書く**（下の例）。
- **`--provider {anthropic|ollama|lmstudio}`**: 全工程をそのプロバイダの既定モデルに上書き（= 単一モデル実行）。`providers.<名前>.default_model` があればそれを、無ければ内蔵フォールバック（anthropic→`sonnet` 等）を使用。
- **`--hybrid`**: 重い工程（`stage_04` / `leader`）だけ Anthropic に引き上げ（要 `providers.anthropic`）。`--provider ollama --hybrid` で「軽い工程はローカル・重い工程は Anthropic」。
- 重い工程がオープン/ローカル backend になる場合、起動時に**品質低下の警告**を表示（自動では再ルートしない）。
- これらのフラグは **`--sub-agents` の各ワーカーへ自動的に引き継がれます**。

```bash
# config をいじらず、この実行だけ全工程 Anthropic
uv run python -m pipeline_youtube.main "URL" --provider anthropic

# 軽い工程はローカル Ollama、重い工程 (04/Leader) だけ Anthropic
uv run python -m pipeline_youtube.main "URL" --provider ollama --hybrid
```

> **Anthropic を既定にする (config が真実)**: `models` の各工程を `{"provider": "anthropic", "model": "sonnet"}` 等にしておけば、フラグ無しで Anthropic。ローカルで回したい時だけ `--provider ollama` を付けます。

### 固有名詞の正規化 (任意)

`glossary_path` に固有名詞辞書 (JSON) を指定すると、Stage 02 が出力本文・ONE_LINER の
**既知の誤変換を決定論的に正規化**します (例: `ビブコーディング` → `Vibe Coding`)。文脈推論
ではなく辞書照合なので幻覚は起きず、辞書に無い語はそのまま残ります。未指定 (`null`) の場合は
正規化を行いません。相対パスは config.json のあるディレクトリ基準で解決されます。

```json
{ "glossary_path": "glossary.json" }
```

辞書フォーマット:

```json
{
  "version": 1,
  "entries": [
    {"canonical": "Vibe Coding", "aliases": ["ビブコーディング", "バイブコーディング"],
     "reading": "ヴァイブコーディング", "category": "concept"}
  ]
}
```

## 使い方

```bash
# 通常実行: プレイリスト全体を 01〜05 まで処理
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx"

# Whisper フォールバックを効かせて実行 (字幕が無い/IP ブロックされた動画を音声から
# 文字起こし)。★`--extra whisper` を付けないと uv が Whisper を毎回剥がす点に注意。
uv run --extra whisper python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx"

# dry-run (Vault 書き込みなし)
uv run python -m pipeline_youtube.main "URL" --dry-run

# Stage 05 をスキップ
uv run python -m pipeline_youtube.main "URL" --skip-synthesis

# Stage 05 だけ再実行
uv run python -m pipeline_youtube.main "URL" --synthesis-only

# 単一動画
uv run python -m pipeline_youtube.main "https://www.youtube.com/watch?v=VIDEO_ID"

# ランタイムでバックエンドを上書き (config はいじらない)
uv run python -m pipeline_youtube.main "URL" --provider anthropic        # 全工程 Anthropic
uv run python -m pipeline_youtube.main "URL" --provider ollama --hybrid  # 軽=ローカル / 重=Anthropic
```

CLI オプションは元リポジトリと同一です。詳細は [docs/cli.md](docs/cli.md) を参照。

## パイプライン概要

1. **Stage 00.5 Router**: プレイリストのジャンル分類 (coding/humanities/business/...)
2. **Stage 01 Scripts**: タイムスタンプ付き文字起こし (YouTube字幕 → 自動生成 → Whisper フォールバック)
3. **Stage 02 Summary**: 意味単位タイムスタンプ範囲付き要約
4. **Stage 03 Capture**: 要点タイムスタンプの動画フレーム抽出 (WebP アニメーション)
5. **Stage 04 Learning**: 上記を「時系列→キャプチャ→要点」3点セットでテーマ単位に再構成
6. **Stage 05 Synthesis**: Agent Teams (α→β→Leader) で全動画を横断統合

## 権利・利用上の注意

元リポジトリと同様、動画由来の生成物は **利用者自身のローカル環境における個人学習目的** でのみ扱うことを想定しています。詳細は元リポジトリの [README](https://github.com/theosera/pipeline-youtube#%E6%A8%A9%E5%88%A9%E5%88%A9%E7%94%A8%E4%B8%8A%E3%81%AE%E6%B3%A8%E6%84%8F) を参照してください。

## ライセンス

本リポジトリのコードは AS-IS で提供されます。
