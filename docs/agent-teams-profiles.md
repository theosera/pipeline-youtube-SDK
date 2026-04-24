# Agent Teams プロファイル運用ガイド (Stage 05)

Stage 05 Synthesis の Agent Teams は **プロファイル** でエージェント構成を切り替えられる。デフォルトは `auto` で、動画本数から自動選択される。CLI `--synthesis-profile` と `config.json` の `synthesis_profile` で明示上書きも可能。

## プロファイル一覧

| プロファイル | 構成 | 想定 n_videos | 主用途 |
|---|---|---|---|
| `standard` (現状と同じ) | α → β → Leader | 3–15 | 中規模プレイリスト全般 |
| `parallel` | α(バッチ並列) → merge → β → Leader | 15+ | 大規模プレイリストでの α 負荷分散 |
| `full` | α → β → Leader → Reviewer | 品質重視 | 公開用途・規約遵守の最終校正が欲しい場合 |
| `parallel+full` | α(並列) → merge → β → Leader → Reviewer | 30+ かつ品質重視 | 長尺公開向け |

`auto` 選択時の対応表:

```text
n_videos < 3            → skip
3  ≤ n_videos ≤ 15      → standard
15 < n_videos ≤ 30      → parallel
30 < n_videos           → parallel+full
```

`full` 相当のレビューアは自動では付かない。レビューアは追加コストが発生するため、必要なときだけ `--synthesis-profile full` / `parallel+full` で明示指定する。

## 適切な場面 / 不適切な場面

### standard が適切
- 3–15 本の中規模プレイリスト
- 既存の α/β/Leader プロンプトで十分な品質が出ているジャンル
- コストとスループットを優先したい

### standard が不適切
- **20+ 本**: α が全動画の 04 md を 1 度に処理するため、タイムアウト・トピック粒度のばらつき・コンテキスト圧迫が起きやすい → `parallel`
- **公開前の最終校正**: 核心要素の出典・矢印圧縮禁止・missing_topic_ids の補足反映などの規約遵守を Reviewer に担保させたい → `full`
- **新規ジャンル / 実験的プロンプト変更後**: 1 ショット品質が未検証 → `full` で保険をかける

### parallel の注意点
- α を動画単位で分割するため、クロスビデオ重複検出の精度が単一 α より落ちる可能性がある。`merge_topics()` でラベル正規化による同一概念マージと `duplication_count` 合算を行うが、エイリアス違い (例: "LLM" vs "大規模言語モデル") は merge 漏れしうる。本数が多い（15 本超）ほど利益が上回るので、**15 本未満には適用しない**。
- 並列度は `max_workers=3` 固定。claude CLI の並行上限を尊重しつつ、無駄な直列待ちを避けるバランス。

### full の注意点
- Reviewer は Leader 出力に対して **修正指示 JSON** を返し、`needs_revision: true` の場合のみ Leader を 1 回再呼び出しする (追加リトライはしない)。既存の Reflexion ループ (β 最大 3 回) とは独立。
- LLM 呼び出しが 1 回増える (通常プロファイルより ~25% の追加コスト目安)。

## CLI 例

```bash
# 自動判定 (未指定と同じ)
uv run python -m pipeline_youtube.main <url>
uv run python -m pipeline_youtube.main <url> --synthesis-profile auto

# 小規模でも品質重視で Reviewer を追加
uv run python -m pipeline_youtube.main <url> --synthesis-profile full

# 大規模を並列 α で処理
uv run python -m pipeline_youtube.main <url> --synthesis-profile parallel

# 長尺を並列 + Reviewer
uv run python -m pipeline_youtube.main <url> --synthesis-profile parallel+full
```

`config.json` で既定値を固定する場合:

```json
{
  "synthesis_profile": "full",
  "models": {
    "alpha": "haiku",
    "beta": "sonnet",
    "leader": "opus",
    "reviewer": "sonnet"
  }
}
```

優先順位は **CLI > config.json > auto**。

## 決定フロー (要約)

```text
┌──────────────────────────┐
│ 明示フラグ or config あり? │
└──────────┬───────────────┘
    yes│  no
       │   └─► n_videos で auto 判定
       └─► そのプロファイルを使用
```

`parallel` 成分 (名称に "parallel" が含まれる) が選択されたら α をバッチ並列、`full` 成分が含まれたら Leader 後に Reviewer を起動する。両成分が直交しているため、`parallel+full` は両方を適用する。

## 既存機能との関係

- **Reflexion (β リトライ)**: すべてのプロファイルで共通。`MAX_BETA_REFLEXION_RETRIES=3` は変更しない。
- **決定論的 coverage** (`compute_coverage`): 全プロファイル共通。LLM 呼び出しではない。
- **動的タイムアウト** (`compute_synthesis_timeouts`): 全プロファイル共通。α バッチ実行も同じ per-call タイムアウトを使用。
- **動画本数スキップ** (`--min-playlist-size`, デフォルト 3): 全プロファイルで先に評価。

## 実装のポイント

- `stages/synthesis.py:_select_profile()` が唯一の切替ロジック。CLI/config の正規化は `main.py` 側で済ませ、この関数は確定した override 名 (または `None`/`"auto"`) を受ける。
- `synthesis/agents.py:call_alpha_batched()` は動画を `batch_size` (デフォルト 10) に分割し、`ThreadPoolExecutor(max_workers=3)` で並列に α を呼ぶ。結果は `merge_topics()` でラベル正規化でマージする。
- `synthesis/agents.py:call_reviewer()` は `LeaderOutput + topics + chapters + coverage` を入力に、`ReviewerFeedback` (`needs_revision`, `fixes[]`) を返す。Leader 再呼び出しは `stages/synthesis.py` 側で 1 回だけ実行。
