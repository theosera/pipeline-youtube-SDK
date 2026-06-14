# サブエージェント並列実行 (`--sub-agents`)

プレイリストの **01〜04 工程** を、逐次（または `--concurrency` のフラット並列）で
処理する代わりに、**N 個の独立したワーカープロセス（サブエージェント）** に分割して
並列実行するオプション。**05 統合フェーズは従来通り**、全シャードの出力をまとめて
1 回だけ走る。

これは**追加機能**であり、デフォルト（`--sub-agents 1`）は従来どおりの単一プロセス実行。
実行時にサブエージェント版を選びたいときだけ `--sub-agents 3` のように指定する。

## 分割ルール

連続レンジに分割し、**あまりは最後のシャード側**に出る。チャンクサイズは
`ceil(n / N)`。

| 動画数 n | `--sub-agents 3` の分割 |
|---|---|
| 23 | 1-8 / 9-16 / 17-23 (8, 8, 7) |
| 9  | 1-3 / 4-6 / 7-9 (3, 3, 3) |
| 7  | 1-3 / 4-6 / 7 (3, 3, 1) |
| 2  | 1 / 2 (空シャードは作らない) |

実装は `pipeline_youtube/parallel.py:split_into_shards`。

## 実行モデル

```text
親 (オーケストレータ)
├─ メタデータ取得で本数 n を確定
├─ run_time を 1 つ固定 (全シャードで共有 → 同じプレイリストフォルダに書き込む)
├─ サブエージェント 1 … python -m pipeline_youtube.main <url> --video-range 0:8  --skip-synthesis  → logs/sub_agent_1_*.log
├─ サブエージェント 2 … --video-range 8:16  --skip-synthesis                                       → logs/sub_agent_2_*.log
├─ サブエージェント 3 … --video-range 16:23 --skip-synthesis                                       → logs/sub_agent_3_*.log
└─ 全シャード完了後 … python -m pipeline_youtube.main <url> --synthesis-only  (= Stage 05 を 1 回)
```

各ワーカーは別プロセスなので**ログと障害が分離**される（1 シャードがハングしても他は
無事）。各ワーカーの標準出力／標準エラーは `logs/sub_agent_<i>_<timestamp>.log` に保存。

`--run-timestamp` / `--video-range` / `--code-bearing` は親が内部的に渡すフラグ（`--help` には
出さない）。ジャンル分類（Stage 00.5 ルーター）は **親で 1 回だけ**実行し、その結果（code_bearing）を
`--code-bearing` / `--no-code-bearing` で全シャードに固定する。これにより、ワーカーが各自で
ルーター（LLM 呼び出し）を再実行して、一過性のエラーやパース差で `code_bearing` がシャード間で
食い違うことを防ぐ。ルーター呼び出しもプレイリストあたり 1 回で済む。

## 使い方

```bash
# 従来どおり (デフォルト・単一プロセス)
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx"

# 3 サブエージェントで 01〜04 を並列、05 は従来どおり 1 回
uv run python -m pipeline_youtube.main "https://www.youtube.com/playlist?list=PLxxx" --sub-agents 3
```

各ワーカー内部では従来の `--concurrency` がそのまま効く（総並列度 ≒ サブエージェント数 ×
`--concurrency`）。API レート/CPU 負荷に注意。

## 制約

- `--sub-agents > 1` は `--dry-run` および位相フラグ
  (`--synthesis-only` / `--resume-reviewed` / `--stop-after-capture`) と併用不可
  （明確なエラーで弾く）。
- シャードは互いに素な動画集合を処理するが、**異なる動画が同一タイトル**の場合に同名ノートの
  衝突回避（`-2`, `-3` サフィックス付与）が別プロセス間で競合しうる。稀だが、その場合は
  `--sub-agents 1` で再実行すれば確実。
- 各シャードがメタデータを取得し直すため、メタデータ取得は数回走る（決定論的なので結果は同一）。
