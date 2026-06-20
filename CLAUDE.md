# CLAUDE.md (pipeline-youtube-SDK)

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.
本リポは Python + Pydantic 製の YouTube パイプライン SDK。

> 3 層設計: 普遍ルール (行動原則 / セキュリティ境界 / エスカレーション) は共通グローバル層
> `CLAUDE.global.md` (= `~/.claude/CLAUDE.md` 想定) にある。本ファイル = リポ固有 + 発火表。
> 詳細な作業規約は `.claude/skills/` に分離し、発火条件付きでオンデマンドにロードする
> (トークン削減)。

<!-- グローバル層を web/local 双方で確実にロードするため明示 import する。
     web セッションは ~/.claude/ を読まないので、リポ同梱の本ファイル経由で読み込む。 -->
@CLAUDE.global.md

## スキル発火表 (★着手前に必ずロード)

タスクが発火条件に一致したら、**着手前に必ず対応スキルをロード**する (裁量で省略しない)。

| 発火条件 (このタスクを始める前に) | 必ずロードするスキル |
|---|---|
| このリポの Python を書く / 直す / レビューする | `py-coding-conventions` |
| コマンド学習ログ機能の hook 設定 (`.claude/settings.json`) を書く・直す / マスキング規則・ログ出力先を変える / `capture-command.sh`・`push-log.sh` を触る | `ops-logging` |

## PR 分割規律 (★PR 作成前に必ず適用)

PR を作成する前に、変更内容を**性質別に分類**する。レビュー容易性のため
**性質が異なるものは束ねない**:

- **異なる実行経路 / 異なるレビュー観点は別 PR** にする。
- **live 実注入 (runtime wiring) と seam-only (準備) は別 PR** にする。
- **依存更新とアプリロジックを混在させない**。
- 束ね PR を作る場合は **Draft かつ umbrella と明記し、直接 merge の対象にしない**
  (個別の分割 PR を merge 対象とする)。

> 1 PR = 1 レビュー観点。チェックリストは `.github/pull_request_template.md` の
> Change Type / PR Scope Check を使う。

## Architecture invariant: main.py is a thin orchestrator (★ハードルール / 常時適用)

`main.py` は合成ルート (composition root)。残してよいのは
**CLI 定義 (引数/オプション)・段階の実行順序・モジュールの配線・終了/エラー処理**のみ。
グローバル CLAUDE.md が普遍ルールだけを持つのと同じ発想で、main.py には「普遍的な制御フロー」
だけを置き、各機能の HOW はモジュールへ出す (これが main.py 肥大化を招いた反省。非 SDK 版でも同様)。

- 機能の HOW (ロジック / パース / I/O / 分岐) は専用モジュールへ置き、main.py からは
  **呼び出す・配線する**だけにする (例: `cli_config.py` / `video_processing.py` /
  `run_result.py` / `proper_noun_sheet.py`)。
- 切替・モード・プロバイダ選択は `if/elif` の累積ではなく **config 値 + registry/strategy** で
  表現する (例: マルチ LLM の `providers/registry.py` (`invoke_llm`)、フォールバック chain の
  `fetchers=[...]`、`use_innertube` のような config フラグ)。
- 1 機能で main.py に増えてよいのは原則「呼び出し or 配線 数行」。これを超える追加は、
  **先に対象モジュールへ抽出**してから行う。
- 目安: `main.py` ≤ ~500 行。超過が見込まれる変更は抽出を着手条件とする。
- リトマス試験: main.py を 2 分読んで「何が・どの順で・何に繋がって起きるか」が分かること。
  HOW が漏れていたら抽出のサイン。

> 新機能の着手前に「配置先モジュール」と「main.py への変更 = なし / 配線のみ (想定行数)」を
> 要件として宣言する。オーケストレータを編集せずに足せない設計は、まだ main.py 依存が残っている。
> (型優先など実装規約は `py-coding-conventions` skill 側。本節は常時適用の構造ハードルール。)

## See also

- `docs/main-architecture.md` — main.py の合成ルート地図 (入口→cli→command→各段の配線)。
  **入口層 (main/cli/command/cli_validation/runtime/input_resolver/execution_plan/
  pipeline_runner/synthesis_runner/reporting) の import・配線・順序を変えたら同じ PR で更新する**
- `CLAUDE.global.md` — 全リポ共通のグローバル層 (行動原則 / セキュリティ境界 / 発火規律)
- `.claude/skills/py-coding-conventions/` — コーディング規約 (発火時ロード)
- `.claude/skills/ops-logging/SKILL.md` — git/shell/MCP 操作の学習ログ hook (設定の正典)
- `README.md` — 概要・セットアップ
- `docs/ai-coding-conventions.md` — AI-native コーディング規約 (原本。skill が参照)
- `pyproject.toml` — 依存・ruff/mypy 設定
