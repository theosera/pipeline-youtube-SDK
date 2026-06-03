# CLAUDE.md (pipeline-youtube-SDK)

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.
本リポは Python + Pydantic 製の YouTube パイプライン SDK。

> 3 層設計: 普遍ルール (行動原則 / セキュリティ境界 / エスカレーション) は共通グローバル層
> `CLAUDE.global.md` (= `~/.claude/CLAUDE.md` 想定) にある。本ファイル = リポ固有 + 発火表。
> 詳細な作業規約は `.claude/skills/` に分離し、発火条件付きでオンデマンドにロードする
> (トークン削減)。

## スキル発火表 (★着手前に必ずロード)

タスクが発火条件に一致したら、**着手前に必ず対応スキルをロード**する (裁量で省略しない)。

| 発火条件 (このタスクを始める前に) | 必ずロードするスキル |
|---|---|
| このリポの Python を書く / 直す / レビューする | `py-coding-conventions` |

## See also

- `CLAUDE.global.md` — 全リポ共通のグローバル層 (行動原則 / セキュリティ境界 / 発火規律)
- `.claude/skills/py-coding-conventions/` — コーディング規約 (発火時ロード)
- `README.md` — 概要・セットアップ
- `docs/ai-coding-conventions.md` — AI-native コーディング規約 (原本。skill が参照)
- `pyproject.toml` — 依存・ruff/mypy 設定
