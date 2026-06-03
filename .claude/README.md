# .claude/

このディレクトリは Claude Code がセッション開始時に参照する project-level 設定。
本リポは 3 層設計の一部: グローバル層 (`../CLAUDE.global.md`) = ガードレール /
project `../CLAUDE.md` = リポ固有ハードルール + スキル発火表 / `skills/` = 発火条件付き
の作業規約 RAG。トークン削減のため、コーディング規約は常時ロードせず発火時にロードする。

## skills/

project-level スキル (`<name>/SKILL.md`) を置く。ディレクトリ名がスキル名。
`description` のトリガ文 + 親 `CLAUDE.md` の**スキル発火表**で発火する。

| スキル | 発火条件 | 用途 |
|---|---|---|
| `py-coding-conventions` | このリポの Python を書く/直す前 | AI-native 規約の発火用サマリ (原本は `docs/ai-coding-conventions.md`) |
