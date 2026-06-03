---
name: py-coding-conventions
description: pipeline-youtube-SDK (Python + Pydantic) のコーディング規約。型優先 (Tier1=Pydantic スキーマ) / 型ヒント徹底 / エラー全送 / メタデータ削減 / 品質ゲート3層 (ruff→Pyright→mypy strict) / LSP で決定論的型情報を記述時注入。**このリポの Python を書く・直す・レビューする前に必ずこの Skill をロードしてから**着手せよ。規約原本は `docs/ai-coding-conventions.md`。常時 CLAUDE.md に @import するとトークンを食うため、発火条件付きで分離している。
# allowed-tools は Read のみを事前承認する (規約原本 docs/ai-coding-conventions.md 読取のため)。
# これは「事前承認の最小化」であって他ツールの禁止ではない。allowed-tools は列挙ツールを
# 承認なしで使えるようにするだけで、未列挙ツールはセッション通常の permission 設定に従う
# (= 本 Skill 中も Bash/Edit 等は通常どおり都度承認で使える)。Read-only 境界が要るなら
# deny / disallowed 設定で別途強制すること。
allowed-tools: Read
---

# py-coding-conventions

pipeline-youtube-SDK (Python + Pydantic) の AI-native コーディング規約。CLAUDE.md の
常時 `@import` から発火条件付きスキルへ分離したもの。**Python を書く前に**ロード。

## まず原本を読む (single source of truth)

規約の**全文は `docs/ai-coding-conventions.md`**。コードを書く/直す前にこの doc を
Read すること。本 SKILL.md は重複を避けるため**圧縮した発火用サマリ**のみを持つ
(原本を二重化しない = DRY / トークン最小)。

## 7原則 (圧縮サマリ — 詳細は原本 docs)

1. **エラーは全送** — スタックトレースを省略しない。`mypy --show-traceback` / `pytest -v`。
   報告は4点 (再現コマンド `pytest tests/...` / 全 trace / 関連ファイル / 環境)。
2. **型で構造を表現** — `NewType` で branded type / Pydantic `BaseModel` で schema-first /
   union error は型で明示 / `field: T | None` を明示し None チェック必須。
   `Any` は避ける (ruff/mypy strict で検出)。外部入力は Pydantic でパースしてから扱う。
3. **メタデータ削減** — スキーマは DRY / DB 結果は件数+サンプル / ログは最新10行+エラー /
   API は例示でなく型定義を渡す。
4. 言語選定に AI 相性度 (本リポは型4/err3/doc4 = alt。型ヒント徹底で底上げ)。
5. **コードの形が説明書** — 型で言えることはコメントにしない。変数名は人間向け・型は AI 向け。
6. **テストより先に型チェック** — 構造指示 (型エラー) は事後報告 (テスト失敗) より修正精度が高い。
7. **コンテキスト圧縮を習慣化** (Context Rot 対策。S/N 最大化、不可逆要約を避ける)。
8. **LSP で決定論的型情報を記述時に注入** — 推測でなくグラウンドトゥルース。

## 品質ゲート3層 (守備範囲が別物。重ねる)

```
Layer3 Convention : ruff                    規約・複雑度        =「良く書けてるか」
Layer2 Semantic   : LSP = Pyright（記述時） 型整合・シンボル解決 =「動くか」
                  + mypy strict（CIゲート / コミット時）
Layer1 Syntax     : Python パーサー          構文の正当性
```
Pyright (エディタ即時) と mypy (CIゲート) は補完関係。LSP セットアップは
`npm install -g pyright` → `/plugin → Discover → "LSP" → Python → user scope`。
依存・ruff/mypy 設定は `pyproject.toml`。
