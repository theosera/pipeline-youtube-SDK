# AI-Native Coding Conventions (Python)

原則は「説明でなく構造を渡す」「全部渡すか、要約して渡すかを使い分ける」。
コードの読者は3者（人間・AIエージェント・ユーザー）。AIは型構造から読むため、型を最優先で設計する。
本リポは Python + Pydantic（型4 / err3 / doc4 = AI相性 alt）。型ヒント徹底でAI相性を底上げする。

## 1. AIが読む優先順位（設計のTier）

```
Tier1 型定義・スキーマ(Pydantic)   ← AST的に解析される。最優先で表現
Tier2 エラー処理(型で表現)
Tier3 テスト(型チェック後 / mypy)
Tier4 ドキュメント・コメント       ← 型で言えることはコメントにしない
```
変数名は人間向け、型はAI向け。型があれば変数名が var1/var2 でもAIは修正できる。
AIへのフィードバックは構造指示型を優先: テスト失敗(事後報告=何が起きたか)より
型エラー(構造指示=どこが構造的に矛盾か)の方が修正成功率・精度が高い(論文の比較実験)。
→ コメント(嘘をつける)・変数名(省略可)でなく、mypy が検証する型/Pydanticスキーマに意図を埋める。

## 2. エラー報告（AIへ渡すとき）= 全送

- スタックトレースは省略・削除しない（「宝の山」）。
- `mypy --show-traceback`, `pytest -v` で詳細出力を ON。
- 報告は4点セット: 再現コマンド(`pytest tests/...`) / 全スタックトレース / 関連ファイル一式 / 環境。

## 3. 型駆動テンプレ

```python
from typing import NewType, Optional
from pydantic import BaseModel, Field, EmailStr

# branded_type : ドメイン概念を専用型に（誤送防止）
UserId = NewType("UserId", str)

# schema_first : スキーマを真実の源に。コード内で再定義しない
class User(BaseModel):
    id: str = Field(..., description="UUID")
    email: EmailStr

# union_error : エラーケースを型で明示
class Result(BaseModel):
    ok: bool
    data: Optional[User] = None
    error: Optional[str] = None
```
`Any` は避ける（ruff/mypy strict で検出）。外部入力は Pydantic でパースしてから扱う。
`field: T | None` を明示し None チェックを必須化。

## 4. コンテキスト/コスト最適化（メタデータ爆発を防ぐ）

請求の大半は指示でなくメタデータ（スキーマ重複・全ログ・冗長ツール出力）。

```yaml
checklist:
  - スキーマはDRY。一度定義し参照キーで指す（重複送信しない）
  - DBクエリ結果は「件数 + サンプル1-2件」だけ
  - ログは「最新10行 + エラー部分」に圧縮
  - APIは例示データでなく「型定義」を渡す
  - キャッシュ維持: システムプロンプトに時刻/乱数を入れない
削減目安: スキーマ統一 50-80% / ログフィルタ 70-90% / 関連性フィルタ 40-60%
```

Context Rot: 入力が長いほど精度は低下する(全主要LLMで実証 / Lost in the Middle)。
モデルは先頭・末尾に注意が偏り中間を無視する → 量でなく信号雑音比(S/N)を最大化。
圧縮は可逆(構造保持+参照マーカー)で行い、不可逆な要約は避ける(元に戻せない)。
コンテキスト削減はコスト・精度・レイテンシの三方向に同時効果を持つ。

## 5. 品質ゲート3層 / LSP（決定論的情報の注入）

LLM はトークン確率生成器でコードを「理解」しない → 型違反・未定義参照・ハルシネーションは
構造的必然。対策は決定論的情報を記述時に注入し、検知を実行時→記述時へ上流シフトすること
（エラーループ・トークン消費を削減）。守備範囲が別物の3層を重ねる（競合しない）:

```
Layer3 Convention : ruff                    規約・複雑度        =「良く書けてるか」
Layer2 Semantic   : LSP = Pyright（記述時） 型整合・シンボル解決 =「動くか」
                  + mypy strict（CIゲート / コミット時）
Layer1 Syntax     : Python パーサー          構文の正当性
```
- LSP=言語仕様の決定論的判定(誤検知ほぼ0) / Lint=設定で可変。両方使う。
- Pyright(エディタ即時) と mypy(CIゲート) は補完関係。
- Claude Code: `npm install -g pyright` → `/plugin` → Discover → "LSP" → Python → user scope。

## 6. 原則（要約）

1 エラーは全送  2 型で構造を表現  3 メタデータ削減  4 言語選定にAI相性度
5 コードの形が説明書  6 テストより先に型チェック  7 コンテキスト圧縮を習慣化
8 LSPで決定論的型情報を記述時に注入（推測でなくグラウンドトゥルース）
