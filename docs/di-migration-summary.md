# DI 移行サマリ — 大域 reach-in 撤去（cache / vault_root）

> 本セッションで完了した「モジュール大域状態の依存性注入(DI)化」の作業記録。
> 目的: `get_cache()` / `get_vault_root()` の**隠れた大域 reach-in を撤去**し、依存を
> `Runtime` 経由の**明示注入（型シグネチャ上の必須引数）**に置き換える。
> 規律: `migration-plan-sdk.md §3-6/§4-4`（「真の並列化を阻むグローバル状態の除去」）の消化。

## 1. 成果サマリ

| 系列 | 対象 reach-in | PR | 状態 |
|---|---|---|---|
| cache DI | `get_cache()` | #55→#56→#57→#58→#59 | ✅ 完了（本番 reach-in 0 件） |
| vault_root DI | `get_vault_root()` | #63→#64→#65→#66 | ✅ 完了（本番 reach-in 0 件） |

- 全 PR **挙動不変**（本番経路は注入済み、撤去はデッドなフォールバックのみ）。
- 各 PR で 3 層ゲート通過: `ruff check` / `ruff format --check` / `mypy strict`（新規エラーなし）/ `pytest`。
- 最終全スイート: **1014 passed, 5 skipped**。
- `get_cache()` / `get_vault_root()` の**直接 reach-in は本番コードから 0 件**を確認。

## 2. PR 分割規律（CLAUDE.md 準拠）

「1 PR = 1 レビュー観点」「異なる実行経路は別 PR」「live 実注入と seam-only は別 PR」に従い、
**実行経路ごとに分割** + 最後に **cleanup（フォールバック撤去）** を 1 本。

### cache DI（5 本）
1. **#55 transcript**: `fetch_with_fallback` / `warm_transcript_cache` 必須化。波及順の都合で
   共有ハブ `run_stage_scripts` / `_process_video` / `_run_videos_concurrent` も本 PR で必須化。
2. **#56 capture**: `run_stage_capture` 必須化。
3. **#57 code-fetch**: `fetch_snippets_for_urls` 必須化。
4. **#58 invoke_llm**（最大の連鎖）: `invoke_llm` + 全 caller（summary/learning/correction/
   genres/synthesis agents/evaluators）必須化。
5. **#59 cleanup**: 大域 singleton（`_cache` / `get_cache` / `reset_cache`）撤去、
   `configure_cache` を純粋ファクトリ化（戻り値を `Runtime.cache` へ注入）。

### vault_root DI（4 本）
1. **#63 synthesis**: `run_stage_synthesis` 必須化 + `synthesis_runner` 配線。
   - **Codex P2 修正**を内包: `Runtime.vault_root` を `cfg.vault_root.expanduser().resolve()` で
     正規化（合成ルート）。検証（`ensure_safe_path`）と書き込み（`vault_root / safe_rel`）が
     同一の解決済みルートを使い、symlink された vault でも乖離しない。
2. **#64 per-video hub**: `compute_note_paths` / `create_placeholder_notes` / `run_stage_capture` /
   `_process_video` / `_run_videos_concurrent` / `_proper_noun_sheet_path` 必須化 + `pipeline_runner` 配線。
3. **#65 resume**: `_find_summary_md` / `_filter_to_reviewed` / `_collect_existing_learning_bodies` 必須化。
4. **#66 cleanup**: seam の `get_vault_root()` フォールバック撤去（`path_safety.ensure_safe_path` /
   `checkpoint._find_learning_folder` / `is_video_complete` / `get_completed_video_ids` を必須化）、
   付随して `resume._load_existing_04_body` / `report.resolve_eval_dir` / `run_stage_evaluation` /
   `chapter.validate_chapter_relative_path` を必須化。

## 3. 設計判断

- **必須化（required）を選択**: optional+フォールバックではなく `cache: Cache` / `vault_root: Path` の
  必須引数化。型シグネチャ上に依存が現れ、mypy が未注入を記述時に検出する（コーディング規約
  「テストより先に型チェック」）。
- **leaf 必須化の型連鎖**: leaf を必須化すると `Optional` を渡せなくなり呼び出し元も必須化が伝播。
  共有ハブ（`_process_video` 等）は最初の PR で先に必須化し、後続 PR を各 leaf だけで完結させた。
- **テストの注入値**: LLM/IO をモックするテストは無効キャッシュ `Cache(None, enabled=False)`、
  キャッシュ挙動検証テストは `configure_cache(...)` の戻り値、vault は `config.get_vault_root()`
  または `tmp_path`（config 未設定のテスト）を明示注入。
- **config 大域の残置**: `config.get_vault_root` / `_vault_root` / `set_vault_root` は
  `build_runtime` の**検証用途**とテスト設定の利便として残す。隠れた prod reach-in ではなくなった。
  完全撤去は全テストの再 churn を伴うため別タスク（§5）。

## 4. 途中で検出・修正した付随事項

- **Codex P2（#63）**: `Runtime.vault_root` が raw `cfg.vault_root` だった件 → 合成ルートで
  解決済みパスに正規化（上記 #63）。
- **CI の `ruff format --check`**: ローカルゲートに `ruff check` だけでなく `ruff format --check` を
  追加（cache 引数追記で 100 桁超になった呼び出しが CI でのみ落ちた #57 を是正）。
- **大量テスト churn の委譲**: 機械的な引数追記（cache: ~117 サイト / vault_root: ~65 サイト）は
  サブエージェントに分割委譲し、親が ground-truth の全スイートで検証して仕上げた。

## 5. 残タスク（本セッション範囲外）

「モジュール大域状態の除去」トラックの未消化分:

1. **`official._api` 大域 singleton**（両リポ）: `transcript/official.py` の lazy-init
   `YouTubeTranscriptApi` インスタンス。`migration-plan §3-6/§6-3` が名指しした並列化阻害要因。
2. **`providers/models config` getter**（SDK のみ）: `get_providers_config()` / `get_models_config()`
   の 2 reach-in。
3. **`config._vault_root` 大域の完全撤去**（SDK）: `get_vault_root` / `set_vault_root` を
   検証専用関数化し、テストの `config.set_vault_root` 依存を除去（大規模 churn）。
4. **非 SDK リポ (`pipeline-youtube`)**: vault_root DI は #77 で services 第一歩のみ。
   同様の経路別 DI + フォールバック撤去が未実施（cache モジュールは非 SDK には存在せず対象外）。

## 6. 検証コマンド

```bash
ruff check pipeline_youtube/ tests/
ruff format --check pipeline_youtube/ tests/
mypy pipeline_youtube/
uv run pytest               # 1014 passed, 5 skipped
# 本番 reach-in が 0 件であることの確認:
grep -rn "get_cache()\|get_vault_root()" pipeline_youtube/ --include="*.py" | grep -v "def get_"
```
