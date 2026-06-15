# Stage 01 文字起こし — 高速取得＋LLM校正（誤変換訂正）

Stage 01 は2段構成:

1. **01a 高速文字起こし（LLM 非介在）**: YouTube は auto-captions、`--local-media` は Whisper。
   速度優先で、誤変換はこの段では直さない。
2. **01b 誤変換訂正（任意・LLM＋web search）**: チャンク化したトランスクリプトを LLM（既定 Opus）に渡し、
   固有名詞・専門用語の誤変換を **web 検索で事実確認**して訂正する。**要約はしない**。

01b は **タイムスタンプを保持**し、訂正済みテキストを `TranscriptResult.snippets` に畳み戻すので、
Stage 02/03/04 が訂正済みトランスクリプトを消費する（02/03 の `[MM:SS ~ MM:SS]` 契約は不変）。

## 有効化（オプトイン）

01b は**有料・低速**なので既定で OFF。`config.json` で有効化する:

```jsonc
{
  "transcript_correction": true,
  "models": {
    "stage_01_correct": {"provider": "anthropic", "model": "opus"}
  }
}
```

- `transcript_correction: true` で 01b を実行。`false`（既定）なら 01a の生トランスクリプトをそのまま出力。
- 校正は **Anthropic プロバイダに固定**（web search は Anthropic 専用）。`--provider` / `--hybrid` を付けても
  `stage_01_correct` は常に Anthropic に保たれる（`selection.apply_selection`）。
- Anthropic の **server-side web_search ツール**＋**拡張思考（thinking）**が自動で有効化される。

## 挙動・安全性

- **ベストエフォート**: LLM エラー・不正 JSON・件数不一致は**その場の生テキストにフォールバック**し、
  Stage 01 を止めない／タイムスタンプをずらさない。
- **バッチ処理**: 長尺は一定チャンク数ごとに分割して校正（既定 40 チャンク/回）。
- **コスト/レイテンシ**: 長尺 × Opus × web search は高コスト。費用が気になる場合は
  `stage_01_correct` の model を下げる、または `transcript_correction: false` のままにする。
- `--dry-run` では 01b はスキップ（課金回避）。

## キャッシュ（モデル変更で再実行される）

- 校正のLLM出力は `(provider, model, system, prompt)` キーの LLM キャッシュに載るため、
  `stage_01_correct` のモデルを変えれば**別キー**になり再校正される（古い結果のサイレント再利用なし）。
- 01a の whisper 生トランスクリプトのキャッシュ tier 名も **backend+model で修飾**したので
  （`whisper_cache_tag()`）、`whisper_backend`/`whisper_model` を変えれば再文字起こしされる
  （Codex 指摘のキャッシュキー問題を解消）。

## 仕組み（タイムスタンプ保持）

LLM には `[idx] (MM:SS) text` 形式で番号付きチャンクを渡し、`[{"idx", "text"}]` の **JSON で 1:1 訂正**を
返させる。行の統合・分割・並べ替え・時刻改変は禁止。idx で元チャンクへ写し戻し、`start` を再付与する。
