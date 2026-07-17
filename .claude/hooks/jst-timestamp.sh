#!/bin/bash
# UserPromptSubmit hook — re-inject the two standing reply rules every turn.
#
# Claude Code feeds a UserPromptSubmit hook's stdout into the model context
# before it answers. Emitting the standing rules on *every* prompt (instead of
# once at session start) keeps them from decaying in long conversations, so each
# reply reliably (1) keeps non-command prose in Japanese and (2) closes by
# running the JST clock command and showing its output as the completion time.
#
# Committed to the repo's .claude/ so it applies in both local and Claude Code
# on the web without a per-chat instruction. Best-effort and non-blocking: any
# failure exits 0, so a hook hiccup never interrupts the turn.
set -uo pipefail

cat <<'EOF'
回答ルール（このセッションで厳守）:
1. コマンド類（シェルコマンド・コードブロック内の実行コマンド等）を除き、説明・回答文はすべて日本語で書く。
2. 回答の最後に必ず次のコマンドを実行し、その出力（JST の完了時刻）を 1 行で表示する:
   TZ=Asia/Tokyo date "+%Y-%m-%d %H:%M:%S JST"
EOF
exit 0
