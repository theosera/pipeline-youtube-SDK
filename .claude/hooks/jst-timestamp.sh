#!/bin/bash
# UserPromptSubmit hook — make every reply end with the current JST time.
#
# Claude Code injects a UserPromptSubmit hook's stdout into the model context
# before it answers, so emitting the current Japan-Standard-Time clock (plus a
# one-line instruction to echo it) makes each reply close with a
# `[YYYY-MM-DD HH:mm+JST]` line — no per-chat instruction needed. Works in both
# local and web sessions because it is committed to the repo's .claude/.
#
# JST is UTC+9 year-round (no DST), so the time is derived from UTC arithmetic
# and never depends on the container shipping the Asia/Tokyo tzdata zone; it
# falls back to TZ-based formatting only if `date -d` is unavailable.
#
# Best-effort and non-blocking: any failure exits 0, so a clock hiccup never
# interrupts the turn (mirrors the ops-logging hooks' "never block work" rule).
set -uo pipefail

now="$(date -u -d '+9 hours' '+%Y-%m-%d %H:%M' 2>/dev/null \
  || TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M' 2>/dev/null || true)"

# If the clock could not be read at all, stay silent rather than inject a
# malformed instruction.
[ -n "$now" ] || exit 0

printf '回答ルール: この回答の最後に、現在のシステム時刻を 1 行で `[%s+JST]` の形式で記載してください。\n' "$now"
exit 0
