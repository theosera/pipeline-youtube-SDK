#!/usr/bin/env bash
# ops-logging Stop hook.
# Commit & push the ops logs accumulated during this turn — once. No-op when
# nothing changed (no empty commits, no spam). Never blocks the turn: exits 0
# even on push failure.
set -euo pipefail

LOG_REPO="${OPS_LOG_REPO:-$HOME/terminal-ops-logs}"
[ -d "$LOG_REPO/.git" ] || exit 0

# Stage only generated markdown logs — never `git add -A` (3-repo culture).
find "$LOG_REPO" -name '*.md' -not -path '*/.git/*' -print0 \
  | xargs -0 -r git -C "$LOG_REPO" add --

# Nothing staged → done.
git -C "$LOG_REPO" diff --cached --quiet && exit 0

git -C "$LOG_REPO" commit -q -m "ops log: $(date '+%Y-%m-%d %H:%M')" || exit 0

# Push with light exponential backoff; failure must not block the turn.
for delay in 0 2 4 8; do
  [ "$delay" -gt 0 ] && sleep "$delay"
  if git -C "$LOG_REPO" push -u origin HEAD >/dev/null 2>&1; then
    exit 0
  fi
done
exit 0
