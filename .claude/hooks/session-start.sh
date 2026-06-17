#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Guarantees that the `uv` resolved on PATH satisfies this repo's pin
# (`required-version = ">=0.11.8,<0.12"` in pyproject.toml) and that project
# dependencies are synced, so tests (`uv run pytest`) and linters
# (`uv run ruff check`) work from the first turn of a remote session.
#
# Why this is needed: some base images ship an older `uv` (e.g. 0.8.x) earlier
# on PATH than a compatible 0.11.x, which makes `uv run`/`uv sync` fail the
# version pin. This hook selects a compatible uv (preferring one already on the
# image, falling back to a pinned install) and pins it onto PATH for the session.
#
# Synchronous + idempotent + non-interactive.
set -euo pipefail

# Only run on a fresh session start. SessionStart also fires on resume/clear/
# compact; re-syncing there would needlessly block mid-session (the env is
# already prepared from the original startup). Read the payload only when stdin
# is piped (the real hook invocation), so manual runs from a terminal still work.
if [ ! -t 0 ]; then
  HOOK_SOURCE="$(cat | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("source",""))
except Exception: pass' 2>/dev/null || true)"
  if [ -n "${HOOK_SOURCE:-}" ] && [ "$HOOK_SOURCE" != "startup" ]; then
    exit 0
  fi
fi

# Only relevant in the remote (web) environment; local machines manage their own uv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Matches the pyproject pin: >=0.11.8,<0.12
REQ_MIN="0.11.8"
INSTALL_VERSION="0.11.21"

log() { echo "[session-start] $*" >&2; }

# Succeeds if "$1 --version" reports a uv in [0.11.8, 0.12).
uv_satisfies() {
  local v
  v="$("$1" --version 2>/dev/null | awk '{print $2}')" || return 1
  case "$v" in
    0.11.*) [ "$(printf '%s\n%s\n' "$REQ_MIN" "$v" | sort -V | head -n1)" = "$REQ_MIN" ] ;;
    *) return 1 ;;
  esac
}

# Pick a uv that satisfies the pin: PATH first, then known image locations.
UV_BIN=""
for cand in "$(command -v uv 2>/dev/null || true)" /usr/local/bin/uv "$HOME/.local/bin/uv"; do
  [ -n "$cand" ] && [ -x "$cand" ] || continue
  if uv_satisfies "$cand"; then UV_BIN="$cand"; break; fi
done

# Fall back to installing the pinned version into ~/.local/bin.
if [ -z "$UV_BIN" ]; then
  log "no compatible uv found; installing uv ${INSTALL_VERSION}"
  curl -LsSf "https://astral.sh/uv/${INSTALL_VERSION}/install.sh" \
    | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
  UV_BIN="$HOME/.local/bin/uv"
fi

UV_DIR="$(cd "$(dirname "$UV_BIN")" && pwd)"
log "using uv: $UV_BIN ($("$UV_BIN" --version 2>/dev/null))"

# Make the chosen uv win for the rest of the session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"${UV_DIR}:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
export PATH="${UV_DIR}:$PATH"

# Sync project dependencies (cached in the container image after first run).
log "running uv sync"
"$UV_BIN" sync

log "done"
