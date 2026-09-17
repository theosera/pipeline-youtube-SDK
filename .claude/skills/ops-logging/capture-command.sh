#!/usr/bin/env bash
# ops-logging PostToolUse hook.
# Append "command + intent" of a git / shell / GitHub(MCP) action to the
# terminal-ops-logs repo. Records COMMAND + INTENT ONLY — stdout is never read,
# and token/credential patterns in the command string are fully masked.
# Never blocks the originating tool: always exits 0.
set -euo pipefail

LOG_REPO="${OPS_LOG_REPO:-$HOME/terminal-ops-logs}"
# Log repo not cloned (e.g. out-of-scope web session) → no-op, do not block.
[ -d "$LOG_REPO/.git" ] || exit 0
command -v jq >/dev/null 2>&1 || exit 0

payload="$(cat)"
tool="$(printf '%s' "$payload" | jq -r '.tool_name // empty')"
cwd="$(printf '%s' "$payload" | jq -r '.cwd // empty')"

# --- command + intent per tool kind --------------------------------------
case "$tool" in
  Bash)
    cmd="$(printf '%s' "$payload"  | jq -r '.tool_input.command // empty')"
    intent="$(printf '%s' "$payload" | jq -r '.tool_input.description // ""')"
    ;;
  mcp__github__*)
    # MCP GitHub call has no shell command. Record the tool + a SMALL ALLOWLIST
    # of structural metadata only — never bodies / titles / comments / file
    # contents, which can carry private text or secrets the regex mask cannot
    # catch (keeps the "command + intent only" guarantee).
    safe="$(printf '%s' "$payload" | jq -c '
      (.tool_input // {})
      | {owner, repo, pullNumber, issue_number, branch, base, head, ref, sha,
         path, method, name, tag, state, mergeMethod}
      | with_entries(select(.value != null))' 2>/dev/null || printf '{}')"
    cmd="$tool $safe"
    intent="GitHub MCP operation (args redacted to safe metadata)"
    ;;
  *) exit 0 ;;
esac
[ -n "$cmd" ] || exit 0

# --- secret masking (command string is the only free-text we store) ------
mask() {
  # A credential in tool output is usually QUOTED ("access_token": "…",
  # {'api_key':'…'}), and the keyword rule at the bottom cannot see it: it ends
  # the value at whitespace, and a JSON line has none, so it never even starts —
  # the character after the keyword is a quote, not one of `=:` or a space.
  # The two rules below mask the quoted run instead, one per quote character so
  # the closing quote can be required without a back-reference (POSIX ERE has
  # none). The value is escape-aware, or `password: "p@ss\"word"` would end at
  # the ESCAPED quote and leave `word"` in the clear.
  #
  # A closing quote is REQUIRED, and that bound is the point. Without it the
  # value runs to end of line, which is F12's failure in a new place — a rule
  # that runs past its intended end. It bites here because these two hooks mask
  # whole Bash command strings and whole note bodies: `grep -n "token: " src/*.ts`
  # offers the string's CLOSING quote as an opening one, and everything after it
  # would be blanked. Requiring the close costs nothing, because an unterminated
  # value still falls through to the keyword rule below and is masked to
  # whitespace there, exactly as it was before these rules existed.
  #
  # That keyword rule is the fallback that keeps this change from ever masking
  # less than before. (It has changed once since this paragraph was first
  # written: its value class is dash-bounded now, for the marker reason below.
  # The test pins its SHIPPED spelling, not identity with an older one.)
  #
  # `Authorization: <scheme> <credential>` is TWO tokens, and that keyword rule
  # ends its value at the first whitespace: it eats the SCHEME word and leaves
  # the credential in the clear one space to the right of a `***MASKED***`
  # marker, which reads as a successful redaction. `Bearer` was the only shape
  # that escaped, because the dedicated rule above takes the token AFTER the
  # scheme -- which is also why `Bearer` is absent from the scheme list below:
  # that rule fires first, so nothing bearer-shaped ever reaches this one. The
  # rule takes the CREDENTIAL and leaves the scheme word standing, even though
  # the keyword rule below then masks that word too and the note ends up
  # carrying two markers side by side. Consuming the scheme here would read
  # better and measures WORSE: that rule ends its value at whitespace, so
  # `***MASKED***"` is a single token to it and the closing quote of a
  # `curl -H "<header>" <url>` goes with it. The value stops at a quote or a
  # comma as well as at whitespace, so that command keeps its closing quote and
  # its URL -- the same bound, and the same reason, as the quoted-run rules
  # above.
  #
  # The scheme is an ALLOWLIST, not `[A-Za-z][A-Za-z0-9-]*`. A general scheme
  # word turns this into "mask the second word after any keyword", which reaches
  # this note's own UNQUOTED frontmatter (`project:` / `repos: [...]` /
  # `tags: [...]`, masked value by value further down) and leaves the note
  # unparseable for a checkout named after a mask keyword. The list holds the
  # single-opaque-token schemes; an unlisted scheme is left exactly where the
  # keyword rule had it.
  #
  # What keeps the marker intact for the range is the VALUE CLASS, not an
  # address: the scheme rule below carries none. sed applies each `-e` in order
  # to the pattern space AS IT STANDS, so a substitution here can destroy the
  # text a LATER rule's ADDRESS is matched against -- and the PEM range below
  # is addressed on the BEGIN marker. A value class that can cross a five-dash
  # run takes `token: Basic <PEM BEGIN marker>` whole, the range never opens,
  # and the body lines that follow behind a `cat -n` / `> ` / `grep -n` prefix
  # are written out VERBATIM (the prefixed catch-all further down now takes
  # those lines too, which is why the tests silence it when they measure this
  # rule alone). Measured at 54 of 54 (6 scheme spellings x 3 prefixes x 3
  # marker placements) with the plain class, and 0 of 54 with the dash-bounded
  # one. Two things that do NOT fix it: excluding a leading `-` from the value
  # class closes only that one spelling, since a value of `X<PEM BEGIN marker>`
  # starts at `X` and swallows the marker anyway; and moving this rule below
  # the PEM rules disables it outright, because the keyword rule below has
  # already replaced the scheme word with `***MASKED***` by the time it would
  # run. A class that cannot cross `-----` is what holds, and it costs nothing:
  # on such a line the rule still masks the credential up to the dashes.
  #
  # The negated marker address survives on exactly TWO rules: the UNBOUNDED
  # double- and single-quoted halves above, whose value must run to the closing
  # quote and so cannot be dash-bounded -- they skip marker lines instead, and
  # their bounded twins cover those lines. (An earlier version of this
  # paragraph said the address sat on this rule; the tests that count the
  # addressed rules say two, and they are right.)
  #
  # RESIDUE, recorded here rather than left for the next reader to discover: a
  # PARAMETER-LIST scheme closes only PARTLY. A Digest header carries
  # `username=`, `realm=` and `response=` parameters with QUOTED values, and the
  # value class ends at the first `"`, so the `response=` hash stays readable
  # beside the marker -- the very shape this rule closes for the opaque schemes.
  # Dropping `,` from the value class does not help; what stops it is the quote.
  # OAuth 1.0a headers have the same shape. It is unchanged ground rather than
  # new ground (the keyword rule alone masked the scheme word and stopped in the
  # same place), and a test pins it so the marker is never read as more than it
  # is.
  #
  # A PEM key body often arrives with a LINE PREFIX, and the whole-line rule
  # at the bottom sees none of them: `cat -n` writes a line number and a TAB,
  # a quoted transcript writes `> `, `grep -n` writes `file:12:`. (A diff `+` is
  # NOT one of those: `+` is in the base64 alphabet, so a `+`-prefixed body line
  # was already whole-line base64 and was already masked. The tests pin both
  # halves of that, because the wrong half is easy to assume.) So the body is
  # masked INSIDE the marker range, where a long base64 run is key material
  # whatever precedes it on the line.
  #
  # That range SUBSTITUTES runs -- it never blanks a line, and that is the whole
  # design. The session-archive copy of this function masks an ASSEMBLED note
  # that already carries that hook's `~~~~~~` fences (the ops-logging copy masks
  # a command string and folds it to one line afterwards, so no fence of its own
  # is at stake there), and a range that replaces whole lines deletes a CLOSING
  # fence along with the key: blank an odd number of fence lines and the parity
  # of everything after them inverts, and the next tool result is read as
  # top-level prose -- untrusted output promoted to something the operator said.
  # A run substitution cannot reach a fence line at all (no `~` is in the run
  # class), so the note's structure survives whatever the range covers. That is
  # also what makes the bound below safe: with a blanking action, ANY bound -- a
  # line ceiling, a blank line, the very tilde run this range ends at -- can blank
  # a fence line somewhere other than the END marker, and invert the parity of
  # everything after it.
  #
  # The range ends at the END marker, or 100 lines after the BEGIN marker,
  # whichever comes first (2026-09-17; before that it also ended at the next
  # column-0 `~~~` run). The cap is the whole of the bound, and it is worth
  # saying what each half of the old bound did and why it went:
  #
  #   - The cap is a line count, not a fence. The renderer fences tool
  #     results, thinking and tool inputs, and writes assistant and user TEXT
  #     turns at top level; a marker planted in EITHER kind of turn now runs
  #     on through whatever follows -- the next fenced block included -- for
  #     up to 100 lines, substituting every 12+ character run on the way. That
  #     run class is NOT base64: it is every alphanumeric plus `+ / = \`, and
  #     `/` is a member, so a run does not stop at a path separator. It
  #     consumes ordinary twelve-letter words, whole hashes, and a whole
  #     absolute path as a SINGLE run (`/home/runner/work/repo/checkout`).
  #     What replaces them is `***MASKED***`, the same token a genuine
  #     redaction produces, so content destroyed this way reads as routine
  #     hygiene rather than as damage and prompts no one to look at it.
  #     Measured on a note of 30 synthetic `git log` lines and 3 repeated
  #     absolute paths: clean, 0 tokens masked and 30/30 SHAs and 3/3 paths
  #     kept; with ONE 31-byte marker planted, 91 masked and 0/30 and 0/3
  #     kept. That is the cost, taken deliberately -- this range is what masks
  #     a prefixed key body at all -- and the cap is what bounds it: a planted
  #     marker can spoil at most the 100 lines after it, not the rest of the
  #     note. It cannot cost structure, because a substitution never deletes a
  #     line. Tests pin both the reach and the cap.
  #   - The `~~~` terminator is gone because it was content-controlled. Content
  #     is fenced but emitted VERBATIM, so a column-0 tilde run in a tool
  #     result's OWN body closed the range early, and the address was `^~{3,}`,
  #     unanchored at its right end, so even a `~~~ label` the renderer scores
  #     as closing nothing closed it. Plant one between a key's own BEGIN line
  #     and its body and the range closed before the body started: 6 of 6 body
  #     lines leaked behind a `cat -n` prefix -- the construction an attacker
  #     picks, and a Critical review finding on the branch that shipped it.
  #     Removing it fails closed: a key's body is masked whatever the content
  #     around it says, and the prefixed catch-all below takes prefixed body
  #     lines outside any range besides. What was traded for that is the
  #     availability failure the terminator had bounded, and the cap bounds it
  #     instead -- at 100 lines rather than at a line the attacker chooses.
  #   - Why 100: an RSA-4096 key body is about 50 lines and an ed25519 key
  #     under 10, prefix or not, so a real key sits inside the cap with room.
  #     Armor that runs longer (a PGP MESSAGE carrying a file) is base64-only
  #     line by line, and the two whole-line catch-alls below take those lines
  #     with no range at all, so the cap costs it nothing THERE. It does cost
  #     one shape, and a test measures it rather than rounding it away: a
  #     single body longer than the cap behind a prefix the catch-alls do not
  #     admit (a diff `-`, an RSA-8192 body of about 107 lines) keeps its tail.
  #   - How the cap counts, and why it is not a sed range. The first spelling
  #     was nested, `/BEGIN/,+100{ /BEGIN/,/END/ {...} }`, and sed does not
  #     re-check a range's first address while the range is open: a BEGIN
  #     that fell inside a window already open -- the second of two keys in
  #     one `git diff`, or a real key after a bare marker the model quoted --
  #     did not restart the count, the window closed in the middle of that
  #     body, and its remaining `-`-prefixed lines were written out in the
  #     clear (change-scan F1 / F5 on this change, 2026-09-17). So the window
  #     is a COUNTER in the hold space instead: a BEGIN line sets it to `o`,
  #     unconditionally; while it reads `o` plus at most 100 `x`, the line is
  #     masked and one `x` is appended; an END line empties it. Every BEGIN
  #     restarts the 100 lines, and END still closes early, so the cap stays
  #     a ceiling on the reach and not a floor. The `x` command swaps pattern
  #     and hold space, which is why the rule below reads as a dance of
  #     swaps: the counter has to be in the pattern space to be tested, and
  #     the line has to be back there to be masked. Only POSIX sed is used --
  #     hold space, `{}` blocks and interval expressions -- and the CI runner
  #     (GNU) and the operator's shell (BSD) both run the tests, including a
  #     mutation that makes the reset conditional on a closed counter and
  #     watches the planted shape leak exactly the ten lines the scan named.
  #
  # Read the two copies separately here, because this rule replaces something
  # different in each. `archive-session.sh` gains it outright: no input it
  # masked before is masked less. `capture-command.sh` had a whole-line
  # BLANKING range over the same markers, and this trades in both directions
  # at once. It masks LESS inside a block: base64 runs shorter than 12
  # characters -- a final body line of 4 or 8, where roughly one key size in
  # eight lands, at most 6 bytes of the trailing DER field -- plus non-base64
  # header text such as `Proc-Type:` (the whole-line short-run rule below
  # takes the final line since 2026-09-17, so that residue is now runs UNDER
  # 12 that share a line with other text). It also masks MORE,
  # in two ways that are not small: it gains the whole-line base64 catch-all
  # it never had, and it removes an UNBOUNDED failure. The old range had no
  # terminator but the END marker, so under POSIX sed an unterminated marker
  # anywhere in a multi-line command blanked every REMAINING line of the
  # logged command -- a `grep` for the marker text destroyed all 501 lines of
  # a command carrying no key material at all. A run substitution cannot do
  # that, and the line cap gives the range a second way to close. That failure
  # was this copy's alone; the archive copy never had the mode, which is why
  # it is recorded here and not as a general note.
  #
  # The range deliberately does NOT end at a blank line: an RFC 1421 encrypted
  # key writes `Proc-Type:` / `DEK-Info:` headers, then a BLANK LINE, and only
  # then its body -- a blank-line bound would stop exactly where the key material
  # starts.
  #
  # The whole-line rule below stays LAST: it is the catch-all for a body
  # pasted without its markers. `archive-session.sh` already carried it and
  # it is byte-identical there; `capture-command.sh` gains it here. Nothing
  # added above can make it MASK less than before. It does FIRE less often --
  # the in-range rule pre-empts it on lines it would have blanked -- and a
  # pre-empted line is masked just as completely, though not byte-for-byte:
  # this rule replaces the WHOLE line and so drops any surrounding
  # whitespace, where a run substitution keeps it, leaving `  ***MASKED***  `
  # rather than `***MASKED***`. Measured across every whitespace shape this
  # rule accepts, that retained whitespace is the ONLY residue, and no key
  # material survives on either path. The distinction is drawn here rather
  # than left for a reader to trip over.
  #
  # Two rules were added on 2026-09-17 for the two residues measured behind a
  # PREFIX (a `cat -n` number and TAB, a `> ` quote, a `grep -n` file:12:):
  #   * in range, a line that is NOTHING but an optional prefix and a run of 1-11
  #     characters is masked whole. That is a key body's short final line
  #     (`Zg==`), which the 12+ rule leaves as residue. The rule is deliberately
  #     WHOLE-LINE: a run of 1-11 characters embedded in prose stays, because the
  #     range also reaches prose when a marker is planted in an unfenced turn,
  #     and masking the last word of every reached line is the availability
  #     failure this file spent its history avoiding.
  #   * outside any range, a line that is only a prefix and a 32+ run has the
  #     run masked -- the prefixed twin of the bare catch-all below it. This is
  #     what closes the tilde construction above: a `~~~` planted between BEGIN
  #     and the body closes the range, and the body lines then fall to this rule
  #     instead of surviving behind their prefix. Its cost is the prefixed
  #     64-hex line (a `cat -n` over a shasum listing), masked like the bare one.
  #     The action is ANCHORED to the captured prefix, not `s/<run>/.../`:
  #     `/` is in the run class, so a `grep -n` path that is itself a 32+
  #     run of the class (`/home/runner/work/vaultkeys/vaultkeys/id:12:...`)
  #     would be the leftmost match, and the body after `:12:` would be
  #     written out in the clear while the line reads as masked (change-scan
  #     finding on this change, 2026-09-17; pinned with the anchoring taken out).
  sed -E \
    -e 's/gh[pousr]_[A-Za-z0-9]{20,}/***MASKED***/g' \
    -e 's/github_pat_[A-Za-z0-9_]{20,}/***MASKED***/g' \
    -e 's#(://[^/:@[:space:]]+):[^/@[:space:]]+@#\1:***MASKED***@#g' \
    -e 's/([Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+)([^[:space:]-]|-{1,4}[^[:space:]-])+/\1***MASKED***/g' \
    -e "/-----(BEGIN|END) ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/!s/((token|key|secret|password|pat|authorization|bearer)['\"]?[=:[:space:]]+\")([^\"\\\\]|\\\\.)*\"/\1***MASKED***\"/Ig" \
    -e "s/((token|key|secret|password|pat|authorization|bearer)['\"]?[=:[:space:]]+\")([^\"\\\\-]|\\\\.|-{1,4}([^\"\\\\-]|\\\\.))*-{0,4}\"/\1***MASKED***\"/Ig" \
    -e "/-----(BEGIN|END) ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/!s/((token|key|secret|password|pat|authorization|bearer)['\"]?[=:[:space:]]+')([^'\\\\]|\\\\.)*'/\1***MASKED***'/Ig" \
    -e "s/((token|key|secret|password|pat|authorization|bearer)['\"]?[=:[:space:]]+')([^'\\\\-]|\\\\.|-{1,4}([^'\\\\-]|\\\\.))*-{0,4}'/\1***MASKED***'/Ig" \
    -e "s/((token|key|secret|password|pat|authorization|bearer)[=:[:space:]]+(Basic|Digest|Token|ApiKey|OAuth|SSWS)[[:space:]]+)([^[:space:],\"'-]|-{1,4}[^[:space:],\"'-])+/\1***MASKED***/Ig" \
    -e 's/((token|key|secret|password|pat|authorization|bearer)[=:[:space:]]+)([^[:space:]-]|-{1,4}[^[:space:]-])+/\1***MASKED***/Ig' \
    -e '/-----BEGIN ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/{x;s/.*/o/;x;}' \
    -e 'x;/^ox{0,100}$/{s/$/x/;x;s/[A-Za-z0-9+\/=]{12,}/***MASKED***/g;s/^([[:space:]]*([0-9]+[[:space:]]*[|:>]?[[:space:]]*|[>|]+[[:space:]]*|[^[:space:]:]+:[0-9]+:[[:space:]]*)?)[A-Za-z0-9+\/=]{1,11}[[:space:]]*$/\1***MASKED***/;/-----END ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/{x;s/.*//;x;};x;};x' \
    -e 's/-----BEGIN ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/***MASKED***/g' \
    -e 's/-----END ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/***MASKED***/g' \
    -e 's/AKIA[0-9A-Z]{16}/***MASKED***/g' \
    -e 's/sk-[A-Za-z0-9_-]{20,}/***MASKED***/g' \
    -e 's/AIza[0-9A-Za-z_-]{35}/***MASKED***/g' \
    -e 's/xox[baprs]-[A-Za-z0-9-]{10,}/***MASKED***/g' \
    -e '/^[[:space:]]*[A-Za-z0-9+\/=]{32,}[[:space:]]*$/s/.*/***MASKED***/' \
    -e '/^[[:space:]]*([0-9]+[[:space:]]*[|:>]?[[:space:]]*|[>|]+[[:space:]]*|[^[:space:]:]+:[0-9]+:[[:space:]]*)[A-Za-z0-9+\/=]{32,}[[:space:]]*$/s/^([[:space:]]*([0-9]+[[:space:]]*[|:>]?[[:space:]]*|[>|]+[[:space:]]*|[^[:space:]:]+:[0-9]+:[[:space:]]*))[A-Za-z0-9+\/=]{32,}([[:space:]]*)$/\1***MASKED***\3/'
}
# ⭐ 1 行の byte 上限。⛔ 上限が要る理由は可読性ではなく【リポの成長】である:
#    実測 2026-09-09 — 5,004 行のうち 2,000 B を超えるのは 357 行 (7.1%) だけだが、
#    その 357 行が全体 3,350,112 B の 43% を占めていた。日付ファイルはコミットの
#    たびに丸ごと新しい blob として積まれるので、長い行は二次的に効く。
#    ⇒ 2,000 B で切ると 3,350,112 B → 約 2,622,773 B (22% 減)。中央値は 322 B
#      なので、⭕ ほとんどの行は無傷のまま通る。
# ⛔ byte で切ると UTF-8 の途中で切れる。⭐ iconv -c で末尾の不完全な列を落とすので、
#    この関数は locale に依存しない (実測: C / en_US.UTF-8 / ja_JP.UTF-8 の 3 つで
#    同じ出力・いずれも妥当な UTF-8)。⛔ ${s:0:N} は locale で文字/byte が入れ替わる。
# ⚠️ 失うもの: 長いコマンドの末尾。⭐ 省略した byte 数を必ず書き残すので、
#    「短いコマンドだった」と「切られた」を後から区別できる。
CLIP_BYTES="${OPS_LOG_CLIP_BYTES:-2000}"
# ⭐ intent の取り分は予算の 1/4 まで。⛔ 先に intent を丸ごと確保すると、実測で
#    9,154 B の intent が在るため cmd が潰れる — コマンドが主記録なので本末転倒。
#    実測 2026-09-09 (6,056 行): intent の中央値 24 B / 90%点 86 B ⇒ ⭕ 実運用では
#    ほぼ全額が cmd に回る。
INTENT_BYTES=$(( CLIP_BYTES / 4 ))
clip() {
  s=$1; max=$2
  n=$(printf '%s' "$s" | wc -c | tr -d ' ')
  [ "$n" -le "$max" ] && { printf '%s' "$s"; return; }
  # ⛔ iconv は末尾が不完全な列だと rc=1 を返す (実測)。この hook は `set -euo pipefail`
  #    の下で走るので、許容しないと【代入ごと】落ちて行が無音で消える。head -c が
  #    早くパイプを閉じると printf が SIGPIPE で 141 になるのも同じ。⇒ 明示的に飲む。
  #    ⭕ 「hook はツール実行をブロックしない」は ops-logging のハードルール。
  head=$(printf '%s' "$s" | head -c "$max" | iconv -f UTF-8 -t UTF-8 -c 2>/dev/null || true)
  printf '%s …(%s B 省略)' "$head" "$((n - max))"
}

# ⭐ 予算は【行 1 本】に対して掛ける。⛔ フィールドごとに掛けると、両方が上限に
#    達した行が上限の 2 倍になる (実測: 2,000 B を超える intent が 26 行 実在)。
intent_masked="$(clip "$(printf '%s' "$intent" | mask | tr '\n' ' ')" "$INTENT_BYTES")"
intent_n=$(printf '%s' "$intent_masked" | wc -c | tr -d ' ')
cmd_masked="$(clip "$(printf '%s' "$cmd"    | mask | tr '\n' ' ')" "$(( CLIP_BYTES - intent_n ))")"

# --- route to <origin_repo>/<date>.md ------------------------------------
# Every repo logs into its OWN folder, named after the origin repo — created
# on first command. Prefer the git repository root's name (correct even when
# cwd is a subdirectory); fall back to the cwd basename. No catch-all bucket.
repo_root="$(git -C "${cwd:-.}" rev-parse --show-toplevel 2>/dev/null || true)"
repo="$(basename "${repo_root:-${cwd:-}}" 2>/dev/null || true)"
# Sanitize to a single safe path segment: keep [A-Za-z0-9._-], everything else
# becomes '-'. Strip leading '-'/'.' so the name can't look like a git flag or
# resolve to '.'/'..'; empty result falls back to a fixed bucket.
repo="$(printf '%s' "$repo" | tr -c 'A-Za-z0-9._-' '-')"
while [ "${repo#[-.]}" != "$repo" ]; do repo="${repo#[-.]}"; done
[ -n "$repo" ] || repo='unknown'
branch="$(git -C "${cwd:-.}" branch --show-current 2>/dev/null || echo '-')"
[ -n "$branch" ] || branch='-'
date="$(date +%Y-%m-%d)"
dir="$LOG_REPO/$repo"
file="$dir/$date.md"
mkdir -p "$dir"

# Frontmatter + table header once per file.
if [ ! -f "$file" ]; then
  {
    printf -- '---\n'
    printf 'date: %s\n' "$date"
    printf 'target_repo: %s\n' "$repo"
    printf 'branch: %s\n' "$branch"
    printf 'tags: [git, gh, shell]\n'
    printf -- '---\n\n'
    printf '# %s — %s command log\n\n' "$date" "$repo"
    printf '| time | branch | command | intent |\n'
    printf '|---|---|---|---|\n'
  } >> "$file"
fi

esc() { printf '%s' "$1" | sed 's/|/\\|/g'; }   # escape pipes for the md table
printf '| %s | %s | `%s` | %s |\n' \
  "$(date +%H:%M:%S)" "$(esc "$branch")" "$(esc "$cmd_masked")" "$(esc "$intent_masked")" \
  >> "$file"

exit 0
