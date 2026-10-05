#!/usr/bin/env bash
# ops-logging PostToolUse hook.
# Append "command + intent" of a git / shell / GitHub(MCP) action to the
# terminal-ops-logs repo. Records COMMAND + INTENT ONLY — stdout is never read,
# and token/credential patterns in the command string are masked by mask()
# below (see its comments for the shapes it names and the ones it does not).
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
  # Every original record runs through all masking rules. A separate F frame
  # carries only a pre-masked syntax skeleton, never original credential text.
  # Final weaving preserves structural fence prefixes/runs while replacing all
  # fence info with MASK. Inline marker runs receive ordinary masking throughout.
  # When CR counts survive, only structural physical segments use the skeleton;
  # if a legacy rule consumed CRs, use the full skeleton to restore structure.
  # That fallback also masks neighboring non-fence segments on the same LF record.
  # Probe a fixed nonsecret whitespace set with the same ambient sed class
  # used by the old cleanup. The C-byte scanner then preserves that locale
  # classification with bounded UTF-8 lookahead instead of widening the class.
  # The same fixed probe preserves ambient case folds for the Unicode letters
  # equivalent to ASCII s/i/k on some supported locales; N input stays untouched.
  local legacy_profile
  legacy_profile="$(printf 'w\302\205\nw\302\240\nw\341\232\200\nw\341\240\216\nw\342\200\200\nw\342\200\201\nw\342\200\202\nw\342\200\203\nw\342\200\204\nw\342\200\205\nw\342\200\206\nw\342\200\207\nw\342\200\210\nw\342\200\211\nw\342\200\212\nw\342\200\250\nw\342\200\251\nw\342\200\257\nw\342\201\237\nw\343\200\200\nw\357\273\277\ns:\305\277\ni:\304\261\ni:\304\260\nk:\342\204\252\n' | sed -En -e 's/^w([[:space:]])$/w\1/p' -e 's/^s:(s)$/s:\1/Ip' -e 's/^i:(i)$/i:\1/Ip' -e 's/^k:(k)$/k:\1/Ip')"
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
  #
  # Shapes added on 2026-09-24, each a residue a scan named against this
  # function (A-47 = the 2026-09-18 change-scan F3, and the 2026-09-19
  # whole-repo scan's F6), and two that were tried and taken out again:
  #   * NOT here: the CHECK-THEN-APPEND value (`grep -q "token: " f || echo
  #     "token: V" >> f`, the 2026-09-18 change scan's F2) and YAML's doubled
  #     apostrophe (`'pre''fix'`, #186). Every rule tried for either one ran
  #     BEFORE the keyword fallback, and every one of them reached a credential
  #     the old rules masked, because a rule that runs before the fallback can
  #     move where the fallback's value ends or erase a keyword label the
  #     fallback would have read: a quote-bounded keyword pass (unanchored it
  #     took the `key` inside `--key` and ate the next `token:`; anchored on a
  #     quote it still ate `key:` after `"secret `), `''` in the single-quoted
  #     class (a stray `'` carried a value into the next one), and a
  #     continuation after a masked value (its escape alternative deleted an
  #     escaped space). Owner decision on this change took all of them out;
  #     they are tracked for a change of their own (#232, #186).
  #     #186 is now handled (#232): one rule reads a masked value that sits
  #     right after an opening single quote and is followed by `''` and the
  #     rest of the scalar (`'***MASKED***''fix'`), and masks it to the closing
  #     quote. The opening quote is required: a marker another rule left
  #     (`curl -u user:***MASKED***'' https://x; echo 'done'`) is followed by a
  #     shell's empty `''`, not a YAML escape, and without it the rule deleted
  #     everything up to the next apostrophe (Codex on #250). It runs AFTER every rule that reads a value -- right
  #     after the quoted passwd / passphrase rules -- so it can only mask more,
  #     and it reads no backslash escapes (YAML single quotes have none). The
  #     same rule placed before the keyword fallback, where the tried
  #     continuation ran, turned about 1,050 of 4,000 random lines per seed
  #     into regressions against main; placed last, none.
  #   * `PGP PRIVATE KEY BLOCK` and the RFC 4716 armor
  #     (`---- BEGIN SSH2 ENCRYPTED PRIVATE KEY ----`, four dashes and spaces)
  #     never opened the range. The first is broken by the keyword rules before
  #     the range's address sees it -- `KEY BLOCK` reads as keyword, separator,
  #     value -- and the second is not five-dash armor at all. Widening the
  #     shared marker regex is NOT the fix: that regex is also the line-skip
  #     address on the unbounded quoted halves, and widening it makes them skip
  #     a one-line JSON value that carries the PGP armor, which they mask whole
  #     today. So these two armors get their own pair of rules instead: the
  #     FIRST rule of all opens the window on the untouched line, before any
  #     keyword rule can reach the marker, and a rule right after the in-range
  #     body rule closes it on the END line, in the spelling the keyword rules
  #     leave behind as well as the original. The shared regex, and so the
  #     address, are unchanged.
  #   * A diff `-` joins the prefixes the prefixed catch-all admits, so a
  #     `-`-prefixed 32+ body line is masked outside any range too. That is the
  #     prefix the scan named, and it is also what the line cap used to cost: a
  #     single body longer than 100 lines behind a `-` kept its tail.
  #   * Credentials in an ARGUMENT position, which no keyword precedes: a
  #     `-u` / `--user` value that joins a name and a secret with a colon, a
  #     MySQL-family client's `-p` with the password attached, and
  #     `redis-cli`'s `-a` / `--pass`. Each is anchored on its flag (the `-p`
  #     and `-a` rules on the client's name too, because `-p` alone is
  #     `mkdir -p`, `cp -pR` and `ssh -p2222`), and each value class is the
  #     dash-bounded one, so none of them can take a marker. The words allowed
  #     between the client's name and its flag are capped at twelve: an
  #     unbounded word run let every start on a line of repeated client names
  #     scan to the end before failing, quadratic under glibc's regex (the
  #     change scan's F1; BSD sed stays linear either way). `passwd` and
  #     `passphrase` (`--passphrase V`, `--passphrase=V`) get a rule of
  #     their OWN, right after the keyword fallback, and are NOT in
  #     the shared alternation: there, a leftmost match starting on them took
  #     the real keyword after them as their value -- `--passphrase --key S`,
  #     or a prompt's closing quote before `PASSWORD="a b c"` -- and the secret
  #     after it went to the log (the second change scan on this change, F2 /
  #     F3). Running after every older keyword rule, they only ever meet a
  #     value that is already masked. `auth` and `credential` do NOT join at
  #     all: they are subcommands
  #     (`gh auth status`, `git credential fill`) and the keyword rule would
  #     mask the word after them in every such command. Logs written before
  #     this change can still hold argument-position credentials in the clear.
  #     Other argument spellings (`openssl -passin pass:V`, `sshpass -p V`) are
  #     still unnamed.
  # What the series costs, measured 2026-09-24 by passing every tracked text file
  # here (101 files / 40,479 lines) and the live ops-log clone (70 files /
  # 27,487 lines), one whole-file stream each, through the old and new mask():
  # no line anywhere is masked LESS. What is masked MORE, beyond the shapes
  # above: the word after `passwd` / `passphrase` in prose (`a passwd entry`,
  # `a strong passphrase you choose`) -- the same cost `password` has always
  # had -- and, where a comment QUOTES one of the two new armors, the window's
  # usual reach: 12+ runs and short whole lines for up to 100 lines after it,
  # the planted-marker cost the range already carries for the shared marker.
  # Every rule these additions brought carries the address /```|~~~/! -- it
  # does not run on a line that holds a fence run at all. mask() runs over the
  # assembled note AFTER the fence balance of each turn has been decided, and a
  # backtick fence's info string cannot hold a backtick, so "```mysql -p`x`" is
  # not a fence until a rule deletes the backticks and leaves
  # "```mysql -p***MASKED***", which is (the second change scan's F1). Keeping
  # the two characters out of the value classes was tried first and was not
  # enough -- an escape alternative still took "\`" (the third scan's F1) --
  # and it cost the other direction: a value holding either character was
  # masked only up to it (its F2). On any OTHER line a substitution cannot make
  # a fence run, because it always inserts `***MASKED***`, never nothing, so
  # it cannot join two backtick runs into one. The cost is the line guard's
  # usual one: a credential on a line that also holds a fence run is left to
  # the older rules, as it was before this change. The older keyword rules
  # share the deletion root and are left as they were here; it is tracked on
  # its own.
  # An over-reach the measurement caught was fixed rather than accepted: the
  # `-u` rule read `date -u '+%Y-%m-%dT%H:%M'` as a name and a secret, so its
  # name class excludes `%` and `+`.
  # REPEATED -p / -a flags (#234, 2026-09-25): of `mysql -p<A> -p<B>` only the
  # last value was masked -- the greedy word run before the flag swallowed the
  # earlier ones. The two rules now run in a loop (`:m` ... `tm`, `:r` ...
  # `tr`) until they match nothing, and each pass is global, so a pass masks
  # the last reachable flag in every client's window at once and the passes
  # are bounded by the flags in one twelve-word window, not by the line. (A
  # pass WITHOUT `g` restarts at the start of the line once per occurrence:
  # measured on a first spelling, 4,000 `redis-cli -a x` on one line took
  # 195 s under BSD sed; a test pins the ratio.)
  # Inside the loops a value is replaced by a SENTINEL, `***MASKEDP***`, not by
  # the marker, and a value that begins with the sentinel is not a value to
  # the looped rules -- that is what stops a pass from matching its own output:
  # a masked occurrence is backtracked past and the next pass reaches the one
  # before it. One rule right after the second loop turns every sentinel into
  # `***MASKED***`, so the sentinel never leaves mask(); it is delimited with
  # `#` so the shared redactor's lift of `s/<shape>/***MASKED***/g` rules does
  # not read the sentinel as a secret shape. The exclusion is
  # spelled as the complement of the sentinel's one prefix (a leading part of
  # it, then any other character), so `***abc` and `*abc` are values. It is not
  # keyed on the marker itself: the earlier token rules (`sk-`, `AKIA`, ...)
  # and the in-range run rule leave `***MASKED***` at the START of a value
  # with the rest of it after, and an exclusion on the marker left that rest
  # in the clear where main masked it (change scan r4 on #243, reproduced).
  # That complement needs a character after the leading part, so a value
  # that IS a leading part (`*`, `**`), optionally followed by dashes (`*-`,
  # `***M-`), matched nothing (Codex P1 on #243); a rule of its own inside each
  # loop takes such a value when it ends there, and re-emits what ended it. It
  # cannot take the sentinel, which is thirteen characters and is not followed
  # by a dash. A run of five or more dashes also ends it (`-p*-----`,
  # `-p***MASK-----rest`): the value class cannot cross that run, so without
  # it such a value matched nothing and came out whole, where main masked the
  # part before the run (Codex on the copies of #243, reproduced). The cost: a secret that begins with the literal `***MASKEDP***`
  # is not masked by the looped rules (pinned).
  # The word run and the value class are otherwise unchanged -- quoted values
  # with spaces and flag-shaped words inside quoted arguments are left for
  # #232, where a shell-aware reading of both was measured to need escapes, a
  # fallback for unclosed quotes and bounded quoted pieces.
  # A QUOTED passwd / passphrase value (#232: `--passphrase "a b"`,
  # `passwd: 'a b'`) is taken to its closing quote by one rule per quote,
  # placed AFTER the argument-position loops. A quoted rule can start at the
  # closing quote of a label (`"Enter passwd: "`) and take everything up to
  # the next quote; placed before the `-u`, mysql and redis-cli rules, that
  # span removed the flag or client name they key on, and the quoted
  # credential after it -- masked on main -- was written out (change scan F1 /
  # F2 on #232 a, reproduced). After every rule that reads a value, the span
  # can only mask more. The cost: text between a label's closing quote and
  # the next quote is masked too (`grep "passwd:"***MASKED***"...`).
  # #295: after the existing value/armor rules, a streaming tag pass precedes
  # #291s final physical-line substitution. Only a quote after an attribute =
  # opens a quoted value: earlier keyword rules may remove an entire first
  # attribute, including its opening quote. Quoted angles therefore remain in
  # the start tag. This stage retains LF/CR, including inside a value; the
  # final legacy fallback can remove CR within its own bounded match. The
  # syntax weave restores structural separators if that changes segment counts.
  # Each byte enters/leaves the tag array once. Name/attribute/value lookahead
  # strings have fixed maximum lengths; no suffix search or growing tag string
  # is repeated. A new unquoted < abandons an unfinished tag before restarting.
  # Fence metadata never resets markup or credential context. The final weave
  # restores syntax only after every rule has processed the complete input.
  # A start tag may span at most 32 physical lines including its opening line;
  # LF and bare CR advance the count, while CRLF advances it only once. At the
  # attempted 33rd line, emit the candidate unchanged and resume ordinary scans.
  # Buffer a candidate tag, plus direct body text only to its physical line
  # end or next <. A matching close is required before masking that body text.
  # An unrelated next tag or an unclosed element therefore keeps following prose.
  # A tag without > is
  # emitted unchanged; the retained #291 final rule still covers malformed
  # quoted tags that it masked before. Earlier sed rules keep their order.
  # As in #291, all attribute names/values (including lang/class) are masked.
  # XML-like prose in a matching element is also masked. Its following word
  # stays intact. Appending one LF lets awk preserve the preceding sed
  # streams separators, including its final LF choice. C locale scans bytes.
  # Split each physical record once. BSD awk can rescan a complete string inside
  # every substr($0, i, 1), even when length($0) is cached, making long lines
  # quadratic. Empty-separator split is supported by the required GNU/macOS
  # awk implementations (and mawk); no broader POSIX portability is assumed.
  # Normalize NUL before any awk can truncate the rest of its input record.
  # A visible non-whitespace ? keeps both sides without joining fields or
  # turning fence info into closing whitespace. The archive renderer and
  # refence comparisons apply the same mapping before measuring structure.
  { LC_ALL=C tr '\000' '?'; printf '\n'; } | LC_ALL=C awk '
    # Byte widths for the renderer closing-whitespace union. Only a wholly
    # whitespace suffix is structural; a partial UTF-8 sequence never matches.
    function blank_width(at, last,    c, pair, triple) {
      if (at > last) return 0
      c = original_bytes[at]
      if (c == " " || c == "\t" || c == "\013" || c == "\014") return 1
      if (at + 1 > last) return 0
      pair = c original_bytes[at + 1]
      if (pair == "\302\205" || pair == "\302\240") return 2
      if (at + 2 > last) return 0
      triple = pair original_bytes[at + 2]
      if (triple == "\341\232\200" || triple == "\341\240\216" || triple == "\342\200\250" || triple == "\342\200\251" || triple == "\342\200\257" || triple == "\342\201\237" || triple == "\343\200\200" || triple == "\357\273\277") return 3
      if (pair == "\342\200" && index("\200\201\202\203\204\205\206\207\210\211\212", original_bytes[at + 2])) return 3
      return 0
    }
    function structural(first, last,    j, c, count, container, spaces, end, width, k) {
      j = first
      while (j <= last) {
        c = original_bytes[j]
        width = blank_width(j, last)
        if (width) {
          if (!container) { if (c != " ") spaces = 4; else spaces++ }
          j += width; continue
        }
        if (c == ">") { container = 1; j++; continue }
        if (c ~ /^[-+*]$/ && blank_width(j + 1, last)) {
          container = 1; j++; continue
        }
        if (c ~ /^[0-9]$/) {
          end = j
          while (end <= last && end - j < 10 && original_bytes[end] ~ /^[0-9]$/) end++
          if (end - j <= 9 && original_bytes[end] ~ /^[.)]$/ && blank_width(end + 1, last)) {
            container = 1; j = end + 1; continue
          }
        }
        if (c != "`" && c != "~") return 0
        end = j
        while (end <= last && original_bytes[end] == c) end++
        if (end - j < 3 || (!container && spaces > 3)) return 0
        # Invalid backtick info is ordinary text. Legacy masking and the
        # archive refence pass must still process any newly valid result.
        if (c == "`") {
          for (k = end; k <= last; k++) if (original_bytes[k] == "`") return 0
        }
        prefix_end = end - 1
        marker = c
        return 1
      }
      return 0
    }
    function skeleton(first, last, protected, end, tick,    j, marked, closing, width) {
      printf "%s", protected ? "P" : "N"
      closing = protected
      for (j = end + 1; protected && j <= last; j += width) {
        width = blank_width(j, last)
        if (!width) { closing = 0; break }
      }
      for (j = first; j <= last; j++) {
        if (protected && (j <= end || closing)) {
          # Ordered-list ordinals may themselves be credential digits. Emit
          # constant zeroes of the same width, never those original digits.
          printf "%s", (j <= end && original_bytes[j] ~ /^[0-9]$/ ? "0" : original_bytes[j])
          marked = 0
        } else if (!marked) { printf "%s", "***MASKED***"; marked = 1 }
      }
    }
    {
      n = split($0, original_bytes, "")
      segments = 0; any_fence = 0; first = 1
      for (i = 1; i <= n + 1; i++) {
        if (i <= n && original_bytes[i] != "\r") continue
        segments++
        starts[segments] = first; ends[segments] = i - 1
        prefix_end = first - 1; marker = ""
        selected[segments] = structural(first, i - 1)
        prefixes[segments] = prefix_end; markers[segments] = marker
        any_fence = any_fence || selected[segments]
        first = i + 1
      }
      printf "F%d", any_fence
      if (any_fence) {
        for (i = 1; i <= segments; i++) {
          if (i > 1) printf "\r"
          skeleton(starts[i], ends[i], selected[i], prefixes[i], markers[i])
        }
      }
      # Terminate every framed record; BSD sed must not create an extra record
      # by completing a missing final LF. The input sentinel tracks EOF instead.
      printf "\nN%s\n", $0
    }
  ' |
  sed -E \
    -e '/^F/b' \
    -e 's/^N//' \
    -e '/-----BEGIN PGP PRIVATE KEY BLOCK-----|---- BEGIN SSH2 ENCRYPTED PRIVATE KEY ----/{x;s/.*/o/;x;}' \
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
    -e "/\`\`\`|~~~/!s/((passwd|passphrase)[=:[:space:]]+)([^[:space:]\"'-]|-{1,4}[^[:space:]\"'-])+/\1***MASKED***/Ig" \
    -e '/-----BEGIN ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/{x;s/.*/o/;x;}' \
    -e 'x;/^ox{0,100}$/{s/$/x/;x;s/[A-Za-z0-9+\/=]{12,}/***MASKED***/g;s/^([[:space:]]*([0-9]+[[:space:]]*[|:>]?[[:space:]]*|[>|]+[[:space:]]*|[^[:space:]:]+:[0-9]+:[[:space:]]*)?)[A-Za-z0-9+\/=]{1,11}[[:space:]]*$/\1***MASKED***/;/-----END ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/{x;s/.*//;x;};x;};x' \
    -e '/-----END PGP PRIVATE KEY (BLOCK|\*\*\*MASKED\*\*\*)-----|---- END SSH2 ENCRYPTED PRIVATE KEY ----/{x;s/.*//;x;}' \
    -e 's/-----BEGIN ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/***MASKED***/g' \
    -e 's/-----END ([A-Z0-9 ]*PRIVATE KEY|PGP MESSAGE)-----/***MASKED***/g' \
    -e 's/AKIA[0-9A-Z]{16}/***MASKED***/g' \
    -e 's/sk-[A-Za-z0-9_-]{20,}/***MASKED***/g' \
    -e 's/AIza[0-9A-Za-z_-]{35}/***MASKED***/g' \
    -e 's/xox[baprs]-[A-Za-z0-9-]{10,}/***MASKED***/g' \
    -e "/\`\`\`|~~~/!s/((^|[[:space:]])(-u|--user)(=|[[:space:]]+)[\"']?[^[:space:]:\"'/%+]+:)([^[:space:]\"'-]|-{1,4}[^[:space:]\"'-])+/\1***MASKED***/g" \
    -e ':m' \
    -e "/\`\`\`|~~~/!s/((mysql|mysqldump|mysqladmin|mariadb|mariadb-dump)([[:space:]]+[^[:space:]|;&]+){0,12}[[:space:]]+-p[\"']?)([^[:space:]\"'*-]|\\*[^[:space:]\"'*-]|\\*-{1,4}[^[:space:]\"'-]|\\*\\*[^[:space:]\"'*-]|\\*\\*-{1,4}[^[:space:]\"'-]|\\*\\*\\*[^[:space:]\"'M-]|\\*\\*\\*-{1,4}[^[:space:]\"'-]|\\*\\*\\*M[^[:space:]\"'A-]|\\*\\*\\*M-{1,4}[^[:space:]\"'-]|\\*\\*\\*MA[^[:space:]\"'S-]|\\*\\*\\*MA-{1,4}[^[:space:]\"'-]|\\*\\*\\*MAS[^[:space:]\"'K-]|\\*\\*\\*MAS-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASK[^[:space:]\"'E-]|\\*\\*\\*MASK-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKE[^[:space:]\"'D-]|\\*\\*\\*MASKE-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKED[^[:space:]\"'P-]|\\*\\*\\*MASKED-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKEDP[^[:space:]\"'*-]|\\*\\*\\*MASKEDP-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKEDP\\*[^[:space:]\"'*-]|\\*\\*\\*MASKEDP\\*-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKEDP\\*\\*[^[:space:]\"'*-]|\\*\\*\\*MASKEDP\\*\\*-{1,4}[^[:space:]\"'-]|-{1,4}[^[:space:]\"'-])([^[:space:]\"'-]|-{1,4}[^[:space:]\"'-])*/\1***MASKEDP***/g" \
    -e "/\`\`\`|~~~/!s/((mysql|mysqldump|mysqladmin|mariadb|mariadb-dump)([[:space:]]+[^[:space:]|;&]+){0,12}[[:space:]]+-p[\"']?)(\\*|\\*\\*|\\*\\*\\*|\\*\\*\\*M|\\*\\*\\*MA|\\*\\*\\*MAS|\\*\\*\\*MASK|\\*\\*\\*MASKE|\\*\\*\\*MASKED|\\*\\*\\*MASKEDP|\\*\\*\\*MASKEDP\\*|\\*\\*\\*MASKEDP\\*\\*)-{0,4}([[:space:]\"'|;&]|-----|$)/\1***MASKEDP***\5/g" \
    -e 'tm' \
    -e ':r' \
    -e "/\`\`\`|~~~/!s/(redis-cli([[:space:]]+[^[:space:]|;&]+){0,12}[[:space:]]+(-a|--pass)[[:space:]]+[\"']?)([^[:space:]\"'*-]|\\*[^[:space:]\"'*-]|\\*-{1,4}[^[:space:]\"'-]|\\*\\*[^[:space:]\"'*-]|\\*\\*-{1,4}[^[:space:]\"'-]|\\*\\*\\*[^[:space:]\"'M-]|\\*\\*\\*-{1,4}[^[:space:]\"'-]|\\*\\*\\*M[^[:space:]\"'A-]|\\*\\*\\*M-{1,4}[^[:space:]\"'-]|\\*\\*\\*MA[^[:space:]\"'S-]|\\*\\*\\*MA-{1,4}[^[:space:]\"'-]|\\*\\*\\*MAS[^[:space:]\"'K-]|\\*\\*\\*MAS-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASK[^[:space:]\"'E-]|\\*\\*\\*MASK-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKE[^[:space:]\"'D-]|\\*\\*\\*MASKE-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKED[^[:space:]\"'P-]|\\*\\*\\*MASKED-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKEDP[^[:space:]\"'*-]|\\*\\*\\*MASKEDP-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKEDP\\*[^[:space:]\"'*-]|\\*\\*\\*MASKEDP\\*-{1,4}[^[:space:]\"'-]|\\*\\*\\*MASKEDP\\*\\*[^[:space:]\"'*-]|\\*\\*\\*MASKEDP\\*\\*-{1,4}[^[:space:]\"'-]|-{1,4}[^[:space:]\"'-])([^[:space:]\"'-]|-{1,4}[^[:space:]\"'-])*/\1***MASKEDP***/g" \
    -e "/\`\`\`|~~~/!s/(redis-cli([[:space:]]+[^[:space:]|;&]+){0,12}[[:space:]]+(-a|--pass)[[:space:]]+[\"']?)(\\*|\\*\\*|\\*\\*\\*|\\*\\*\\*M|\\*\\*\\*MA|\\*\\*\\*MAS|\\*\\*\\*MASK|\\*\\*\\*MASKE|\\*\\*\\*MASKED|\\*\\*\\*MASKEDP|\\*\\*\\*MASKEDP\\*|\\*\\*\\*MASKEDP\\*\\*)-{0,4}([[:space:]\"'|;&]|-----|$)/\1***MASKEDP***\5/g" \
    -e 'tr' \
    -e 's#\*\*\*MASKEDP\*\*\*#***MASKED***#g' \
    -e "/\`\`\`|~~~/!s/((passwd|passphrase)['\"]?[=:[:space:]]+\")([^\"\\\\-]|\\\\.|-{1,4}([^\"\\\\-]|\\\\.))*-{0,4}\"/\1***MASKED***\"/Ig" \
    -e "/\`\`\`|~~~/!s/((passwd|passphrase)['\"]?[=:[:space:]]+')([^'\\\\-]|\\\\.|-{1,4}([^'\\\\-]|\\\\.))*-{0,4}'/\1***MASKED***'/Ig" \
    -e "/\`\`\`|~~~/!s/'\\*\\*\\*MASKED\\*\\*\\*''([^']|'')*'/'***MASKED***'/g" \
    -e '/^[[:space:]]*[A-Za-z0-9+\/=]{32,}[[:space:]]*$/s/.*/***MASKED***/' \
    -e '/^[[:space:]]*([0-9]+[[:space:]]*[|:>]?[[:space:]]*|[>|]+[[:space:]]*|[^[:space:]:]+:[0-9]+:[[:space:]]*|-)[A-Za-z0-9+\/=]{32,}[[:space:]]*$/s/^([[:space:]]*([0-9]+[[:space:]]*[|:>]?[[:space:]]*|[>|]+[[:space:]]*|[^[:space:]:]+:[0-9]+:[[:space:]]*|-))[A-Za-z0-9+\/=]{32,}([[:space:]]*)$/\1***MASKED***\3/' \
    -e 's/^/N/' | LC_ALL=C awk '
    # Buffer one output record so its safe metadata stays paired even when a
    # multiline tag delays output. Store bounded chunks, not one cell per byte.
    # Each concatenation stops at 256 bytes (at most 267 after a MASK token).
    function flush_output(newline,    j) {
      if (output_chunk_size) output_bytes[++output_used] = output_chunk
      output_chunk = ""; output_chunk_size = 0
      printf "F%s\nN", frames[output_record]
      delete frames[output_record++]
      for (j = 1; j <= output_used; j++) { printf "%s", output_bytes[j]; delete output_bytes[j] }
      output_used = 0
      if (newline) printf "\n"
    }
    function emit(s) {
      if (s == "\n") flush_output(1)
      else {
        output_chunk = output_chunk s
        output_chunk_size += length(s)
        if (output_chunk_size >= 256) {
          output_bytes[++output_used] = output_chunk
          output_chunk = ""; output_chunk_size = 0
        }
      }
    }
    function flush_tag(    j) {
      for (j = 1; j <= used; j++) emit(tag[j])
      clear_tag()
    }
    function clear_tag(    j) {
      for (j = 1; j <= used; j++) delete tag[j]
      used = 0
      state = 0
    }
    function finish_value() {
      if (attr == "type" && value == "password") password_input = 1
      attr = value = ""
      attribute = 0
    }
    function attribute_byte(c) {
      if (quote != "") {
        if (c == quote) { quote = ""; finish_value() }
        else if (length(value) <= 8) value = value tolower(c)
        return
      }
      if (attribute == 3) {
        if (c ~ /[[:space:]]/) return
        value = ""
        if (c == "\042" || c == "\047") { quote = c; return }
        attribute = 4
      }
      if (attribute == 4) {
        if (c ~ /[[:space:]]/) finish_value()
        # Keep one byte past password/ so longer slash-bearing types cannot
        # truncate to the terminal form recognized by close_tag().
        else if (length(value) <= 9) value = value tolower(c)
        return
      }
      if (c == "=" && (attribute == 1 || attribute == 2)) { attribute = 3; return }
      if (c ~ /[[:space:]]/) { if (attribute == 1) attribute = 2; return }
      if (attribute != 1) { attr = ""; attribute = 1 }
      if (length(attr) <= 4) attr = attr tolower(c)
    }
    function close_tag(    j, marked) {
      # This function sees > itself. Only an unfinished bare type can treat
      # its final slash as />; quoted or whitespace-terminated values cannot.
      if (attribute == 4 && attr == "type" && value == "password/") password_input = 1
      if (attribute == 4) finish_value()
      if (!label && !password_input) { flush_tag(); return }
      for (j = 1; j <= name_end; j++) emit(tag[j])
      for (j = name_end + 1; j < used; j++) {
        if (tag[j] == "\n" || tag[j] == "\r") { emit(tag[j]); marked = 0 }
        else if (!marked && tag[j] ~ /[[:space:]]/) emit(tag[j])
        else if (!marked) { emit("***MASKED***"); marked = 1 }
      }
      emit(">")
      body = label && tag[used - 1] != "/"
      if (body) {
        body_name_length = name_end - 1
        for (j = 2; j <= name_end; j++) body_name[j - 1] = tolower(tag[j])
      }
      clear_tag()
    }
    function secret_component(s) {
      return s ~ /^(token|key|secret|password|passwd|passphrase|pat|authorization|bearer)$/
    }
    function finish_body(mask,    j, marked) {
      for (j = 1; j <= body_used; j++) {
        if (!mask || body_text[j] ~ /[[:space:]]/) emit(body_text[j])
        else if (!marked) { emit("***MASKED***"); marked = 1 }
        delete body_text[j]
      }
      for (j = 1; j <= body_name_length; j++) delete body_name[j]
      body_used = body_name_length = body = 0
    }
    function closing_byte(c,    j, ok) {
      closing[++closing_used] = c
      if (closing_used == 2) ok = c == "/"
      else if (closing_used <= body_name_length + 2) ok = tolower(c) == body_name[closing_used - 2]
      else if (c == ">") {
        finish_body(1)
        for (j = 1; j <= closing_used; j++) { emit(closing[j]); delete closing[j] }
        closing_used = 0
        return
      }
      else ok = c ~ /[[:space:]]/ && c != "\n" && c != "\r"
      if (ok) return
      finish_body(0)
      for (j = 1; j <= closing_used; j++) { byte(closing[j]); delete closing[j] }
      closing_used = 0
    }
    function byte(c) {
      if (body == 2) { closing_byte(c); return }
      if (body == 1) {
        if (c == "<") { body = 2; closing_used = 1; closing[1] = c; return }
        if (c == "\n" || c == "\r") finish_body(0)
        else { body_text[++body_used] = c; return }
      }
      if (state == 0) {
        if (c != "<") { emit(c); return }
        state = 1; used = 1; tag[used] = c; tag_line = logical_line
        tail = component = ""; namespaced = local_secret = 0
        return
      }
      if (state == 1) {
        if (c ~ /[A-Za-z0-9_.:-]/) {
          tag[++used] = c
          if (c == ":") { tail = component = ""; namespaced = 1; local_secret = 0 }
          else {
            if (length(tail) <= 13) tail = tail tolower(c)
            if (c ~ /[_.-]/) { local_secret = local_secret || secret_component(component); component = "" }
            else if (length(component) <= 13) component = component tolower(c)
          }
          return
        }
        label = local_secret || secret_component(component)
        if ((!label && (tail != "input" || namespaced)) || c !~ /[[:space:]\/>]/) {
          flush_tag()
          byte(c)
          return
        }
        name_end = used
        state = 2; quote = attr = value = ""; attribute = password_input = 0
      }
      if (quote == "" && c == "<") {
        flush_tag()
        byte(c)
        return
      }
      tag[++used] = c
      if (quote == "" && c == ">") { close_tag(); return }
      attribute_byte(c)
    }
    function flush_pending(    j) {
      finish_body(0)
      for (j = 1; j <= closing_used; j++) { emit(closing[j]); delete closing[j] }
      closing_used = 0
      flush_tag()
    }
    function feed(c) {
      if (c == "\r" || (c == "\n" && !previous_cr)) {
        logical_line++
        if (state && logical_line - tag_line >= 32) flush_tag()
      }
      previous_cr = c == "\r"
      byte(c)
    }
    BEGIN { output_record = 1 }
    /^F/ { frames[++input_record] = substr($0, 2); next }
    /^N/ {
      if (normal_records++) feed("\n")
      line_length = split($0, line_bytes, "")
      for (i = 2; i <= line_length; i++) feed(line_bytes[i])
    }
    END { flush_pending(); if (output_used || output_chunk_size) flush_output(0) }
  ' | sed -e '' | { cat; printf '\n'; } | LC_ALL=C CON295_LEGACY_PROFILE="$legacy_profile" awk '
    # Equivalent to the final legacy label cleanup, with a match-local budget.
    # Quotes are ordinary bytes here, just as in its old [^<>]* body.
    # Seal bounded chunks before CR segment indexes or the record end advance.
    # The same 256-byte threshold bounds concatenation independently of input.
    function seal_clean_chunk() {
      if (clean_chunk_size) clean_parts[++clean_used] = clean_chunk
      clean_chunk = ""; clean_chunk_size = 0
    }
    function legacy_emit(c) {
      if (c == "\r") {
        seal_clean_chunk()
        clean_ends[clean_segments] = clean_used
        clean_starts[++clean_segments] = clean_used + 1
      } else {
        clean_chunk = clean_chunk c
        clean_chunk_size += length(c)
        if (clean_chunk_size >= 256) seal_clean_chunk()
      }
    }
    function legacy_clear(    j) {
      for (j = 1; j <= legacy_used; j++) delete legacy_tag[j]
      legacy_used = legacy_state = legacy_cr = 0
      legacy_word = ""
    }
    function legacy_flush(    j) {
      for (j = 1; j <= legacy_used; j++) legacy_emit(legacy_tag[j])
      legacy_clear()
    }
    # Beyond 32 physical lines, stop parsing/buffering this legacy candidate.
    # Keep masking opaquely to <, > or record end, preserving CR separators.
    # An unclosed overbound candidate therefore loses the rest of that record.
    function legacy_opaque(    j) {
      for (j = 1; j <= legacy_prefix; j++) legacy_emit(legacy_tag[j])
      opaque_marked = 0
      for (j = legacy_prefix + 1; j <= legacy_used; j++) {
        if (legacy_tag[j] == "\r") { legacy_emit("\r"); opaque_marked = 0 }
        else if (!opaque_marked) { legacy_emit("***MASKED***"); opaque_marked = 1 }
      }
      legacy_clear()
      legacy_state = 4
    }
    function legacy_width(at, last,    c, pair, triple) {
      c = value_bytes[at]
      if (c ~ /^[[:space:]]$/) return 1
      if (at + 1 > last) return 0
      pair = c value_bytes[at + 1]
      if (pair in legacy_blanks) return 2
      if (at + 2 > last) return 0
      triple = pair value_bytes[at + 2]
      return (triple in legacy_blanks) ? 3 : 0
    }
    function legacy_letter(at, last,    c, pair, triple) {
      c = value_bytes[at]
      if (c ~ /^[A-Za-z]$/) { folded_letter = tolower(c); return 1 }
      if (at + 1 > last) return 0
      pair = c value_bytes[at + 1]
      if (pair in legacy_folds) { folded_letter = legacy_folds[pair]; return 2 }
      if (at + 2 > last) return 0
      triple = pair value_bytes[at + 2]
      if (triple in legacy_folds) { folded_letter = legacy_folds[triple]; return 3 }
      return 0
    }
    function legacy_cleanup(    i, j, c, n, width) {
      for (j = 1; j <= clean_used; j++) delete clean_parts[j]
      for (j = 1; j <= clean_segments; j++) { delete clean_starts[j]; delete clean_ends[j] }
      clean_used = 0; clean_segments = 1; clean_starts[1] = 1
      clean_chunk = ""; clean_chunk_size = 0
      n = split(value, value_bytes, "")
      for (i = 1; i <= n; i++) {
        c = value_bytes[i]
        if (legacy_state && legacy_state != 4 && c == "\r" && ++legacy_cr >= 32) legacy_opaque()
        if (c == "<") {
          legacy_flush()
          legacy_state = 1; legacy_used = 1; legacy_tag[1] = c
          continue
        }
        if (!legacy_state) { legacy_emit(c); continue }
        if (legacy_state == 4) {
          if (c == ">") { legacy_emit(c); legacy_clear() }
          else if (c == "\r") { legacy_emit(c); opaque_marked = 0 }
          else if (!opaque_marked) { legacy_emit("***MASKED***"); opaque_marked = 1 }
          continue
        }
        if (legacy_state == 1) {
          width = legacy_letter(i, n)
          if (width && length(legacy_word) < 13) {
            legacy_word = legacy_word folded_letter
            for (j = 0; j < width; j++) legacy_tag[++legacy_used] = value_bytes[i + j]
            i += width - 1
            continue
          }
          width = legacy_width(i, n)
          if (legacy_word !~ /^(token|key|secret|password|passwd|passphrase|pat|authorization|bearer)$/ || !width) {
            legacy_flush(); legacy_emit(c); continue
          }
          legacy_state = 2
        }
        if (legacy_state == 2) {
          width = legacy_width(i, n)
          if (width) {
            for (j = 0; j < width; j++) legacy_tag[++legacy_used] = value_bytes[i + j]
            legacy_prefix = legacy_used
            i += width - 1
            continue
          }
          legacy_state = 3
        }
        if (c == ">") {
          for (j = 1; j <= legacy_prefix; j++) legacy_emit(legacy_tag[j])
          legacy_emit("***MASKED***"); legacy_emit(">")
          legacy_clear()
        } else legacy_tag[++legacy_used] = c
      }
      legacy_flush()
      seal_clean_chunk()
      clean_ends[clean_segments] = clean_used
    }
    function clean_segment(segment,    j) {
      for (j = clean_starts[segment]; j <= clean_ends[segment]; j++) printf "%s", clean_parts[j]
    }
    function render(    j) {
      legacy_cleanup()
      if (!selected) {
        for (j = 1; j <= clean_segments; j++) {
          if (j > 1) printf "\r"
          clean_segment(j)
        }
        return
      }
      for (j = 1; j <= skeleton_count; j++) {
        if (j > 1) printf "\r"
        if (clean_segments == skeleton_count && substr(skeleton_segments[j], 1, 1) == "N") clean_segment(j)
        else printf "%s", substr(skeleton_segments[j], 2)
      }
    }
    BEGIN {
      profile_count = split(ENVIRON["CON295_LEGACY_PROFILE"], accepted_profile, "\n")
      for (i = 1; i <= profile_count; i++) {
        if (substr(accepted_profile[i], 1, 1) == "w") legacy_blanks[substr(accepted_profile[i], 2)] = 1
        else if (substr(accepted_profile[i], 2, 1) == ":") legacy_folds[substr(accepted_profile[i], 3)] = substr(accepted_profile[i], 1, 1)
      }
    }
    /^F/ {
      if (pending) { render(); printf "\n"; pending = 0 }
      selected = substr($0, 2, 1) == "1"
      if (selected) skeleton_count = split(substr($0, 3), skeleton_segments, "\r")
      next
    }
    /^N/ { value = substr($0, 2); pending = 1; next }
    /^$/ { final_lf = 1 }
    END { if (pending) { render(); if (final_lf) printf "\n" } }
  '
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
# ⭐ 畳むのは LF と【単独の CR】の両方 (#246)。CommonMark は CR も行末とみなすので、
#    LF だけを畳むと、コマンドや intent の中の CR で表の行が終わり、残りが表の外の
#    地の文として描画された。⛔ U+2028 / U+2029 / 改ページは CommonMark の行末ではない
#    ので畳まない (畳むと、読み手が 1 行と見るものを書き手が変えることになる)。
intent_masked="$(clip "$(printf '%s' "$intent" | mask | tr '\r\n' '  ')" "$INTENT_BYTES")"
intent_n=$(printf '%s' "$intent_masked" | wc -c | tr -d ' ')
cmd_masked="$(clip "$(printf '%s' "$cmd"    | mask | tr '\r\n' '  ')" "$(( CLIP_BYTES - intent_n ))")"

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
