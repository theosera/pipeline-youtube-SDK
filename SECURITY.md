# Security Policy

## Supported Versions

This project is developed on `main` and has no released version line.
Only the current `main` receives security fixes — there is no backporting
to older commits or tags.

## Reporting a Vulnerability

Please **do not** open a public issue for security problems.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/theosera/pipeline-youtube-SDK/security/advisories/new).

Please include:

- what the issue is and where it occurs (file / stage / CLI flag)
- how to reproduce it, ideally as a failing test or a minimal command
- what an attacker could achieve

This is a personal project maintained by one person, so reports are
handled on a best-effort basis with no guaranteed response time. You will
get a reply either way — if a report is declined, with the reasoning.

`theosera/pipeline-youtube` and `theosera/pipeline-youtube-SDK` share most
of their pipeline code, so a flaw in one is usually present in the other.
A single report covering both is fine.

## Scope

This pipeline processes **untrusted input**: YouTube titles, descriptions,
and transcripts, plus LLM output derived from them. Reports in the
following areas are especially welcome.

- Path traversal or writes escaping the configured vault root
- Concealment attacks on filenames / frontmatter — invisible characters,
  bidi controls, mixed-script homoglyphs
- Content that survives the chapter-body validator and reaches the vault
  (active HTML, Templater tokens, embeds outside the allow-list)
- Prompt injection that changes what the pipeline writes to disk or which
  external calls it makes
- Sandbox escape from the Docker capture backend

### Out of scope

- Dependency CVEs — already tracked by Dependabot and the `pip-audit` CI
  job. Report only if the automation missed one.
- Quality or accuracy of LLM output, including hallucinated content that
  is not a security boundary violation.
- Attacks that assume the operator's own machine or vault is already
  compromised. The pipeline trusts the host it runs on.
- Costs incurred by API usage.
