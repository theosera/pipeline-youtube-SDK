"""Hands-on mode (--handson): single long-form talk video → step-by-step tutorial.

New architecture package for the transcript segment classification
(LECTURE / QA / TIPS) and the hands-on generation that builds on it:

- ``schemas``   — typed contracts + JSON parsers + H:MM:SS formatting
- ``segmenter`` — LLM segment classification + deterministic normalization
- ``planner``   — step planning + lossless QA/TIPS insight assignment
- ``steps``     — per-step body generation + MOC / final QA-Tips summary
- ``writer``    — vault writes (fold → filename → frontmatter → validate)

Package invariant (long-video safety): every LLM round-trip uses integer
seconds; H:MM:SS strings are render-only (``schemas.fmt_hms``). No module
in this package may parse clock-string timestamps — the legacy
``[MM:SS ~ MM:SS]`` regexes cap at 99:59 and silently drop ranges beyond
100 minutes, which long-form talks routinely exceed.
"""
