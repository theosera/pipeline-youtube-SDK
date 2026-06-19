"""Pure domain-layer exception types.

横断的に使われる純粋な例外型 (docstring のみ・ロジックなし) を集約する。
``domain/`` は他のどの層にも依存しないため、ここに置くことで config /
synthesis / glossary など各層が共通の例外契約を参照できる。

ここに置かないもの:
- ``transient`` 属性などロジックを持つ provider 例外 (``LLMError``)。
- モジュール固有の継承階層 (``CaptureBackendError`` → ``DockerBackendNotReady``)。
- 定義モジュール内でのみ raise/except する局所例外
  (``SummaryOutputError`` / ``EvaluationParseError``)。
"""

from __future__ import annotations


class VaultRootError(ValueError):
    """Raised when vault_root fails safety checks."""


class SynthesisParseError(RuntimeError):
    """Raised when an agent's JSON output cannot be parsed."""


class GlossaryParseError(ValueError):
    """Raised when glossary JSON is structurally malformed.

    Strict by design: unlike the advisory LLM-output parsers, the
    glossary is first-party data, so a bad payload should fail the build
    rather than be silently dropped.
    """


class GlossaryConflictError(ValueError):
    """Raised when two entries claim the same variant for different canonicals.

    Surfaced at ``Normalizer`` build time (see ``glossary.normalizer``).
    Carries the offending key so the data error is actionable.
    """


__all__ = [
    "GlossaryConflictError",
    "GlossaryParseError",
    "SynthesisParseError",
    "VaultRootError",
]
