"""Pure domain layer: data structures & contracts shared across the pipeline.

純粋なデータ構造・契約だけを置く葉レイヤ。``domain/`` は他のどの層にも依存しない
(no imports from cli/runtime/stages/providers/services)。各層はここを参照する。
"""

from __future__ import annotations

from .errors import (
    GlossaryConflictError,
    GlossaryParseError,
    SynthesisParseError,
    VaultRootError,
)
from .results import VideoRunResult
from .transcript import (
    TranscriptNotAvailable,
    TranscriptResult,
    TranscriptSnippet,
    TranscriptSource,
)
from .video import VideoMeta

__all__ = [
    "GlossaryConflictError",
    "GlossaryParseError",
    "SynthesisParseError",
    "TranscriptNotAvailable",
    "TranscriptResult",
    "TranscriptSnippet",
    "TranscriptSource",
    "VaultRootError",
    "VideoMeta",
    "VideoRunResult",
]
