"""In-process optimistic draft-ahead for STANDALONE speculative decoding.

This package intentionally has no dependency on the remote SPECTRE scheduler.
The Target remains authoritative and the Drafter only owns speculative state.
"""

from sglang.srt.speculative.specstream_inproc.ahead_state import (
    AheadPhase,
    AheadRequestState,
    ReconcileResult,
    longest_common_prefix,
)
from sglang.srt.speculative.specstream_inproc.resource_controller import (
    CoexecutionAction,
    CoexecutionMode,
    InProcessResourceController,
)

__all__ = [
    "AheadPhase",
    "AheadRequestState",
    "CoexecutionAction",
    "CoexecutionMode",
    "InProcessResourceController",
    "ReconcileResult",
    "longest_common_prefix",
]
