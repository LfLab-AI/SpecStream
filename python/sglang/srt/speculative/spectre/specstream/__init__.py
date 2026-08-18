"""SpecStream extensions for the SPECTRE target verifier.

The package is intentionally local to SPECTRE.  It does not modify the
generic HiCache/offload data path and must never be imported by the remote
drafter.
"""

from sglang.srt.speculative.spectre.specstream.config import SpecStreamConfig
from sglang.srt.speculative.spectre.specstream.state import TargetTieredKVState

__all__ = ["SpecStreamConfig", "TargetTieredKVState"]
