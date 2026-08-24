"""Let pure SpecStream tests run without importing the optional SGLang frontend.

The production package remains untouched.  A fully installed SGLang environment
uses its normal modules; a source-only checkout gets lightweight namespace
packages pointing at the implementation directories.
"""

from pathlib import Path
import sys
import types


_PYTHON_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = {
    "sglang": _PYTHON_ROOT / "sglang",
    "sglang.srt": _PYTHON_ROOT / "sglang" / "srt",
    "sglang.srt.speculative": _PYTHON_ROOT / "sglang" / "srt" / "speculative",
    "sglang.srt.speculative.spectre": (
        _PYTHON_ROOT / "sglang" / "srt" / "speculative" / "spectre"
    ),
    "sglang.srt.speculative.spectre.specstream": (
        _PYTHON_ROOT / "sglang" / "srt" / "speculative" / "spectre" / "specstream"
    ),
}

if "sglang" not in sys.modules:
    for name, path in _PACKAGES.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
