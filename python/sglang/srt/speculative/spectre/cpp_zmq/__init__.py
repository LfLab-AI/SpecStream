import importlib
import subprocess
import warnings
from pathlib import Path

FORCE_BUILD = False
EXPECTED_PROTOCOL_SCHEMA_VERSION = 2


def build_cpp_zmq():
    build_script = Path(__file__).parent / "scripts" / "build_cpp_zmq.sh"
    if build_script.exists():
        print("[spectre_zmq] Compiling C++ extension...")
        ret = subprocess.run(
            ["bash", str(build_script)],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            raise RuntimeError(
                f"Failed to build spectre_zmq:\n{ret.stdout}\n{ret.stderr}"
            )


def _needs_rebuild() -> bool:
    package_dir = Path(__file__).parent
    extensions = list(package_dir.glob("spectre_zmq*.so"))
    if not extensions:
        return True
    source_paths = [package_dir / "setup.py"]
    source_paths.extend((package_dir / "include").glob("**/*"))
    source_paths.extend((package_dir / "src").glob("**/*"))
    newest_source = max(
        path.stat().st_mtime for path in source_paths if path.is_file()
    )
    newest_extension = max(path.stat().st_mtime for path in extensions)
    return newest_source > newest_extension


def _try_import():
    try:
        module = importlib.import_module(".spectre_zmq", __package__)
        version_fn = getattr(module, "protocol_schema_version", None)
        version = int(version_fn()) if version_fn is not None else 0
        if version != EXPECTED_PROTOCOL_SCHEMA_VERSION:
            warnings.warn(
                "stale spectre_zmq protocol extension: "
                f"expected schema {EXPECTED_PROTOCOL_SCHEMA_VERSION}, got {version}; "
                "refusing to load it"
            )
            return None
        return module
    except ImportError as e:
        warnings.warn(f"spectre_zmq not found: {e}")
        return None


if FORCE_BUILD or _needs_rebuild():
    build_cpp_zmq()

success = _try_import()
if not success:
    raise ImportError(
        "spectre_zmq protocol extension is missing or stale. Run "
        "python/sglang/srt/speculative/spectre/cpp_zmq/scripts/build_cpp_zmq.sh "
        "and restart Target and Drafter."
    )

from .spectre_zmq import DealerEndpoint, RouterEndpoint, protocol_schema_version

if int(protocol_schema_version()) != EXPECTED_PROTOCOL_SCHEMA_VERSION:
    raise ImportError(
        "spectre_zmq protocol extension is stale after rebuild; remove "
        "cpp_zmq/build and cpp_zmq/spectre_zmq*.so, then rerun "
        "cpp_zmq/scripts/build_cpp_zmq.sh"
    )

try:
    from .spectre_zmq import set_spectre_log_level
except ImportError:

    def set_spectre_log_level(level: int) -> None:
        return None


__all__ = [
    "DealerEndpoint",
    "RouterEndpoint",
    "protocol_schema_version",
    "set_spectre_log_level",
]
