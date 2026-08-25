from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import torch


class _UInt128(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint64), ("high", ctypes.c_uint64)]


def _check_return(code: int, operation: str) -> None:
    if int(code) == 0:
        return
    try:
        detail = os.strerror(abs(int(code)))
    except ValueError:
        detail = "unknown error"
    raise OSError(int(code), f"{operation} failed: {detail}")


class SMController:
    """Validated libsmctrl wrapper for stream or dedicated-process masks."""

    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        *,
        device_index: int | None = None,
        mask_scope: str = "stream",
    ) -> None:
        if getattr(torch.version, "hip", None):
            raise RuntimeError("SpecStream SM/TPC control is NVIDIA CUDA-only")
        if not torch.cuda.is_available():
            raise RuntimeError("SpecStream SM/TPC control requires CUDA")
        self.device_index = (
            torch.cuda.current_device() if device_index is None else int(device_index)
        )
        if mask_scope not in {"stream", "global"}:
            raise ValueError("mask_scope must be 'stream' or 'global'")
        self.mask_scope = mask_scope
        self.library_path = self._resolve_library(library_path)
        try:
            self._lib = ctypes.CDLL(str(self.library_path))
        except OSError as exc:
            raise RuntimeError(
                "failed to load libsmctrl for SpecStream; build "
                "csrc/specstream_smctrl first and pass "
                f"--specstream-smctrl-library. attempted={self.library_path}"
            ) from exc
        self._configure_abi()
        self.total_tpcs = self.get_tpc_count()

    @staticmethod
    def _resolve_library(
        library_path: str | os.PathLike[str] | None,
    ) -> Path | str:
        if library_path:
            return Path(library_path).expanduser().resolve()
        env_path = os.environ.get("SGLANG_SPECSTREAM_SMCTRL_LIBRARY")
        if env_path:
            return Path(env_path).expanduser().resolve()
        discovered = ctypes.util.find_library("smctrl")
        if discovered:
            return discovered
        repo_default = (
            Path(__file__).resolve().parents[6]
            / "csrc"
            / "specstream_smctrl"
            / "build"
            / "libsmctrl.so"
        )
        return repo_default

    def _configure_abi(self) -> None:
        lib = self._lib
        lib.libsmctrl_get_tpc_info_cuda.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
        ]
        lib.libsmctrl_get_tpc_info_cuda.restype = ctypes.c_int
        lib.libsmctrl_make_mask.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.libsmctrl_make_mask.restype = ctypes.c_int
        lib.libsmctrl_make_mask_ext.argtypes = [
            ctypes.POINTER(_UInt128),
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.libsmctrl_make_mask_ext.restype = ctypes.c_int
        lib.libsmctrl_set_global_mask.argtypes = [ctypes.c_uint64]
        lib.libsmctrl_set_global_mask.restype = None
        lib.libsmctrl_set_stream_mask.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.libsmctrl_set_stream_mask.restype = ctypes.c_int
        lib.libsmctrl_set_stream_mask_ext.argtypes = [ctypes.c_void_p, _UInt128]
        lib.libsmctrl_set_stream_mask_ext.restype = ctypes.c_int
        lib.libsmctrl_validate_stream_mask.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        lib.libsmctrl_validate_stream_mask.restype = ctypes.c_int

    def get_tpc_count(self) -> int:
        count = ctypes.c_uint32()
        code = self._lib.libsmctrl_get_tpc_info_cuda(
            ctypes.byref(count), self.device_index
        )
        _check_return(code, "libsmctrl_get_tpc_info_cuda")
        if count.value < 1:
            raise RuntimeError("libsmctrl reported zero TPCs")
        return int(count.value)

    def validate_range(self, low: int, high_exclusive: int) -> tuple[int, int]:
        low, high_exclusive = int(low), int(high_exclusive)
        if low < 0 or high_exclusive <= low or high_exclusive > self.total_tpcs:
            raise ValueError(
                "invalid TPC range "
                f"[{low}, {high_exclusive}); device has {self.total_tpcs} TPCs"
            )
        return low, high_exclusive

    @staticmethod
    def _stream_pointer(stream: torch.cuda.Stream) -> ctypes.c_void_p:
        pointer = int(stream.cuda_stream)
        if pointer == 0:
            raise ValueError("CUDA stream has a null handle")
        return ctypes.c_void_p(pointer)

    def set_stream_mask(
        self, stream: torch.cuda.Stream, low: int, high_exclusive: int
    ) -> int:
        low, high_exclusive = self.validate_range(low, high_exclusive)
        if self.total_tpcs <= 64:
            mask = ctypes.c_uint64()
            code = self._lib.libsmctrl_make_mask(
                ctypes.byref(mask), low, high_exclusive
            )
            _check_return(code, "libsmctrl_make_mask")
            if self.mask_scope == "global":
                self._lib.libsmctrl_set_global_mask(mask)
                return int(mask.value)
            stream_pointer = self._stream_pointer(stream)
            code = self._lib.libsmctrl_set_stream_mask(stream_pointer, mask)
            _check_return(code, "libsmctrl_set_stream_mask")
            return int(mask.value)

        if self.mask_scope == "global":
            raise RuntimeError(
                "process-global libsmctrl masks support at most 64 TPCs"
            )
        stream_pointer = self._stream_pointer(stream)
        mask_ext = _UInt128()
        code = self._lib.libsmctrl_make_mask_ext(
            ctypes.byref(mask_ext), low, high_exclusive
        )
        _check_return(code, "libsmctrl_make_mask_ext")
        code = self._lib.libsmctrl_set_stream_mask_ext(stream_pointer, mask_ext)
        _check_return(code, "libsmctrl_set_stream_mask_ext")
        return int(mask_ext.low) | (int(mask_ext.high) << 64)

    def validate_stream_mask(
        self,
        stream: torch.cuda.Stream,
        low: int,
        high_exclusive: int,
        *,
        echo: bool = False,
    ) -> None:
        low, high_exclusive = self.validate_range(low, high_exclusive)
        code = self._lib.libsmctrl_validate_stream_mask(
            self._stream_pointer(stream), low, high_exclusive, bool(echo)
        )
        _check_return(code, "libsmctrl_validate_stream_mask")
