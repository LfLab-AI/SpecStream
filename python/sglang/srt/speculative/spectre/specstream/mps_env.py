from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


@dataclass(frozen=True)
class MPSEnvironment:
    active_thread_percentage: int | None = None
    client_priority: int | None = None
    sm_partition: str = ""
    pipe_directory: str = ""

    @property
    def configured(self) -> bool:
        return bool(
            self.active_thread_percentage is not None
            or self.sm_partition
            or self.pipe_directory
        )


def _optional_int(env: Mapping[str, str], name: str) -> int | None:
    value = env.get(name)
    if value is None or not str(value).strip():
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def read_mps_environment(
    env: Mapping[str, str] | None = None,
) -> MPSEnvironment:
    values = os.environ if env is None else env
    percentage = _optional_int(values, "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    if percentage is not None and not 1 <= percentage <= 100:
        raise ValueError("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE must be in [1, 100]")
    priority = _optional_int(values, "CUDA_MPS_CLIENT_PRIORITY")
    if priority is not None and priority not in (0, 1):
        raise ValueError("CUDA_MPS_CLIENT_PRIORITY must be 0 or 1")
    return MPSEnvironment(
        active_thread_percentage=percentage,
        client_priority=priority,
        sm_partition=str(values.get("CUDA_MPS_SM_PARTITION", "")),
        pipe_directory=str(values.get("CUDA_MPS_PIPE_DIRECTORY", "")),
    )
