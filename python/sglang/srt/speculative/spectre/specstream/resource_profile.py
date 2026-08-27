from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ResourceProfileEntry:
    target_shape: str
    draft_bs: int
    draft_ctx_bucket: str
    draft_tpcs: int
    draft_step_ms: float
    target_slowdown: float
    target_baseline_ms: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResourceProfileEntry":
        entry = cls(
            target_shape=str(value["target_shape"]),
            draft_bs=int(value["draft_bs"]),
            draft_ctx_bucket=str(value["draft_ctx_bucket"]),
            draft_tpcs=int(value["draft_tpcs"]),
            draft_step_ms=float(value["draft_step_ms"]),
            target_slowdown=float(value["target_slowdown"]),
            target_baseline_ms=float(value.get("target_baseline_ms", 0.0)),
        )
        if not entry.target_shape:
            raise ValueError("target_shape cannot be empty")
        if entry.draft_bs < 1 or entry.draft_tpcs < 1:
            raise ValueError("draft_bs and draft_tpcs must be positive")
        if not math.isfinite(entry.draft_step_ms) or entry.draft_step_ms <= 0:
            raise ValueError("draft_step_ms must be positive")
        if not math.isfinite(entry.target_slowdown) or entry.target_slowdown < 0:
            raise ValueError("target_slowdown cannot be negative")
        if not math.isfinite(entry.target_baseline_ms) or entry.target_baseline_ms < 0:
            raise ValueError("target_baseline_ms cannot be negative")
        shape = re.fullmatch(r"verify_bs(\d+)_q(\d+)_ctx(.+)", entry.target_shape)
        if shape is not None:
            shape_bs = int(shape.group(1))
            shape_ctx_bucket = shape.group(3)
            if shape_bs != entry.draft_bs:
                raise ValueError(
                    "target_shape batch size does not match draft_bs: "
                    f"{entry.target_shape} vs {entry.draft_bs}"
                )
            if shape_ctx_bucket != entry.draft_ctx_bucket:
                raise ValueError(
                    "target_shape context bucket does not match draft_ctx_bucket: "
                    f"{entry.target_shape} vs {entry.draft_ctx_bucket}"
                )
        return entry


@dataclass(frozen=True)
class ResourceProfile:
    gpu: str
    draft_model: str
    target_model: str
    total_tpcs: int
    entries: tuple[ResourceProfileEntry, ...]

    @classmethod
    def load(cls, path: str | Path) -> "ResourceProfile":
        profile_path = Path(path)
        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                "SpecStream SM control requires an offline resource profile; "
                f"file not found: {profile_path}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to read SpecStream resource profile {profile_path}: {exc}"
            ) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResourceProfile":
        schema_version = int(value.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(f"unsupported resource profile schema: {schema_version}")
        entries = tuple(
            ResourceProfileEntry.from_dict(item) for item in value.get("entries", ())
        )
        if not entries:
            raise ValueError("resource profile must contain at least one entry")
        entry_keys = {
            (
                entry.target_shape,
                entry.draft_bs,
                entry.draft_ctx_bucket,
                entry.draft_tpcs,
            )
            for entry in entries
        }
        if len(entry_keys) != len(entries):
            raise ValueError("resource profile contains duplicate calibration entries")
        total_tpcs = int(value.get("total_tpcs") or max(e.draft_tpcs for e in entries))
        if total_tpcs < max(e.draft_tpcs for e in entries):
            raise ValueError("total_tpcs is smaller than a calibrated draft_tpcs value")
        return cls(
            gpu=str(value.get("gpu", "")),
            draft_model=str(value.get("draft_model", "")),
            target_model=str(value.get("target_model", "")),
            total_tpcs=total_tpcs,
            entries=entries,
        )

    def matching(
        self,
        *,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
    ) -> tuple[ResourceProfileEntry, ...]:
        exact = tuple(
            entry
            for entry in self.entries
            if entry.target_shape == target_shape
            and entry.draft_bs == draft_bs
            and entry.draft_ctx_bucket == draft_ctx_bucket
        )
        if exact:
            return exact
        # Calibration generators may use a wildcard shape/bucket for a
        # deliberately conservative envelope.  Never interpolate between
        # unobserved shapes.
        return tuple(
            entry
            for entry in self.entries
            if entry.target_shape in (target_shape, "*")
            and entry.draft_bs == draft_bs
            and entry.draft_ctx_bucket in (draft_ctx_bucket, "*")
        )

    def safe_entries(
        self,
        *,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
        slowdown_budget: float,
        slack_us: float | None,
        guard_us: float,
    ) -> tuple[ResourceProfileEntry, ...]:
        candidates: Iterable[ResourceProfileEntry] = self.matching(
            target_shape=target_shape,
            draft_bs=draft_bs,
            draft_ctx_bucket=draft_ctx_bucket,
        )
        safe = []
        for entry in candidates:
            if entry.target_slowdown > slowdown_budget:
                continue
            if slack_us is not None:
                required_us = entry.draft_step_ms * 1000.0 + guard_us
                if required_us > slack_us:
                    continue
            safe.append(entry)
        return tuple(sorted(safe, key=lambda item: item.draft_tpcs))

    def select_safe(
        self,
        *,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
        slowdown_budget: float,
        slack_us: float,
        guard_us: float,
    ) -> ResourceProfileEntry | None:
        safe = self.safe_entries(
            target_shape=target_shape,
            draft_bs=draft_bs,
            draft_ctx_bucket=draft_ctx_bucket,
            slowdown_budget=slowdown_budget,
            slack_us=slack_us,
            guard_us=guard_us,
        )
        return safe[-1] if safe else None

    def select_catchup(
        self,
        *,
        target_shape: str,
        draft_bs: int,
        draft_ctx_bucket: str,
    ) -> ResourceProfileEntry | None:
        matches = self.matching(
            target_shape=target_shape,
            draft_bs=draft_bs,
            draft_ctx_bucket=draft_ctx_bucket,
        )
        return max(matches, key=lambda item: item.draft_tpcs, default=None)


def context_bucket(context_tokens: int) -> str:
    context_tokens = max(int(context_tokens), 0)
    for limit, label in (
        (2048, "2k"),
        (4096, "4k"),
        (8192, "8k"),
        (16384, "16k"),
        (32768, "32k"),
        (65536, "64k"),
    ):
        if context_tokens <= limit:
            return label
    return "64k+"
