from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ComplementaryTPCPartition:
    total_tpcs: int
    draft_low: int
    draft_high: int
    target_low: int
    target_high: int

    @property
    def draft_tpcs(self) -> int:
        return self.draft_high - self.draft_low

    @property
    def target_tpcs(self) -> int:
        return self.target_high - self.target_low


def build_complementary_tpc_partition(
    total_tpcs: int,
    draft_ranges: Iterable[tuple[int, int]],
) -> ComplementaryTPCPartition | None:
    total = int(total_tpcs)
    if total < 2:
        raise ValueError("complementary TPC partition requires at least 2 TPCs")

    normalized = []
    for low, high in draft_ranges:
        low, high = int(low), int(high)
        if low < 0 or high <= low or high > total:
            raise ValueError(
                f"invalid Draft TPC range [{low}, {high}) for total_tpcs={total}"
            )
        if low != 0:
            raise ValueError(
                "SpecStream complementary partition currently requires every "
                f"Draft grant to be a prefix [0,k); got [{low},{high})"
            )
        normalized.append((low, high))

    if not normalized:
        return None

    draft_high = max(high for _, high in normalized)
    if draft_high >= total:
        raise ValueError(
            "Draft grant consumes all physical TPCs, leaving no Target TPC"
        )

    return ComplementaryTPCPartition(
        total_tpcs=total,
        draft_low=0,
        draft_high=draft_high,
        target_low=draft_high,
        target_high=total,
    )
