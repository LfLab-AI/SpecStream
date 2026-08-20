from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class StreamWorkItem:
    rid: str
    round_id: int
    layer_id: int
    q_len: int
    chunk_idx: int
    token_begin: int
    token_end: int
    bytes: int
    deadline_ns: int
    cpu_descriptor: Any = None

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_begin


@dataclass(frozen=True)
class CohortKey:
    layer_id: int
    q_bucket: int
    dtype: str
    head_dim: int
    local_kv_heads: int
    chunk_tokens: int


@dataclass(frozen=True)
class CohortPlan:
    key: CohortKey
    items: tuple[StreamWorkItem, ...]
    packed_bytes: int
    staging_slot: int


def compatibility_key(item: StreamWorkItem) -> CohortKey:
    descriptor = item.cpu_descriptor
    descriptors = (
        tuple(descriptor) if isinstance(descriptor, (list, tuple)) else (descriptor,)
    )
    if not descriptors:
        raise ValueError("cohort CPU descriptor cannot be empty")
    first_tensor = descriptors[0].tensor
    return CohortKey(
        layer_id=item.layer_id,
        q_bucket=item.q_len,
        dtype=str(first_tensor.dtype),
        head_dim=int(first_tensor.shape[-1]),
        local_kv_heads=int(first_tensor.shape[-2]),
        chunk_tokens=sum(int(value.tensor.shape[0]) for value in descriptors),
    )


def build_cohort_plans(
    work_items: Iterable[StreamWorkItem],
    *,
    max_cohort_size: int,
    num_staging_slots: int,
    max_cohort_delay_us: float,
    now_ns: int | None = None,
) -> list[CohortPlan]:
    if max_cohort_size < 1 or num_staging_slots < 1:
        raise ValueError("cohort and staging sizes must be positive")
    now_ns = time.time_ns() if now_ns is None else now_ns
    if max_cohort_delay_us < 0:
        raise ValueError("max_cohort_delay_us cannot be negative")
    buckets: dict[CohortKey, list[StreamWorkItem]] = defaultdict(list)
    immediate: list[StreamWorkItem] = []
    for item in work_items:
        # An expired item never waits for peers, which bounds P99 inflation.
        if item.deadline_ns <= now_ns:
            immediate.append(item)
        else:
            buckets[compatibility_key(item)].append(item)

    plans: list[CohortPlan] = []
    slot = 0
    for item in sorted(immediate, key=lambda value: (value.deadline_ns, value.bytes)):
        plans.append(
            CohortPlan(
                compatibility_key(item),
                (item,),
                item.bytes,
                slot % num_staging_slots,
            )
        )
        slot += 1

    for key in sorted(
        buckets,
        key=lambda value: (
            value.layer_id,
            value.q_bucket,
            value.dtype,
            value.head_dim,
            value.local_kv_heads,
            value.chunk_tokens,
        ),
    ):
        items = sorted(buckets[key], key=lambda value: (value.deadline_ns, value.bytes))
        for begin in range(0, len(items), max_cohort_size):
            group = tuple(items[begin : begin + max_cohort_size])
            plans.append(
                CohortPlan(
                    key,
                    group,
                    sum(item.bytes for item in group),
                    slot % num_staging_slots,
                )
            )
            slot += 1
    return plans
