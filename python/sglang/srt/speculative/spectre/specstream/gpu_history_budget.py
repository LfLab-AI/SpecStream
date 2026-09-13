"""GPU History admission limits expressed in global Target KV token slots.

The cache retains CPU-backed replicas in the existing allocator; it never
allocates a second KV pool. All layers and all local KV heads are represented
by each slot. For Qwen3-32B BF16 TP=2, one slot is 128 KiB per rank.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUHistoryBudget:
    capacity_tokens: int
    free_headroom_tokens: int
    live_reserve_tokens: int
    automatic: bool


def resolve_gpu_history_budget(
    *,
    requested_tokens: int,
    allocator_size: int,
    available_tokens: int,
    retained_history_tokens: int = 0,
    reclaimable_seal_tokens: int = 0,
    active_requests: int = 0,
    active_tail_tokens: int = 512,
    chunk_tokens: int = 2048,
    max_q: int = 8,
    min_free_tokens: int = 0,
    prefill_reserve_tokens: int = 0,
    page_size: int = 1,
) -> GPUHistoryBudget:
    """Resolve explicit limits or an opt-in (-1) pressure-sensitive limit.

    ``reclaimable_seal_tokens`` includes only D2H-completed slots whose CPU
    replica is safe, including the seal currently being published. In-flight
    D2H, active Tail, Frontier and pending prefills are never counted as free.
    ``prefill_reserve_tokens`` must cover the largest admissible next prompt
    allocation (or prefill batch). Auto mode also keeps a separate allocation
    guard and room for every active request's bounded Tail/Frontier.

    Explicit zero and positive budgets preserve the prior configuration
    semantics. Allocation-time eviction remains required in every mode.
    """
    requested_tokens = int(requested_tokens)
    if requested_tokens < -1:
        raise ValueError("GPU History budget must be -1 (auto) or nonnegative")
    size = max(0, int(allocator_size))
    guard = max(0, int(min_free_tokens))
    page = max(1, int(page_size))
    if requested_tokens >= 0:
        return GPUHistoryBudget(
            min(requested_tokens, max(0, size - guard)),
            guard,
            0,
            False,
        )

    # A missing prompt reserve cannot safely enable opportunistic admission.
    # The caller can still install its eviction callback before this is known.
    if size == 0 or int(prefill_reserve_tokens) <= 0:
        return GPUHistoryBudget(0, size, 0, True)
    available = max(0, min(int(available_tokens), size))
    replicas = max(0, int(retained_history_tokens))
    sealed = max(0, int(reclaimable_seal_tokens))
    non_cache_live = max(0, size - available - replicas - sealed)
    tail_frontier = max(0, int(active_requests)) * (
        max(0, int(active_tail_tokens))
        + max(1, int(chunk_tokens))
        - 1
        + 2 * max(1, int(max_q))
    )
    live_reserve = max(non_cache_live, tail_frontier)
    free_headroom = int(prefill_reserve_tokens) + max(guard, int(chunk_tokens), page)
    free_headroom = ((free_headroom + page - 1) // page) * page
    capacity = max(0, size - live_reserve - free_headroom)
    capacity = (capacity // page) * page
    return GPUHistoryBudget(capacity, free_headroom, live_reserve, True)
