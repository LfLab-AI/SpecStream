import pytest

from sglang.srt.speculative.spectre.specstream.config import SpecStreamConfig
from sglang.srt.speculative.spectre.specstream.gpu_history_budget import resolve_gpu_history_budget


@pytest.mark.parametrize("requested", [0, 8192, 49152])
def test_explicit_cache_limits_preserve_global_token_semantics(requested):
    budget = resolve_gpu_history_budget(
        requested_tokens=requested, allocator_size=98304,
        available_tokens=1000, retained_history_tokens=10000,
        active_requests=8, prefill_reserve_tokens=40960,
    )
    assert budget.capacity_tokens == requested
    assert not budget.automatic


def test_automatic_budget_reserves_full_prompt_tail_frontier_and_guard():
    budget = resolve_gpu_history_budget(
        requested_tokens=-1, allocator_size=98304, available_tokens=90000,
        retained_history_tokens=4096, reclaimable_seal_tokens=2048,
        active_requests=8, prefill_reserve_tokens=40960,
        active_tail_tokens=512, chunk_tokens=2048, max_q=8,
    )
    assert budget.automatic
    assert budget.free_headroom_tokens == 40960 + 2048
    assert budget.live_reserve_tokens == 8 * (512 + 2047 + 16)
    assert budget.capacity_tokens == 98304 - budget.free_headroom_tokens - budget.live_reserve_tokens
    assert budget.capacity_tokens > 8192


def test_unsealed_live_slots_and_prefills_reduce_auto_cache_capacity():
    args = dict(
        requested_tokens=-1, allocator_size=98304,
        retained_history_tokens=8192, active_requests=4,
        prefill_reserve_tokens=40960,
    )
    idle = resolve_gpu_history_budget(available_tokens=80000, **args)
    pressure = resolve_gpu_history_budget(available_tokens=20000, **args)
    assert idle.capacity_tokens > pressure.capacity_tokens
    assert pressure.capacity_tokens == 0
    # A completed seal can be retained or freed. A pending D2H cannot.
    completed = resolve_gpu_history_budget(
        available_tokens=20000, reclaimable_seal_tokens=32768, **args,
    )
    assert completed.capacity_tokens > pressure.capacity_tokens


def test_auto_budget_fails_closed_without_known_prefill_reserve():
    budget = resolve_gpu_history_budget(
        requested_tokens=-1, allocator_size=98304, available_tokens=98304,
    )
    assert budget.capacity_tokens == 0


def test_page_aligned_capacity_does_not_borrow_prompt_headroom():
    budget = resolve_gpu_history_budget(
        requested_tokens=-1, allocator_size=10000, available_tokens=10000,
        active_requests=2, prefill_reserve_tokens=3333,
        active_tail_tokens=10, chunk_tokens=64, max_q=4, page_size=64,
    )
    assert budget.capacity_tokens % 64 == 0
    assert budget.capacity_tokens + budget.live_reserve_tokens + budget.free_headroom_tokens <= 10000
    assert budget.free_headroom_tokens >= 3333 + 64


def test_config_accepts_auto_and_rejects_other_negative_budgets():
    assert SpecStreamConfig(gpu_history_cache_tokens=-1).gpu_history_cache_tokens == -1
    with pytest.raises(ValueError, match="global token budget"):
        SpecStreamConfig(gpu_history_cache_tokens=-2)
