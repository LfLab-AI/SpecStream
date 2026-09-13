from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.cpu_history_store import (  # noqa: E402
    SealTicket,
)
from sglang.srt.mem_cache.common import evict_from_tree_cache  # noqa: E402
from sglang.srt.speculative.spectre.specstream.config import (  # noqa: E402
    SpecStreamConfig,
)
from sglang.srt.speculative.spectre.specstream.state import (  # noqa: E402
    TargetTieredKVState,
)
from sglang.srt.speculative.spectre.specstream.verifier import (  # noqa: E402
    SpecStreamTargetRuntime,
    _PendingSeal,
)


class _Event:
    ready = False

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


class _HistoryStore:
    @staticmethod
    def complete_seal(ticket, *, wait=False):
        if wait:
            ticket.event.synchronize()
        elif not ticket.event.query():
            return False
        ticket.completed = True
        ticket.event = None
        return True


class _Allocator:
    def __init__(self, available=100):
        self.freed = []
        self.available = available

    def free(self, slots):
        self.freed.append(slots.clone())
        self.available += int(slots.numel())

    def available_size(self):
        return self.available


def _enable_cache_lifecycle(runtime, *, capacity=0, min_free=0, chunk_tokens=2):
    runtime._gpu_history_req_pool_idx = {}
    runtime._gpu_history_cache_capacity = capacity
    runtime.config = SimpleNamespace(
        gpu_history_min_free_tokens=min_free,
        chunk_tokens=chunk_tokens,
    )


def test_pending_seal_is_published_only_after_event_completion():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState("r", committed_len=8, logical_len=8)
    state.start_seal()
    event = _Event()
    ticket = SealTicket([7], event, [])
    slots = torch.tensor([3, 4, 5, 6])
    runtime.states = {"r": state}
    runtime._pending_seals = {"r": _PendingSeal("r", 0, 0, 4, slots, ticket)}
    runtime.history_store = _HistoryStore()
    runtime.token_to_kv_pool_allocator = _Allocator()
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    )
    _enable_cache_lifecycle(runtime)

    assert runtime._poll_pending_seals() == 0
    assert state.history_len == 0
    assert state.seal_inflight
    assert not runtime.token_to_kv_pool_allocator.freed

    event.ready = True
    assert runtime._poll_pending_seals() == 1
    assert state.history_len == state.tail_start == 4
    assert state.stream_enabled
    assert not state.seal_inflight
    assert torch.equal(
        runtime.req_to_token_pool.req_to_token[0],
        torch.tensor([0, 0, 0, 0, 7, 8, 9, 10]),
    )
    assert len(runtime.token_to_kv_pool_allocator.freed) == 1


def test_terminal_release_is_the_only_blocking_drain():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState("r", committed_len=4, logical_len=4)
    state.start_seal()
    event = _Event()
    runtime.states = {"r": state}
    runtime._pending_seals = {
        "r": _PendingSeal(
            "r", 0, 0, 4, torch.tensor([1, 2, 3, 4]), SealTicket([1], event, [])
        )
    }
    runtime.history_store = _HistoryStore()
    runtime.token_to_kv_pool_allocator = _Allocator()
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[1, 2, 3, 4]])
    )
    _enable_cache_lifecycle(runtime, capacity=2)

    runtime.prepare_request_release("r")
    assert event.ready
    assert "r" not in runtime._pending_seals
    assert state.history_len == 4
    assert state.gpu_history_len == 0
    assert torch.equal(
        runtime.req_to_token_pool.req_to_token[0], torch.tensor([0, 0, 0, 0])
    )


def test_completed_seal_keeps_only_bounded_cpu_backed_gpu_prefix():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState("r", committed_len=8, logical_len=8)
    state.start_seal()
    event = _Event()
    event.ready = True
    slots = torch.tensor([3, 4, 5, 6])
    runtime.states = {"r": state}
    runtime._pending_seals = {
        "r": _PendingSeal("r", 0, 0, 4, slots, SealTicket([7], event, []))
    }
    runtime.history_store = _HistoryStore()
    runtime.token_to_kv_pool_allocator = _Allocator()
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    )
    _enable_cache_lifecycle(runtime, capacity=2)

    assert runtime._poll_pending_seals() == 1
    assert state.history_len == 4
    assert state.gpu_history_len == 2
    assert torch.equal(
        runtime.req_to_token_pool.req_to_token[0],
        torch.tensor([3, 4, 0, 0, 7, 8, 9, 10]),
    )
    assert len(runtime.token_to_kv_pool_allocator.freed) == 1
    assert torch.equal(
        runtime.token_to_kv_pool_allocator.freed[0], torch.tensor([5, 6])
    )


def test_gpu_history_allocation_evicts_only_the_exact_cpu_backed_shortage():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState(
        "r", committed_len=8, history_len=4, gpu_history_len=4, logical_len=8
    )
    runtime.states = {"r": state}
    runtime._pending_seals = {}
    runtime._gpu_history_req_pool_idx = {"r": 0}
    runtime._gpu_history_cache_capacity = 4
    runtime.config = SimpleNamespace(
        gpu_history_min_free_tokens=0,
        chunk_tokens=2,
    )
    runtime.token_to_kv_pool_allocator = _Allocator(available=1)
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    )

    assert runtime.evict_gpu_history_for_allocation(4) == 3
    assert state.gpu_history_len == 1
    assert torch.equal(
        runtime.req_to_token_pool.req_to_token[0],
        torch.tensor([3, 0, 0, 0, 7, 8, 9, 10]),
    )
    assert torch.equal(
        runtime.token_to_kv_pool_allocator.freed[0], torch.tensor([4, 5, 6])
    )


def test_polling_does_not_proactively_evict_to_a_fixed_watermark():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState(
        "r", committed_len=8, history_len=4, gpu_history_len=4, logical_len=8
    )
    runtime.states = {"r": state}
    runtime._pending_seals = {}
    runtime._gpu_history_req_pool_idx = {"r": 0}
    runtime._gpu_history_cache_capacity = 4
    runtime.config = SimpleNamespace(
        gpu_history_min_free_tokens=4,
        chunk_tokens=2,
    )
    runtime.token_to_kv_pool_allocator = _Allocator(available=1)
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    )

    assert runtime._poll_pending_seals() == 0
    assert state.gpu_history_len == 4
    assert runtime.token_to_kv_pool_allocator.available_size() == 1


def test_native_allocation_calls_external_cache_with_exact_requirement():
    allocator = _Allocator(available=1)
    requested = []

    def external_evictor(required_tokens):
        requested.append(required_tokens)
        allocator.available = required_tokens

    allocator._external_kv_cache_evictor = external_evictor
    native_evictions = []
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        is_chunk_cache=lambda: False,
        evict=lambda params: native_evictions.append(params),
    )

    evict_from_tree_cache(tree_cache, 4)

    assert requested == [4]
    assert native_evictions == []
    assert allocator.available_size() == 4


def test_optional_allocation_guard_is_checked_even_without_a_shortage():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState(
        "r", committed_len=8, history_len=4, gpu_history_len=4, logical_len=8
    )
    runtime.states = {"r": state}
    runtime._pending_seals = {}
    runtime._gpu_history_req_pool_idx = {"r": 0}
    runtime._gpu_history_cache_capacity = 4
    runtime.config = SimpleNamespace(
        gpu_history_min_free_tokens=2,
        chunk_tokens=2,
    )
    runtime.token_to_kv_pool_allocator = _Allocator(available=4)
    runtime.token_to_kv_pool_allocator._external_kv_cache_evictor = (
        runtime.evict_gpu_history_for_allocation
    )
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    )
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=runtime.token_to_kv_pool_allocator,
        is_chunk_cache=lambda: False,
        evict=lambda _params: None,
    )

    evict_from_tree_cache(tree_cache, 4)

    assert state.gpu_history_len == 2
    assert runtime.token_to_kv_pool_allocator.available_size() == 6


def test_gpu_history_page_table_hole_drops_only_the_cache_replica():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    state = TargetTieredKVState(
        "r", committed_len=8, history_len=4, gpu_history_len=4, logical_len=8
    )
    runtime.states = {"r": state}
    runtime._gpu_history_req_pool_idx = {"r": 0}
    runtime.token_to_kv_pool_allocator = _Allocator()
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[3, 0, 5, 6, 7, 8, 9, 10]])
    )

    assert runtime._evict_gpu_history_suffix("r", 0) == 4
    assert state.history_len == 4
    assert state.gpu_history_len == 0
    assert torch.equal(
        runtime.req_to_token_pool.req_to_token[0],
        torch.tensor([0, 0, 0, 0, 7, 8, 9, 10]),
    )
    assert torch.equal(
        runtime.token_to_kv_pool_allocator.freed[0], torch.tensor([3, 5, 6])
    )


def test_gpu_history_global_cap_is_shared_fairly_by_active_requests():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    first = TargetTieredKVState(
        "first", committed_len=8, history_len=4, gpu_history_len=4, logical_len=8
    )
    second = TargetTieredKVState("second", committed_len=8, logical_len=8)
    second.start_seal()
    event = _Event()
    event.ready = True
    runtime.states = {"first": first, "second": second}
    runtime._pending_seals = {
        "second": _PendingSeal(
            "second",
            1,
            0,
            4,
            torch.tensor([11, 12, 13, 14]),
            SealTicket([8], event, []),
        )
    }
    runtime._gpu_history_req_pool_idx = {"first": 0}
    runtime._gpu_history_cache_capacity = 4
    runtime.config = SimpleNamespace(
        gpu_history_min_free_tokens=0,
        chunk_tokens=2,
    )
    runtime.history_store = _HistoryStore()
    runtime.token_to_kv_pool_allocator = _Allocator()
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor(
            [
                [3, 4, 5, 6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15, 16, 17, 18],
            ]
        )
    )

    assert runtime._poll_pending_seals() == 1
    assert first.gpu_history_len == second.gpu_history_len == 2
    assert first.gpu_history_len + second.gpu_history_len == 4
    assert torch.equal(
        runtime.req_to_token_pool.req_to_token[:, :4],
        torch.tensor([[3, 4, 0, 0], [11, 12, 0, 0]]),
    )


def test_fully_gpu_resident_history_uses_native_attention_fast_path():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime._poll_pending_seals = lambda: 0
    state = TargetTieredKVState(
        "r", history_len=4, gpu_history_len=4, stream_enabled=True
    )
    runtime.states = {"r": state}
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="r")])
    assert not runtime.batch_requires_streaming(batch)
    state.gpu_history_len = 3
    assert runtime.batch_requires_streaming(batch)

def test_controller_batch_state_counts_only_cpu_history_misses():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime._poll_pending_seals = lambda: 0
    runtime.states = {
        "r": TargetTieredKVState(
            "r", committed_len=8, history_len=4, gpu_history_len=3, logical_len=8
        )
    }
    runtime.history_store = SimpleNamespace(
        layer_ids=(0, 1),
        kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
    )
    runtime.config = SimpleNamespace(chunk_tokens=2)
    req = SimpleNamespace(rid="r", cur_drafts=[1])
    batch = SimpleNamespace(reqs=[req], specstream_rejected=False, is_high_overhead=False)

    batch_state = runtime.collect_batch_state(batch)

    assert batch_state.history_tokens == 1
    assert batch_state.history_bytes == 64
    assert batch_state.num_chunks == 1
    assert batch_state.no_draft_ratio == 0

def test_after_extend_seals_only_completed_prefills():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime.states = {}
    runtime._pending_seals = {}
    runtime._poll_pending_seals = lambda **kwargs: 0
    sealed = []
    runtime._maybe_seal = lambda req, state: sealed.append(
        (req.rid, state.committed_len)
    )

    complete = SimpleNamespace(rid="complete", is_chunked=0)
    partial = SimpleNamespace(rid="partial", is_chunked=1)
    batch = SimpleNamespace(
        reqs=[complete, partial], seq_lens_cpu=torch.tensor([16384, 4096])
    )

    runtime.after_extend(batch)

    assert sealed == [("complete", 16384)]
    assert runtime.states["complete"].committed_len == 16384
    assert runtime.states["partial"].committed_len == 4096


def test_profile_only_control_runtime_never_seals_native_gpu_kv():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime.config = SpecStreamConfig(profile_only=True, coexec_enabled=True)
    runtime._pending_seals = {}
    state = TargetTieredKVState("native", committed_len=16384, logical_len=16384)

    class _MustNotOffload:
        @staticmethod
        def seal_slots_async(**_kwargs):
            raise AssertionError("profile-only mode attempted to offload Target KV")

    runtime.history_store = _MustNotOffload()
    runtime._maybe_seal(SimpleNamespace(rid="native"), state)

    assert state.history_len == 0
    assert not state.stream_enabled
    assert not state.seal_inflight


@pytest.mark.parametrize("automatic", [False, True])
def test_new_seal_never_recreates_an_evicted_gpu_history_prefix(automatic):
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    prefix = 4 if automatic else 2
    state = TargetTieredKVState(
        "r", committed_len=10, logical_len=10, history_len=4,
        gpu_history_len=prefix, tail_start=4,
    )
    state.start_seal()
    runtime.states = {"r": state}
    runtime._pending_seals = {}
    runtime.token_to_kv_pool_allocator = _Allocator(available=4)
    runtime.token_to_kv_pool_allocator.size = 16
    runtime.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[1, 2, 3 if automatic else 0, 4 if automatic else 0, 5, 6, 7, 8, 9, 10]])
    )
    _enable_cache_lifecycle(runtime, capacity=prefix)
    runtime._gpu_history_req_pool_idx = {"r": 0}
    if automatic:
        runtime.config = SpecStreamConfig(gpu_history_cache_tokens=-1, chunk_tokens=2)
        runtime.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(context_length=32, max_prefill_tokens=32),
        )
    ticket = SealTicket([2], None, [], completed=True)
    pending = _PendingSeal("r", 0, 4, 8, torch.tensor([5, 6, 7, 8]), ticket)
    retained, released = runtime._publish_completed_seal(pending, state)
    assert retained == 0 and released == 4
    assert state.history_len == 8
    assert state.gpu_history_len == (0 if automatic else 2)
    assert torch.equal(runtime.req_to_token_pool.req_to_token[0, 4:8], torch.zeros(4, dtype=torch.int64))
    assert torch.equal(runtime.req_to_token_pool.req_to_token[0, 8:], torch.tensor([9, 10]))
    if automatic:
        assert runtime._gpu_history_cache_capacity == 0
        assert torch.equal(runtime.req_to_token_pool.req_to_token[0, :4], torch.zeros(4, dtype=torch.int64))


def test_budget_refresh_supports_explicit_allocator_without_available_method():
    runtime = SpecStreamTargetRuntime.__new__(SpecStreamTargetRuntime)
    runtime.config = SpecStreamConfig(gpu_history_cache_tokens=8192)
    runtime.token_to_kv_pool_allocator = SimpleNamespace(size=98304)
    runtime._update_gpu_history_budget()
    assert runtime._gpu_history_cache_capacity == 8192
    runtime.config = SpecStreamConfig(gpu_history_cache_tokens=-1)
    runtime._update_gpu_history_budget()
    assert runtime._gpu_history_cache_capacity == 0
