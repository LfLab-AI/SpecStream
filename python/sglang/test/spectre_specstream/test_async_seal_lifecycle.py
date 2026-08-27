from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.cpu_history_store import (  # noqa: E402
    SealTicket,
)
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
    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.append(slots.clone())


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

    runtime.prepare_request_release("r")
    assert event.ready
    assert "r" not in runtime._pending_seals
    assert state.history_len == 4


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
