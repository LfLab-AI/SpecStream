from types import SimpleNamespace
import sys
import time

import pytest


torch = pytest.importorskip("torch")

from sglang.srt.speculative.spectre.specstream.verifier import SpecStreamVerifier


class _HistoryStore:
    layer_ids = (0, 1)

    def __init__(self, chunks):
        self.chunks = chunks

    def iter_layer_chunks(self, rid, layer_id, *, history_end):
        assert history_end == 4
        return iter(self.chunks[(rid, layer_id)])


class _Staging:
    num_buffers = 2

    def __init__(self):
        self.independent_submits = []
        self.cohort_submits = []
        self.cohort_direct_submits = []
        self.waited = []

    def submit_many(self, sources, slot):
        transfer = SimpleNamespace(
            kind="independent",
            index=len(self.independent_submits),
            slot=slot,
            tensor=torch.cat(tuple(sources)),
            nbytes=sum(source.nbytes for source in sources),
            submitted_ns=time.perf_counter_ns(),
        )
        self.independent_submits.append((tuple(sources), slot))
        return transfer

    def submit_cohort_groups(self, source_groups, slot):
        transfer = SimpleNamespace(
            kind="cohort",
            index=len(self.cohort_submits),
            slot=slot,
        )
        self.cohort_submits.append((tuple(map(tuple, source_groups)), slot))
        return transfer

    def submit_cohort_groups_direct_async(self, source_groups, slot):
        transfer = SimpleNamespace(
            kind="cohort_direct",
            index=len(self.cohort_direct_submits),
            slot=slot,
        )
        self.cohort_direct_submits.append((tuple(map(tuple, source_groups)), slot))
        return transfer

    def wait_ready(self, transfer):
        self.waited.append(transfer)
        return transfer.tensor

    def mark_consumed(self, transfer):
        return None


def _chunk(start):
    return SimpleNamespace(
        abs_start=start,
        length=2,
        tensor=torch.zeros((2, 2, 1, 2), dtype=torch.float32),
    )


def _verifier(*, max_cohort_size=2, max_cohort_delay_us=1_000_000):
    verifier = SpecStreamVerifier.__new__(SpecStreamVerifier)
    verifier.config = SimpleNamespace(
        chunks_per_transfer=1,
        reference_attention=False,
        max_cohort_delay_us=max_cohort_delay_us,
        max_cohort_size=max_cohort_size,
    )
    verifier.staging = _Staging()
    verifier._single_layer_prefetch = {}
    verifier._batched_layer_prefetch = {}
    chunks = {}
    for rid in ("r0", "r1", "r2", "r3"):
        for layer_id in (0, 1):
            chunks[(rid, layer_id)] = [_chunk(0), _chunk(2)]
    verifier.history_store = _HistoryStore(chunks)
    return verifier


def _items(count):
    return [SimpleNamespace(rid=f"r{i}", history_len=4) for i in range(count)]


def test_independent_prefetch_is_one_global_queue_for_multiple_requests():
    verifier = _verifier()
    items = _items(2)
    meta = SimpleNamespace(round_id=7)

    verifier._prefetch_next_independent_layer(items, meta, SimpleNamespace(layer_id=0))

    assert len(verifier.staging.independent_submits) == 2
    tasks, task_keys = verifier._build_independent_tasks(items, 1)
    transfers = verifier._take_batched_prefetch(
        meta=meta,
        layer_id=1,
        mode="independent",
        items=items,
        task_keys=task_keys,
    )
    assert len(tasks) == 4
    assert set(transfers) == {0, 1}


def test_cohort_prefetch_remains_bounded_across_multiple_plans():
    verifier = _verifier(max_cohort_size=2)
    items = _items(4)
    meta = SimpleNamespace(round_id=11, q_len=5)

    verifier._prefetch_next_cohort_layer(items, meta, SimpleNamespace(layer_id=0))

    # Four requests form two cohort plans.  Only the two global staging slots
    # are prefetched, rather than two slots per request or per plan.
    assert verifier.staging.cohort_submits == []
    assert len(verifier.staging.cohort_direct_submits) == 2
    groups, _, plans, tasks, task_keys = verifier._build_cohort_layer_work(
        items, meta, 1
    )
    transfers = verifier._take_batched_prefetch(
        meta=meta,
        layer_id=1,
        mode="cohort",
        items=items,
        task_keys=task_keys,
    )
    assert len(plans) == 2
    assert len(tasks) == 4
    assert set(groups) == {"r0", "r1", "r2", "r3"}
    assert set(transfers) == {0, 1}


def test_already_batched_requests_cohort_even_with_zero_extra_delay():
    verifier = _verifier(max_cohort_size=4, max_cohort_delay_us=0)
    items = _items(4)
    meta = SimpleNamespace(round_id=12, q_len=5)

    _, _, plans, tasks, _ = verifier._build_cohort_layer_work(items, meta, 0)

    assert len(plans) == 1
    assert tuple(work.rid for work in plans[0].items) == (
        "r0",
        "r1",
        "r2",
        "r3",
    )
    assert len(tasks) == 2


def test_large_steady_state_cohort_avoids_synchronous_host_pack(monkeypatch):
    verifier = _verifier(max_cohort_size=2)
    items = _items(2)
    meta = SimpleNamespace(round_id=13, q_len=5)
    groups, _, _, tasks, _ = verifier._build_cohort_layer_work(items, meta, 0)
    module = sys.modules[SpecStreamVerifier.__module__]
    monkeypatch.setattr(module, "_MAX_SYNCHRONOUS_COHORT_PACK_BYTES", 1)

    verifier._submit_cohort_task(tasks[0], groups, 0)

    assert verifier.staging.cohort_submits == []
    assert len(verifier.staging.cohort_direct_submits) == 1


def test_discard_request_invalidates_shared_prefetch_queue():
    verifier = _verifier()
    items = _items(2)
    meta = SimpleNamespace(round_id=3)
    verifier._prefetch_next_independent_layer(items, meta, SimpleNamespace(layer_id=0))

    verifier.discard_layer_prefetch("r1")

    assert verifier._batched_layer_prefetch == {}


def test_independent_pipeline_steals_first_free_slot_before_last_group():
    verifier = _verifier()
    items = _items(2)
    meta = SimpleNamespace(round_id=17)
    layer = SimpleNamespace(layer_id=0)
    queries = {item.rid: object() for item in items}
    states = {item.rid: object() for item in items}
    verifier.profiler = SimpleNamespace(
        record_attention=lambda *args, **kwargs: None,
        record_h2d=lambda *args, **kwargs: None,
    )
    verifier._update_history_state = lambda state, *args, **kwargs: state
    original_prefetch = verifier._prefetch_next_independent_layer
    steal_after_wait_counts = []

    def record_prefetch(*args, **kwargs):
        if kwargs.get("available_slots") is not None:
            steal_after_wait_counts.append(len(verifier.staging.waited))
        return original_prefetch(*args, **kwargs)

    verifier._prefetch_next_independent_layer = record_prefetch

    verifier._stream_history_independent_batch(items, queries, states, meta, layer)

    # Four current-layer tasks run through two slots.  Slot 0 becomes free
    # after task 2 and starts L+1 while task 3 has not yet been consumed.
    assert len(verifier.staging.waited) == 4
    assert steal_after_wait_counts[0] == 3
    cached = next(iter(verifier._batched_layer_prefetch.values()))
    assert set(cached.transfers) == {0, 1}
