"""Exercise the scheduler with divergent rank-local gates using real CPU Gloo."""

from datetime import timedelta
from types import SimpleNamespace

import pytest


def _run_tp_profile_rank(rank, rendezvous):
    import torch.distributed as dist

    from sglang.srt.speculative.spectre.verifier.spectre_target_scheduler_mixin import (
        SchedulerSpectreTargetMixin,
    )

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:
        # Initially both ranks want samples (warmup). Later they disagree in
        # both directions, modeling locally retired timing/History shapes.
        gates = (True, True, False, True, False, False, True)
        step = 0
        local_gate_calls = []
        observed = []
        decisions = []
        gathered_rounds = []

        def local_gate():
            local_gate_calls.append(step)
            return gates[step] if rank == 0 or step == 0 else not gates[step]

        def gather(sample):
            samples = [None, None]
            dist.all_gather_object(samples, sample)
            gathered_rounds.append(step)
            return samples

        def choose(batch):
            assert rank == 0
            return SimpleNamespace(q=2 + 2 * (step % 4), mode="ordinary")

        runtime = SimpleNamespace(
            controller=object(),
            should_sync_tp_profile=local_gate,
            local_tp_rank_sample=lambda: (rank, step),
            record_tp_rank_samples=lambda samples: observed.append((step, samples)),
            choose_decision=choose,
            record_decision=lambda decision: decisions.append(decision.q),
        )
        scheduler = object.__new__(SchedulerSpectreTargetMixin)
        scheduler.is_rejected = False
        scheduler.tp_rank = rank
        scheduler.tp_size = 2
        scheduler.tp_group = SimpleNamespace(
            rank=rank, ranks=[0, 1], all_gather_object=gather,
        )
        scheduler.tp_cpu_group = dist.group.WORLD
        scheduler._get_specstream_runtime = lambda: runtime
        for step in range(len(gates)):
            batch = SimpleNamespace()
            q = scheduler._decide_speculative_num_draft_tokens(batch)
            assert q == 2 + 2 * (step % 4)
            assert batch.specstream_decision.q == q
            assert batch.specstream_mode == "ordinary"

        expected_rounds = [i for i, gate in enumerate(gates) if gate]
        assert gathered_rounds == expected_rounds
        assert observed == [(i, [(0, i), (1, i)]) for i in expected_rounds]
        assert len(decisions) == len(gates)
        assert local_gate_calls == (list(range(len(gates))) if rank == 0 else [])
    finally:
        dist.destroy_process_group()


def test_tp_profile_gate_disagreement_cannot_reorder_collectives(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU Gloo is required for the TP collective regression")
    # No CUDA/model initialization, no fixed port, and a bounded collective
    # timeout: the old all-gather/broadcast mismatch must fail instead of hang.
    rendezvous = (tmp_path / "tp_profile_gloo_init").resolve().as_uri()
    torch.multiprocessing.spawn(
        _run_tp_profile_rank, args=(rendezvous,), nprocs=2, join=True,
    )
