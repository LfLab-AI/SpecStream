"""Evict committed KV outside an exact GPU tail at stable batch boundaries."""

import json
import os
from pathlib import Path
import time

import numpy as np
import torch


class NativeExactKVTail:
    def __init__(self, worker, target, draft, size):
        if size != 512:
            raise ValueError("SPECSTREAM_NATIVE_STRICT_GPU_TAIL must be 512 when enabled")
        if target is None or draft is None or target.min_gpu_tail != size:
            raise ValueError("Exact KV tails require split Target and Draft pools and matching Target tail protection")
        if worker._native_joint_resize is not None:
            raise ValueError("Exact KV tails currently require fixed physical addressing")
        if not worker.server_args.disable_cuda_graph or not worker.server_args.disable_piecewise_cuda_graph:
            raise ValueError("Exact KV tails require CUDA graphs to be disabled")
        self.size = size
        self.target = target
        self.draft = draft
        self.allocator = worker.token_to_kv_pool_allocator
        self.rank = worker.tp_rank
        self.decode_round = 0
        trace = os.getenv("SPECSTREAM_NATIVE_STRICT_GPU_TAIL_TRACE")
        self.trace = None
        if trace:
            path = Path(trace)
            self.trace = path.with_name(f"{path.stem}.tp{self.rank}{path.suffix or '.jsonl'}")

    @staticmethod
    def _placement(pressure, logical, resident, request_id, name, require_exact):
        if pressure.active:
            gpu = pressure.gpu_map[logical] != 0
            cpu = pressure.cpu_map[logical] != 0
        else:
            gpu = logical <= pressure.gpu_slots
            cpu = np.zeros_like(gpu)
        prefix = len(logical) - resident
        if np.any(gpu == cpu) or not gpu[prefix:].all():
            raise RuntimeError(f"{name} {request_id} has a missing, duplicate, or misplaced committed KV slot")
        on_cpu = int(np.count_nonzero(cpu))
        if on_cpu > prefix or not cpu[:on_cpu].all():
            raise RuntimeError(f"{name} {request_id} CPU KV must be a prefix of the committed sequence")
        if require_exact and on_cpu != prefix:
            raise RuntimeError(f"{name} {request_id} has {len(logical) - on_cpu} GPU KV slots, expected {resident}")
        return logical[on_cpu:prefix]

    def check(self, batch, boundary):
        if boundary == "before_decode":
            self.decode_round += 1
        owners = [
            req for req in batch.reqs
            if req.req_pool_idx is not None and req.kv_committed_len > 0 and not req.finished()
            and (boundary != "prefill" or (
                req.is_chunked <= 0 and req.kv_committed_len >= len(req.origin_input_ids)
            ))
        ]
        if not owners:
            return
        lengths = [int(req.kv_committed_len) for req in owners]
        indices = torch.tensor([req.req_pool_idx for req in owners], dtype=torch.int64,
                               device=self.target.pool.device)
        table = batch.req_to_token_pool.req_to_token[indices, :max(lengths)].cpu().numpy()
        rows = [row[:length] for row, length in zip(table, lengths)]
        if any(np.any(row <= 0) for row in rows):
            raise RuntimeError("An exact KV tail cannot move unallocated or speculative slots")
        target_victims, draft_victims, observations = [], [], []
        for req, logical in zip(owners, rows):
            resident = min(self.size, len(logical))
            exact = boundary == "before_decode"
            target_victims.append(self._placement(self.target, logical, resident, req.rid, "Target", exact))
            draft_victims.append(self._placement(self.draft, logical, resident, req.rid, "Draft", exact))
            observations.append(dict(request_id=req.rid, committed=len(logical),
                                     target_gpu=resident, draft_gpu=resident))

        target_victims = np.concatenate(target_victims)
        draft_victims = np.concatenate(draft_victims)
        if len(target_victims):
            if not self.target.active:
                self.target.activate(self.allocator.free_pages)
            eligible = np.zeros(self.target.logical_slots + 1, dtype=bool)
            eligible[target_victims] = True
            self.target._migrate_committed(eligible, len(target_victims))
        if len(draft_victims):
            if not self.draft.active:
                self.draft.activate(self.allocator.free_pages)
            self.draft._migrate_gpu_victims(draft_victims)

        for req, logical, observed in zip(owners, rows, observations):
            if len(target_victims):
                self._placement(self.target, logical, observed["target_gpu"], req.rid, "Target", True)
            if len(draft_victims):
                self._placement(self.draft, logical, observed["draft_gpu"], req.rid, "Draft", True)
        if self.trace is not None:
            with self.trace.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(
                    time_ns=time.time_ns(), round_id=self.decode_round, boundary=boundary, rank=self.rank,
                    batch_size=len(batch.reqs), target_moved=len(target_victims),
                    draft_moved=len(draft_victims), requests=observations,
                ), separators=(",", ":")) + "\n")
