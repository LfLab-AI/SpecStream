from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TargetTieredKVState:
    """Authoritative Target KV lifecycle for one request.

    Only committed, rollback-free KV may move into ``cpu_blocks``.  CUDA
    objects and tensors deliberately live in the rank-local runtime instead of
    this state so that scheduler/request state remains lightweight.
    """

    rid: str
    committed_len: int = 0
    history_len: int = 0
    logical_len: int = 0
    tail_start: int = 0
    cpu_blocks: list[int] = field(default_factory=list)
    gpu_tail_slots: list[int] = field(default_factory=list)
    stream_enabled: bool = False
    seal_inflight: bool = False
    round_id: int = 0

    def check(self, seal_granularity: int = 1) -> None:
        if seal_granularity < 1:
            raise ValueError("seal_granularity must be positive")
        assert 0 <= self.history_len <= self.committed_len <= self.logical_len
        assert self.tail_start == self.history_len
        assert self.history_len % seal_granularity == 0
        if self.stream_enabled:
            assert self.history_len > 0

    def begin_round(self, committed_len: int, logical_len: int) -> int:
        if committed_len < self.history_len:
            raise AssertionError("committed length cannot move behind sealed history")
        if logical_len < committed_len:
            raise AssertionError(
                "logical length cannot be shorter than committed length"
            )
        self.committed_len = committed_len
        self.logical_len = logical_len
        self.round_id += 1
        return self.round_id

    def finish_round(self, committed_len: int) -> None:
        if committed_len < self.history_len:
            raise AssertionError("rollback reached sealed CPU history")
        self.committed_len = committed_len
        self.logical_len = committed_len

    def mark_sealed(self, seal_end: int, cpu_block_ids: list[int]) -> None:
        if self.seal_inflight is False:
            raise AssertionError("seal completion without an in-flight seal")
        if not self.history_len < seal_end <= self.committed_len:
            raise AssertionError("sealed range must be a new committed prefix")
        self.cpu_blocks.extend(cpu_block_ids)
        self.history_len = seal_end
        self.tail_start = seal_end
        self.stream_enabled = True
        self.seal_inflight = False

    def start_seal(self) -> None:
        if self.seal_inflight:
            raise AssertionError("only one seal may be in flight per request")
        self.seal_inflight = True
