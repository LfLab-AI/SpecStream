from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecStreamRequestMeta:
    rid: str
    req_pool_idx: int
    query_begin: int
    query_end: int
    committed_len: int
    history_len: int
    logical_len: int
    stream_enabled: bool

    @property
    def q_len(self) -> int:
        return self.query_end - self.query_begin

    @property
    def tail_tokens(self) -> int:
        return self.logical_len - self.history_len


@dataclass(frozen=True)
class SpecStreamRoundMeta:
    round_id: int
    q_len: int
    mode: str
    items: tuple[SpecStreamRequestMeta, ...]
    enabled: bool
    cohort_enabled: bool = False
    full_restore_baseline: bool = False
    fallback: bool = False
    fallback_reason: str = ""
    missing_draft_count: int = 0
    coexec_mode: str = "COEXEC"

    @property
    def history_tokens(self) -> int:
        return sum(item.history_len for item in self.items)

    @property
    def context_tokens(self) -> int:
        return sum(item.committed_len for item in self.items)
