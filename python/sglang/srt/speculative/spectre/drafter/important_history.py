"""Static Draft and Target prefill attention scoring for bounded Draft KV.

Algorithm reference: SnapKV's recent-query attention criterion, as documented
in https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/snapkv_press.py
(Apache-2.0). This is an independent paged, two-pass implementation; it does
not copy the Transformers hook or its per-head cache layout.

Only the final observation queries of explicitly selected layers are scored.
The softmax denominator covers their complete causal prefix. Scores for sink
and recent positions are then excluded, and all heads use one shared block
selection compatible with the engine's single absolute-position slot table.
No model outputs, KV values, RoPE positions or global precision settings are
changed. Scoring happens before the caller's first prune, on live Draft KV.

The bounded scheduler owns selection publication, TP score aggregation and KV
release. This module neither writes the cache nor returns slots to its pool.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import dataclass, field
import math
from numbers import Integral
import time
from typing import Iterable, Mapping, Sequence


_ACTIVE_SCORER = ContextVar("specstream_prefill_importance", default=None)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def merge_block_scores(rank_scores: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Average replicated TP score vectors with deterministic CPU reduction."""
    if not rank_scores:
        raise ValueError("importance aggregation requires at least one rank")
    width = len(rank_scores[0])
    if not width or any(len(row) != width for row in rank_scores):
        raise ValueError("importance TP score shapes differ or are empty")
    rows = []
    for row in rank_scores:
        values = tuple(float(value) for value in row)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("importance block scores must be finite and nonnegative")
        rows.append(values)
    return tuple(math.fsum(row[index] for row in rows) / len(rows)
                 for index in range(width))


def select_important_positions(
    block_scores: Sequence[float], *, prompt_len: int, sink_tokens: int,
    recent_tokens: int, important_budget: int, block_tokens: int = 64,
) -> tuple[int, ...]:
    """Choose a common absolute-position set, never exceeding its KV budget.

    Scores must already exclude sink/recent probability mass (the scorer below
    does this). Blocks are ranked by descending score, then ascending absolute
    start. A final block may be trimmed to the remaining budget. The result is
    sorted for the original sparse causal-attention/absolute-RoPE contract.
    """
    prompt_len = _integer(prompt_len, "prompt_len", 1)
    sink_tokens = _integer(sink_tokens, "sink_tokens")
    recent_tokens = _integer(recent_tokens, "recent_tokens")
    important_budget = _integer(important_budget, "important_budget")
    block_tokens = _integer(block_tokens, "block_tokens", 1)
    values = tuple(float(value) for value in block_scores)
    if len(values) != (prompt_len + block_tokens - 1) // block_tokens:
        raise ValueError("importance score vector does not cover the full prompt")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("importance scores must be finite and nonnegative")
    history_begin = min(sink_tokens, prompt_len)
    history_end = max(history_begin, prompt_len - recent_tokens)
    candidates = []
    for index, score in enumerate(values):
        begin = max(history_begin, index * block_tokens)
        end = min(history_end, (index + 1) * block_tokens)
        if begin < end:
            candidates.append((-score, index, begin, end))
    selected = []
    for _, _, begin, end in sorted(candidates):
        remaining = important_budget - len(selected)
        if remaining <= 0:
            break
        selected.extend(range(begin, min(end, begin + remaining)))
    return tuple(sorted(selected))


@contextmanager
def activate_prefill_importance(scorer):
    """Scope the optional backend observation to one synchronous prefill."""
    if _ACTIVE_SCORER.get() is not None:
        raise RuntimeError("nested prefill importance contexts are not supported")
    if not isinstance(scorer, (PrefillImportanceScorer, TargetPrefillImportanceScorer)):
        raise TypeError("expected a Draft or Target prefill importance scorer")
    token = _ACTIVE_SCORER.set(scorer)
    try:
        yield scorer
    finally:
        _ACTIVE_SCORER.reset(token)


def active_prefill_importance():
    return _ACTIVE_SCORER.get()


def observe_prefill_importance(*, q, key_cache, layer, forward_batch, save_kv_cache=True,
                              causal=True, attention_sinks=None):
    """Optional FA-backend hook after its normal KV write, before any prune."""
    scorer = _ACTIVE_SCORER.get()
    if scorer is None or int(layer.layer_id) not in scorer.layer_ids:
        return
    mode = forward_batch.forward_mode
    unsupported = (
        not save_kv_cache or not causal or attention_sinks is not None
        or not mode.is_extend() or mode.is_target_verify() or layer.is_cross_attention
        or (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
    )
    if isinstance(scorer, TargetPrefillImportanceScorer):
        if not unsupported:
            scorer.observe(
                layer_id=int(layer.layer_id),
                q=q.reshape(-1, layer.tp_q_head_num, layer.head_dim),
                key_cache=key_cache,
                forward_batch=forward_batch,
                softmax_scale=float(layer.scaling),
                logit_cap=float(layer.logit_cap or 0),
            )
        return
    if unsupported or int(forward_batch.batch_size) != 1:
        raise RuntimeError("importance scoring requires one complete ordinary MHA/GQA prefill")
    scorer.observe(
        layer_id=int(layer.layer_id),
        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
        key_cache=key_cache,
        cache_locs=forward_batch.out_cache_loc,
        positions=forward_batch.positions,
        softmax_scale=float(layer.scaling),
        logit_cap=float(layer.logit_cap or 0),
    )


@dataclass
class PrefillImportanceScorer:
    """Two-pass softmax over paged keys with O(observation * key_chunk) memory.

    ``finish`` waits only for this scorer's completion events, materializes its
    small CPU score vector, and releases retained GPU score tensors. The caller
    must run it before publishing importance positions/first cache prune. CPU
    tensors are supported for reference tests, without importing torch here.
    """
    prompt_len: int
    layer_ids: Iterable[int]
    sink_tokens: int = 64
    recent_tokens: int = 1024
    observation_tokens: int = 64
    block_tokens: int = 64
    key_chunk_tokens: int = 1024
    _scores: dict = field(default_factory=dict, init=False, repr=False)
    _events: list = field(default_factory=list, init=False, repr=False)
    _snapshot: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.prompt_len = _integer(self.prompt_len, "prompt_len", 1)
        self.layer_ids = tuple(sorted({_integer(value, "layer ID") for value in self.layer_ids}))
        if not self.layer_ids:
            raise ValueError("select at least one importance layer")
        for name in ("sink_tokens", "recent_tokens"):
            setattr(self, name, _integer(getattr(self, name), name))
        for name in ("observation_tokens", "block_tokens", "key_chunk_tokens"):
            setattr(self, name, _integer(getattr(self, name), name, 1))
        if self.key_chunk_tokens % self.block_tokens:
            raise ValueError("key_chunk_tokens must be a multiple of block_tokens")
        self._context_started = time.perf_counter()
        self._host_enqueue_ms = 0.0
        self._logical_read_bytes = 0
        self._temporary_peak_bytes = 0
        self._binding_checked = False
        self._binding_identity = None

    def observe(self, *, layer_id, q, key_cache, cache_locs, positions,
                softmax_scale=None, logit_cap=0.0):
        import torch

        layer_id = _integer(layer_id, "layer_id")
        if layer_id not in self.layer_ids:
            return
        if self._snapshot is not None or layer_id in self._scores:
            raise RuntimeError("importance layer observed twice or after finish")
        if (q.ndim != 3 or key_cache.ndim != 3 or q.shape[0] != self.prompt_len
                or min(q.shape[1:]) < 1 or min(key_cache.shape[1:]) < 1
                or q.shape[2] != key_cache.shape[2]
                or q.shape[1] < key_cache.shape[1] or q.shape[1] % key_cache.shape[1]):
            raise ValueError("importance Q/K must have full-prompt GQA-compatible rank-3 shapes")
        if q.device != key_cache.device or q.dtype != key_cache.dtype:
            raise ValueError("importance Q/K dtype/device mismatch")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("importance scoring supports FP16/BF16/FP32 MHA/GQA KV only")
        if (cache_locs.ndim != 1 or positions.ndim != 1
                or cache_locs.numel() != self.prompt_len or positions.numel() != self.prompt_len
                or cache_locs.device != q.device or positions.device != q.device
                or cache_locs.dtype not in (torch.int32, torch.int64)
                or positions.dtype not in (torch.int32, torch.int64)):
            raise ValueError("importance needs full-prompt integer position and slot vectors")
        host_started = time.perf_counter()
        start = end = None
        if q.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        if not self._binding_checked:
            if not torch.equal(positions, torch.arange(self.prompt_len, device=q.device, dtype=positions.dtype)):
                raise ValueError("importance bootstrap must use original contiguous absolute positions")
            if (bool((cache_locs < 1).any()) or bool((cache_locs >= key_cache.shape[0]).any())
                    or torch.unique(cache_locs).numel() != self.prompt_len):
                raise ValueError("importance bootstrap slots alias, name dummy slot, or exceed the pool")
            self._binding_checked = True
            self._binding_identity = (positions.data_ptr(), cache_locs.data_ptr(), q.device)
        elif self._binding_identity != (positions.data_ptr(), cache_locs.data_ptr(), q.device):
            raise ValueError("importance layers must share one immutable bootstrap position/slot binding")
        scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
        cap = float(logit_cap)
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(cap) or cap < 0:
            raise ValueError("invalid importance attention scale/softcap")

        observations = min(self.observation_tokens, self.prompt_len)
        hq, hk, dim = int(q.shape[1]), int(key_cache.shape[1]), int(q.shape[2])
        groups = hq // hk
        # Original Q and cache K already carry RoPE. Do not rotate or renumber.
        queries = q[-observations:].detach().float().view(observations, hk, groups, dim)
        queries = queries.permute(1, 2, 0, 3).contiguous()
        query_positions = positions[-observations:]
        maxima = torch.full((hk, groups, observations), float('-inf'), device=q.device, dtype=torch.float32)
        normalizers = torch.zeros_like(maxima)
        blocks = (self.prompt_len + self.block_tokens - 1) // self.block_tokens
        scores = torch.zeros(blocks, device=q.device, dtype=torch.float32)

        def logits_for(begin, finish):
            slots = cache_locs[begin:finish].long()
            keys = key_cache.index_select(0, slots).detach().float().permute(1, 2, 0)
            logits = torch.matmul(queries.reshape(hk, groups * observations, dim), keys)
            logits = logits.view(hk, groups, observations, finish - begin)
            logits.mul_(scale)
            if cap > 0:
                logits.div_(cap).tanh_().mul_(cap)
            key_positions = positions[begin:finish]
            causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            logits.masked_fill_(~causal[None, None], float('-inf'))
            return logits

        # First pass gives the exact causal normalizer without storing O*L.
        for begin in range(0, self.prompt_len, self.key_chunk_tokens):
            finish = min(begin + self.key_chunk_tokens, self.prompt_len)
            logits = logits_for(begin, finish)
            updated_max = torch.maximum(maxima, logits.amax(dim=-1))
            normalizers.mul_((maxima - updated_max).exp())
            logits.sub_(updated_max.unsqueeze(-1)).exp_()
            normalizers.add_(logits.sum(dim=-1))
            maxima = updated_max

        # Second pass reduces probability directly to aligned block scores.
        history_begin = min(self.sink_tokens, self.prompt_len)
        history_end = max(history_begin, self.prompt_len - self.recent_tokens)
        for begin in range(0, self.prompt_len, self.key_chunk_tokens):
            finish = min(begin + self.key_chunk_tokens, self.prompt_len)
            logits = logits_for(begin, finish)
            logits.sub_(maxima.unsqueeze(-1)).exp_().div_(normalizers.unsqueeze(-1))
            token_scores = logits.sum(dim=(0, 1, 2)).div_(hq * observations)
            absolute = torch.arange(begin, finish, device=q.device)
            token_scores.mul_((absolute >= history_begin) & (absolute < history_end))
            padded = ((finish - begin + self.block_tokens - 1) // self.block_tokens) * self.block_tokens
            if padded != finish - begin:
                token_scores = torch.nn.functional.pad(token_scores, (0, padded - (finish - begin)))
            reduced = token_scores.view(-1, self.block_tokens).sum(dim=-1)
            first_block = begin // self.block_tokens
            scores[first_block:first_block + reduced.numel()].add_(reduced)

        if end is not None:
            end.record()
            self._events.append((start, end))
        self._scores[layer_id] = scores
        self._logical_read_bytes += 2 * self.prompt_len * hk * dim * key_cache.element_size()
        largest_chunk = min(self.key_chunk_tokens, self.prompt_len)
        # Explicit conservative working-buffer estimate, not allocator/HBM peak.
        self._temporary_peak_bytes = max(self._temporary_peak_bytes,
            2 * observations * hq * dim * 4
            + largest_chunk * hk * dim * (key_cache.element_size() + 4)
            + 3 * hq * observations * largest_chunk * 4
            + observations * largest_chunk * 2
            + 8 * hq * observations * 4
            + 2 * blocks * 4)
        self._host_enqueue_ms += (time.perf_counter() - host_started) * 1000

    def finish(self):
        import torch

        if self._snapshot is not None:
            return copy.deepcopy(self._snapshot)
        finish_started = time.perf_counter()
        if tuple(sorted(self._scores)) != self.layer_ids:
            raise RuntimeError("importance scorer did not observe every selected complete-prefill layer")
        # Per-layer events also make cross-stream finish safe; no model-wide
        # synchronization or change to global precision configuration occurs.
        for _, end in self._events:
            end.synchronize()
        aggregation_start = aggregation_end = None
        if self._events:
            aggregation_start = torch.cuda.Event(enable_timing=True)
            aggregation_end = torch.cuda.Event(enable_timing=True)
            aggregation_start.record()
        averaged = torch.stack([self._scores[index] for index in self.layer_ids]).mean(dim=0)
        if aggregation_end is not None:
            aggregation_end.record()
        values = tuple(float(value) for value in averaged.cpu().tolist())
        if aggregation_end is not None:
            aggregation_end.synchronize()
        merge_block_scores([values])  # reject nonfinite scoring results before publication
        self._snapshot = dict(
            block_scores=list(values), observed_layers=list(self.layer_ids),
            prompt_len=self.prompt_len, observation_tokens=min(self.observation_tokens, self.prompt_len),
            block_tokens=self.block_tokens, key_chunk_tokens=self.key_chunk_tokens,
            sink_tokens=self.sink_tokens, recent_tokens=self.recent_tokens,
            scoring_cuda_ms=(sum(start.elapsed_time(end) for start, end in self._events)
                             + (aggregation_start.elapsed_time(aggregation_end)
                                if aggregation_end is not None else 0.0)),
            scoring_host_call_ms=self._host_enqueue_ms,
            scoring_wall_ms=self._host_enqueue_ms + (time.perf_counter() - finish_started) * 1000,
            prefill_scoring_context_wall_ms=(time.perf_counter() - self._context_started) * 1000,
            scoring_logical_key_read_bytes=self._logical_read_bytes,
            scoring_score_d2h_payload_bytes=len(values) * 4 if self._events else 0,
            scoring_temporary_peak_bytes_estimate=self._temporary_peak_bytes,
            temporary_peak_note="conservative explicit scoring buffers; excludes allocator reserved and model workspace",
            timing_note="host-call timing may overlap ordinary prefill GPU work; do not add it to end-to-end prefill wall time",
        )
        self._scores.clear()
        self._events.clear()
        return copy.deepcopy(self._snapshot)


class TargetPrefillImportanceScorer:
    """Reuse the same attention criterion for each complete Target prefill.

    A request with a cached prefix, a partial prompt, or incomplete layer
    observations yields no hint. The receiver can retain Draft-only scoring
    for that request. Scores contain no Target K/V and use absolute prompt
    coordinates shared by both models.
    """

    def __init__(
        self, prompt_lens: Mapping[str, int], layer_ids: Iterable[int],
        sink_tokens: int = 64, recent_tokens: int = 1024,
        observation_tokens: int = 32, block_tokens: int = 64,
        key_chunk_tokens: int = 1024,
    ):
        self.layer_ids = tuple(sorted({_integer(value, "layer ID") for value in layer_ids}))
        if not self.layer_ids:
            raise ValueError("select at least one Target importance layer")
        self._scorers = {
            rid: PrefillImportanceScorer(
                prompt_len=length, layer_ids=self.layer_ids, sink_tokens=sink_tokens,
                recent_tokens=recent_tokens, observation_tokens=observation_tokens,
                block_tokens=block_tokens, key_chunk_tokens=key_chunk_tokens,
            )
            for rid, length in prompt_lens.items()
        }
        self._unavailable = set()

    def observe(
        self, *, layer_id, q, key_cache, forward_batch, softmax_scale, logit_cap,
    ):
        if layer_id not in self.layer_ids or getattr(forward_batch, "attn_cp_metadata", None) is not None:
            return
        rids = forward_batch.rids
        lengths = forward_batch.extend_seq_lens_cpu
        prefixes = forward_batch.extend_prefix_lens_cpu
        positions = forward_batch.positions
        slots = forward_batch.out_cache_loc
        if (
            not rids or lengths is None or prefixes is None
            or len(rids) != len(lengths) or len(rids) != len(prefixes)
            or len(set(rids)) != len(rids)
            or positions is None or slots is None
            or any(length < 0 for length in lengths)
            or sum(lengths) != q.shape[0]
            or positions.numel() != q.shape[0] or slots.numel() != q.shape[0]
        ):
            return

        start = 0
        for rid, length, prefix in zip(rids, lengths, prefixes):
            end = start + length
            scorer = self._scorers.get(rid)
            if scorer is not None and rid not in self._unavailable:
                if prefix != 0 or length != scorer.prompt_len or layer_id in scorer._scores:
                    self._unavailable.add(rid)
                else:
                    try:
                        scorer.observe(
                            layer_id=layer_id, q=q[start:end], key_cache=key_cache,
                            cache_locs=slots[start:end], positions=positions[start:end],
                            softmax_scale=softmax_scale, logit_cap=logit_cap,
                        )
                    except ValueError:
                        # Layout/geometry failed the Draft scorer's exact binding
                        # checks. The hint is optional for this Target request.
                        self._unavailable.add(rid)
            start = end

    def finish(self):
        snapshots = {}
        for rid, scorer in self._scorers.items():
            if rid in self._unavailable:
                continue
            if scorer._snapshot is None and len(scorer._scores) != len(self.layer_ids):
                continue
            try:
                snapshots[rid] = scorer.finish()
            except ValueError:
                # Non-finite scores cannot be used to rank history.
                self._unavailable.add(rid)
        return snapshots
