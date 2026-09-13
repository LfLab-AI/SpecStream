"""FP32 online-softmax reference for full-history chunk streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch


@dataclass(frozen=True)
class OnlineSoftmaxState:
    max_score: torch.Tensor
    normalizer: torch.Tensor
    weighted_value: torch.Tensor
    # Private per-state device scratch. A cohort reuses it across all History
    # transfers instead of allocating partial split-KV accumulators per chunk.
    # The state belongs to one compute stream; it must not be updated concurrently.
    workspace: dict = field(default_factory=dict, repr=False, compare=False)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query, key, and value must be rank-3 tensors")
    query_count, num_query_heads, head_dim = query.shape
    key_count, num_kv_heads, key_dim = key.shape
    if value.shape[:2] != key.shape[:2]:
        raise ValueError("key and value token/head geometry must match")
    if key_dim != head_dim:
        raise ValueError("query and key head dimensions must match")
    if num_kv_heads < 1 or num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    return (
        query_count,
        key_count,
        num_query_heads,
        num_kv_heads,
        head_dim,
        int(value.shape[-1]),
    )


def _positions(values, expected: int, device: torch.device, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.int64, device=device)
    if result.ndim != 1 or result.numel() != expected:
        raise ValueError(f"{name} must contain exactly {expected} positions")
    return result


def init_online_softmax_state(
    query: torch.Tensor,
    num_kv_heads: int,
    value_head_dim: int | None = None,
) -> OnlineSoftmaxState:
    if query.ndim != 3:
        raise ValueError("query must have shape [Q, Hq, D]")
    query_count, num_query_heads, head_dim = query.shape
    if num_kv_heads < 1 or num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    groups = num_query_heads // num_kv_heads
    value_head_dim = head_dim if value_head_dim is None else int(value_head_dim)
    shape = (query_count, num_kv_heads, groups)
    return OnlineSoftmaxState(
        max_score=torch.full(
            shape, -torch.inf, dtype=torch.float32, device=query.device
        ),
        normalizer=torch.zeros(shape, dtype=torch.float32, device=query.device),
        weighted_value=torch.zeros(
            (*shape, value_head_dim), dtype=torch.float32, device=query.device
        ),
    )


def update_online_softmax_state(
    state: OnlineSoftmaxState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions,
    key_positions,
    *,
    scale: float | None = None,
    causal: bool = True,
    softcap: float = 0.0,
) -> OnlineSoftmaxState:
    """Merge one exact K/V region into an existing online state."""

    (
        query_count,
        key_count,
        num_query_heads,
        num_kv_heads,
        head_dim,
        value_head_dim,
    ) = _validate_inputs(query, key, value)
    groups = num_query_heads // num_kv_heads
    expected_state = (query_count, num_kv_heads, groups)
    if state.max_score.shape != expected_state:
        raise ValueError("online state does not match query/KV head geometry")
    if state.normalizer.shape != expected_state:
        raise ValueError("online normalizer has an invalid shape")
    if state.weighted_value.shape != (*expected_state, value_head_dim):
        raise ValueError("online weighted-value accumulator has an invalid shape")
    if key_count == 0:
        return state

    q_pos = _positions(query_positions, query_count, query.device, "query_positions")
    k_pos = _positions(key_positions, key_count, query.device, "key_positions")
    query_fp32 = query.float().reshape(query_count, num_kv_heads, groups, head_dim)
    scores = torch.einsum("qhgd,khd->qhgk", query_fp32, key.float())
    scores *= float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    if softcap and softcap > 0:
        scores = float(softcap) * torch.tanh(scores / float(softcap))
    if causal:
        allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~allowed[:, None, None, :], -torch.inf)

    chunk_max = scores.amax(dim=-1)
    safe_chunk_max = torch.where(
        torch.isfinite(chunk_max), chunk_max, torch.zeros_like(chunk_max)
    )
    weights = torch.exp(scores - safe_chunk_max.unsqueeze(-1))
    weights = torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))
    chunk_normalizer = weights.sum(dim=-1)
    chunk_weighted_value = torch.einsum("qhgk,khv->qhgv", weights, value.float())

    merged_max = torch.maximum(state.max_score, chunk_max)
    safe_merged_max = torch.where(
        torch.isfinite(merged_max), merged_max, torch.zeros_like(merged_max)
    )
    old_scale = torch.where(
        torch.isfinite(state.max_score),
        torch.exp(state.max_score - safe_merged_max),
        torch.zeros_like(state.max_score),
    )
    chunk_scale = torch.where(
        torch.isfinite(chunk_max),
        torch.exp(chunk_max - safe_merged_max),
        torch.zeros_like(chunk_max),
    )
    return OnlineSoftmaxState(
        max_score=merged_max,
        normalizer=old_scale * state.normalizer + chunk_scale * chunk_normalizer,
        weighted_value=(
            old_scale.unsqueeze(-1) * state.weighted_value
            + chunk_scale.unsqueeze(-1) * chunk_weighted_value
        ),
    )


def finalize_online_softmax_state(
    state: OnlineSoftmaxState,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = state.normalizer.unsqueeze(-1)
    tiny = torch.finfo(torch.float32).tiny
    output = torch.where(
        denominator > 0,
        state.weighted_value / denominator.clamp_min(tiny),
        torch.zeros_like(state.weighted_value),
    )
    lse = torch.where(
        state.normalizer > 0,
        state.max_score + torch.log(state.normalizer.clamp_min(tiny)),
        torch.full_like(state.max_score, -torch.inf),
    )
    query_count, num_kv_heads, groups, value_head_dim = output.shape
    return (
        output.reshape(query_count, num_kv_heads * groups, value_head_dim),
        lse.reshape(query_count, num_kv_heads * groups),
    )


def online_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions,
    key_positions,
    *,
    chunk_size: int,
    scale: float | None = None,
    causal: bool = True,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    _, key_count, _, num_kv_heads, _, value_head_dim = _validate_inputs(
        query, key, value
    )
    key_pos = _positions(key_positions, key_count, query.device, "key_positions")
    state = init_online_softmax_state(query, num_kv_heads, value_head_dim)
    for start in range(0, key_count, chunk_size):
        stop = min(start + chunk_size, key_count)
        state = update_online_softmax_state(
            state,
            query,
            key[start:stop],
            value[start:stop],
            query_positions,
            key_pos[start:stop],
            scale=scale,
            causal=causal,
            softcap=softcap,
        )
    return finalize_online_softmax_state(state)


def full_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions,
    key_positions,
    *,
    scale: float | None = None,
    causal: bool = True,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return online_attention_reference(
        query,
        key,
        value,
        query_positions,
        key_positions,
        chunk_size=max(1, int(key.shape[0])),
        scale=scale,
        causal=causal,
        softcap=softcap,
    )
