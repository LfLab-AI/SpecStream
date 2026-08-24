"""Triton fused updates for packed SpecStream History chunks."""

from __future__ import annotations

import math

import torch

from sglang.srt.speculative.spectre.specstream.online_softmax import (
    OnlineSoftmaxState,
    update_online_softmax_state,
)

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _multi_query_tiled_online_attention_kernel(
        query,
        key,
        value,
        token_indices,
        state_max,
        state_norm,
        state_acc,
        stride_q0,
        stride_q1,
        stride_q2,
        stride_k0,
        stride_k1,
        stride_k2,
        stride_v0,
        stride_v1,
        stride_v2,
        key_count,
        query_position_start,
        key_position_start,
        query_count: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        groups: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        CAUSAL: tl.constexpr,
        INDIRECT_KV: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """FlashAttention-style tile over Q/GQA rows for one KV head.

        Unlike the scalar-row fallback below, every program loads one K/V tile
        and reuses it for BLOCK_M verification/GQA query rows.  The online
        state is kept in natural-log score space so History and the indirect
        paged Tail/Frontier path can be merged without representation changes.
        """

        query_group_row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        kv_head = tl.program_id(1)
        row_mask = query_group_row < query_count * groups
        query_index = query_group_row // groups
        query_group = query_group_row - query_index * groups
        query_head = kv_head * groups + query_group
        d = tl.arange(0, BLOCK_D)
        d_mask = d < head_dim

        q_ptrs = (
            query
            + query_index[:, None] * stride_q0
            + query_head[:, None] * stride_q1
            + d[None, :] * stride_q2
        )
        q = tl.load(
            q_ptrs,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        # [Q, Hkv, G] is contiguous, but rows belonging to a fixed KV head
        # are strided across verification queries.
        state_row = query_index * num_query_heads + kv_head * groups + query_group
        m = tl.load(state_max + state_row, mask=row_mask, other=-float("inf")).to(
            tl.float32
        )
        normalizer = tl.load(state_norm + state_row, mask=row_mask, other=0.0).to(
            tl.float32
        )
        acc = tl.load(
            state_acc + state_row[:, None] * head_dim + d[None, :],
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        log2e = 1.4426950408889634
        for start in tl.range(0, key_count, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            n_mask = n < key_count
            if INDIRECT_KV:
                token_row = tl.load(token_indices + n, mask=n_mask, other=0)
            else:
                token_row = n

            # K is [D, N] and V is [N, D], matching Tensor Core dot layouts.
            k_ptrs = (
                key
                + token_row[None, :] * stride_k0
                + kv_head * stride_k1
                + d[:, None] * stride_k2
            )
            v_ptrs = (
                value
                + token_row[:, None] * stride_v0
                + kv_head * stride_v1
                + d[None, :] * stride_v2
            )
            key_tile = tl.load(
                k_ptrs,
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            value_tile = tl.load(
                v_ptrs,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            scores = tl.dot(q, key_tile) * scale
            if CAUSAL:
                causal_mask = (
                    key_position_start + n[None, :]
                    <= query_position_start + query_index[:, None]
                )
            else:
                causal_mask = True
            scores = tl.where(
                row_mask[:, None] & n_mask[None, :] & causal_mask,
                scores,
                -float("inf"),
            )
            block_max = tl.max(scores, axis=1)
            merged_max = tl.maximum(m, block_max)
            safe_max = tl.where(row_mask, merged_max, 0.0)
            alpha = tl.where(
                m == -float("inf"),
                0.0,
                tl.exp2((m - safe_max) * log2e),
            )
            weights = tl.exp2((scores - safe_max[:, None]) * log2e)
            weights = tl.where(
                row_mask[:, None] & n_mask[None, :] & causal_mask, weights, 0.0
            )

            normalizer = normalizer * alpha + tl.sum(weights, axis=1)
            acc *= alpha[:, None]
            if USE_BF16:
                acc = tl.dot(weights.to(tl.bfloat16), value_tile, acc)
            else:
                acc = tl.dot(weights.to(tl.float16), value_tile, acc)
            m = merged_max

        tl.store(state_max + state_row, m, mask=row_mask)
        tl.store(state_norm + state_row, normalizer, mask=row_mask)
        tl.store(
            state_acc + state_row[:, None] * head_dim + d[None, :],
            acc,
            mask=row_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _packed_history_update_kernel(
        query,
        packed_kv,
        state_max,
        state_norm,
        state_acc,
        stride_q0,
        stride_q1,
        stride_q2,
        key_count,
        num_query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        groups: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        query_index = row // num_query_heads
        query_head = row - query_index * num_query_heads
        kv_head = query_head // groups
        d = tl.arange(0, BLOCK_D)
        d_mask = d < head_dim
        q = tl.load(
            query + query_index * stride_q0 + query_head * stride_q1 + d * stride_q2,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        m = tl.load(state_max + row).to(tl.float32)
        normalizer = tl.load(state_norm + row).to(tl.float32)
        acc = tl.load(state_acc + row * head_dim + d, mask=d_mask, other=0.0).to(
            tl.float32
        )

        for start in tl.range(0, key_count, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            n_mask = n < key_count
            base_k = ((n * 2) * num_kv_heads + kv_head) * head_dim
            base_v = ((n * 2 + 1) * num_kv_heads + kv_head) * head_dim
            key = tl.load(
                packed_kv + base_k[:, None] + d[None, :],
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * q[None, :], axis=1) * scale
            scores = tl.where(n_mask, scores, -float("inf"))
            chunk_max = tl.max(scores, axis=0)
            weights = tl.exp(scores - chunk_max)
            weights = tl.where(n_mask, weights, 0.0)
            chunk_norm = tl.sum(weights, axis=0)
            value = tl.load(
                packed_kv + base_v[:, None] + d[None, :],
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            chunk_acc = tl.sum(weights[:, None] * value, axis=0)
            merged_max = tl.maximum(m, chunk_max)
            old_scale = tl.where(m == -float("inf"), 0.0, tl.exp(m - merged_max))
            chunk_scale = tl.exp(chunk_max - merged_max)
            normalizer = old_scale * normalizer + chunk_scale * chunk_norm
            acc = old_scale * acc + chunk_scale * chunk_acc
            m = merged_max

        tl.store(state_max + row, m)
        tl.store(state_norm + row, normalizer)
        tl.store(state_acc + row * head_dim + d, acc, mask=d_mask)

    @triton.jit
    def _packed_history_cohort_update_kernel(
        query,
        packed_kv,
        valid_tokens,
        state_max,
        state_norm,
        state_acc,
        stride_qb,
        stride_qq,
        stride_qh,
        stride_qd,
        stride_kvb,
        stride_kvn,
        stride_kvpair,
        stride_kvh,
        stride_kvd,
        query_count: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        groups: tl.constexpr,
        chunk_tokens,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        USE_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Tensor-Core cohort tile: one program reuses K/V across Q/GQA rows."""

        query_group_row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        kv_head = tl.program_id(1)
        item = tl.program_id(2)
        row_mask = query_group_row < query_count * groups
        query_index = query_group_row // groups
        query_group = query_group_row - query_index * groups
        query_head = kv_head * groups + query_group
        d = tl.arange(0, BLOCK_D)
        d_mask = d < head_dim

        q_ptrs = (
            query
            + item * stride_qb
            + query_index[:, None] * stride_qq
            + query_head[:, None] * stride_qh
            + d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

        rows_per_item = query_count * num_query_heads
        state_row = (
            item * rows_per_item
            + query_index * num_query_heads
            + kv_head * groups
            + query_group
        )
        m = tl.load(state_max + state_row, mask=row_mask, other=-float("inf")).to(
            tl.float32
        )
        normalizer = tl.load(state_norm + state_row, mask=row_mask, other=0.0).to(
            tl.float32
        )
        valid = tl.load(valid_tokens + item)

        acc = tl.load(
            state_acc + state_row[:, None] * head_dim + d[None, :],
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        log2e = 1.4426950408889634
        for start in tl.range(0, chunk_tokens, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            n_mask = n < valid
            key_ptrs = (
                packed_kv
                + item * stride_kvb
                + n[None, :] * stride_kvn
                + 0 * stride_kvpair
                + kv_head * stride_kvh
                + d[:, None] * stride_kvd
            )
            value_ptrs = (
                packed_kv
                + item * stride_kvb
                + n[:, None] * stride_kvn
                + 1 * stride_kvpair
                + kv_head * stride_kvh
                + d[None, :] * stride_kvd
            )
            key_tile = tl.load(
                key_ptrs,
                mask=d_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            value_tile = tl.load(
                value_ptrs,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            scores = tl.dot(q, key_tile) * scale
            scores = tl.where(
                row_mask[:, None] & n_mask[None, :], scores, -float("inf")
            )
            block_max = tl.max(scores, axis=1)
            merged_max = tl.maximum(m, block_max)
            safe_max = tl.where(row_mask, merged_max, 0.0)
            alpha = tl.where(
                m == -float("inf"),
                0.0,
                tl.exp2((m - safe_max) * log2e),
            )
            weights = tl.exp2((scores - safe_max[:, None]) * log2e)
            weights = tl.where(row_mask[:, None] & n_mask[None, :], weights, 0.0)
            normalizer = normalizer * alpha + tl.sum(weights, axis=1)
            acc *= alpha[:, None]
            if USE_BF16:
                acc = tl.dot(weights.to(tl.bfloat16), value_tile, acc)
            else:
                acc = tl.dot(weights.to(tl.float16), value_tile, acc)
            m = merged_max

        tl.store(state_max + state_row, m, mask=row_mask)
        tl.store(state_norm + state_row, normalizer, mask=row_mask)
        tl.store(
            state_acc + state_row[:, None] * head_dim + d[None, :],
            acc,
            mask=row_mask[:, None] & d_mask[None, :],
        )


def triton_fused_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _launch_multi_query_tiled_update(
    state: OnlineSoftmaxState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None,
    causal: bool,
    query_position_start: int,
    key_position_start: int,
    token_indices: torch.Tensor | None = None,
) -> bool:
    """Launch the Tensor-Core tiled path, returning False for safe fallback."""

    if (
        triton is None
        or not query.is_cuda
        or query.dtype not in (torch.float16, torch.bfloat16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or int(query.shape[-1]) not in (64, 128)
        or query.stride(-1) != 1
        or key.stride(-1) != 1
        or value.stride(-1) != 1
    ):
        return False
    if token_indices is not None:
        if token_indices.device != query.device:
            raise ValueError("Tail token indices must be on the query device")
        if token_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Tail token indices must be int32 or int64")
        if not token_indices.is_contiguous():
            token_indices = token_indices.contiguous()

    query_count, num_query_heads, head_dim = map(int, query.shape)
    _, num_kv_heads, key_head_dim = map(int, key.shape)
    key_count = (
        int(key.shape[0]) if token_indices is None else int(token_indices.numel())
    )
    if value.shape != key.shape or key_head_dim != head_dim:
        raise ValueError("tiled SpecStream K/V geometry mismatch")
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = num_query_heads // num_kv_heads
    expected = (query_count, num_kv_heads, groups)
    if state.max_score.shape != expected or state.normalizer.shape != expected:
        raise ValueError("online-softmax state geometry mismatch")
    if state.weighted_value.shape != (*expected, head_dim):
        raise ValueError("tiled kernel requires value_head_dim == head_dim")

    query_group_rows = query_count * groups
    block_m = 16 if query_group_rows <= 16 else 32
    block_n = 64
    grid = (triton.cdiv(query_group_rows, block_m), num_kv_heads)
    attention_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    _multi_query_tiled_online_attention_kernel[grid](
        query,
        key,
        value,
        key if token_indices is None else token_indices,
        state.max_score,
        state.normalizer,
        state.weighted_value,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_count,
        int(query_position_start),
        int(key_position_start),
        query_count=query_count,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        groups=groups,
        head_dim=head_dim,
        scale=attention_scale,
        CAUSAL=bool(causal),
        INDIRECT_KV=token_indices is not None,
        USE_BF16=query.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=2,
    )
    return True


def update_packed_history_state(
    state: OnlineSoftmaxState,
    query: torch.Tensor,
    packed_kv: torch.Tensor,
    *,
    scale: float | None = None,
    require_fused_cuda: bool = True,
) -> tuple[OnlineSoftmaxState, bool]:
    if packed_kv.ndim != 4 or packed_kv.shape[1] != 2:
        raise ValueError("packed KV must have shape [tokens,2,Hkv,head_dim]")
    key_count, _, num_kv_heads, head_dim = packed_kv.shape
    if query.ndim != 3 or query.shape[-1] != head_dim:
        raise ValueError("query geometry does not match packed KV")
    if query.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by packed KV heads")
    if query.device != packed_kv.device:
        raise ValueError("query and packed KV must be on the same device")
    if key_count == 0:
        return state, False

    can_fuse = bool(query.is_cuda and triton is not None)
    if query.is_cuda and require_fused_cuda and not can_fuse:
        raise RuntimeError("optimized SpecStream attention requires Triton on CUDA")
    if not can_fuse:
        updated = update_online_softmax_state(
            state,
            query,
            packed_kv[:, 0],
            packed_kv[:, 1],
            tuple(range(query.shape[0])),
            tuple(range(key_count)),
            scale=scale,
            causal=False,
        )
        return updated, False

    if query.stride(-1) != 1 or not packed_kv.is_contiguous():
        raise RuntimeError("fused SpecStream requires unit-stride Q and packed KV")
    num_query_heads = int(query.shape[1])
    groups = num_query_heads // int(num_kv_heads)
    expected = (int(query.shape[0]), int(num_kv_heads), groups)
    if state.max_score.shape != expected or state.normalizer.shape != expected:
        raise ValueError("online-softmax state geometry mismatch")
    if state.weighted_value.shape != (*expected, int(head_dim)):
        raise ValueError("fused kernel requires value_head_dim == head_dim")

    if _launch_multi_query_tiled_update(
        state,
        query,
        packed_kv[:, 0],
        packed_kv[:, 1],
        scale=scale,
        causal=False,
        query_position_start=0,
        key_position_start=0,
        token_indices=None,
    ):
        pass
    else:
        block_d = triton.next_power_of_2(int(head_dim))
        attention_scale = (
            float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
        )
        _packed_history_update_kernel[(int(query.shape[0]) * num_query_heads,)](
            query,
            packed_kv,
            state.max_score,
            state.normalizer,
            state.weighted_value,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            int(key_count),
            num_query_heads=num_query_heads,
            num_kv_heads=int(num_kv_heads),
            groups=groups,
            head_dim=int(head_dim),
            scale=attention_scale,
            BLOCK_N=32 if int(head_dim) >= 128 else 64,
            BLOCK_D=block_d,
            num_warps=4,
        )
    return state, True


def update_gpu_tail_state(
    state: OnlineSoftmaxState,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_position_start: int,
    key_position_start: int,
    token_indices: torch.Tensor | None = None,
    scale: float | None = None,
    require_fused_cuda: bool = True,
) -> tuple[OnlineSoftmaxState, bool]:
    """Merge the causal GPU Tail/Frontier using the same tiled online state."""

    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("Tail K/V must have matching [tokens,Hkv,D] shapes")
    if query.ndim != 3 or query.shape[-1] != key.shape[-1]:
        raise ValueError("Tail query geometry mismatch")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Tail Q/K/V must be on the same device")
    key_count = (
        int(key.shape[0]) if token_indices is None else int(token_indices.numel())
    )
    if key_count == 0:
        return state, False

    fused = _launch_multi_query_tiled_update(
        state,
        query,
        key,
        value,
        scale=scale,
        causal=True,
        query_position_start=query_position_start,
        key_position_start=key_position_start,
        token_indices=token_indices,
    )
    if fused:
        return state, True
    if query.is_cuda and require_fused_cuda:
        raise RuntimeError("optimized SpecStream Tail attention requires Triton")
    if token_indices is not None:
        key = key.index_select(0, token_indices.long())
        value = value.index_select(0, token_indices.long())
    updated = update_online_softmax_state(
        state,
        query,
        key,
        value,
        tuple(range(query_position_start, query_position_start + query.shape[0])),
        tuple(range(key_position_start, key_position_start + key_count)),
        scale=scale,
        causal=True,
    )
    return updated, False


def stack_packed_history_cohort_states(
    states: list[OnlineSoftmaxState],
) -> OnlineSoftmaxState:
    if not states:
        raise ValueError("cohort state list cannot be empty")
    return OnlineSoftmaxState(
        torch.stack([state.max_score for state in states]).contiguous(),
        torch.stack([state.normalizer for state in states]).contiguous(),
        torch.stack([state.weighted_value for state in states]).contiguous(),
    )


def split_packed_history_cohort_state(
    state: OnlineSoftmaxState,
) -> list[OnlineSoftmaxState]:
    return [
        OnlineSoftmaxState(
            state.max_score[index],
            state.normalizer[index],
            state.weighted_value[index],
        )
        for index in range(int(state.max_score.shape[0]))
    ]


def update_packed_history_cohort_batched(
    state: OnlineSoftmaxState,
    queries: torch.Tensor,
    packed_kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    scale: float | None = None,
    require_fused_cuda: bool = True,
) -> tuple[OnlineSoftmaxState, bool]:
    """Update an already-stacked cohort without per-chunk state repacking."""

    if queries.ndim != 4 or packed_kv.ndim != 5 or packed_kv.shape[2] != 2:
        raise ValueError("cohort Q/KV geometry must be [B,Q,H,D]/[B,N,2,H,D]")
    cohort_size, query_count, num_query_heads, head_dim = queries.shape
    if valid_tokens.numel() != cohort_size:
        raise ValueError("cohort valid-token count does not match")
    num_kv_heads = int(packed_kv.shape[3])
    if packed_kv.shape[-1] != head_dim or num_query_heads % num_kv_heads:
        raise ValueError("cohort GQA geometry mismatch")
    if packed_kv.dtype != queries.dtype:
        raise ValueError("cohort Q/KV dtypes must match")

    groups = num_query_heads // num_kv_heads
    expected = (cohort_size, query_count, num_kv_heads, groups)
    if state.max_score.shape != expected or state.normalizer.shape != expected:
        raise ValueError("batched online-softmax state geometry mismatch")
    if state.weighted_value.shape != (*expected, head_dim):
        raise ValueError("batched cohort requires value_head_dim == head_dim")

    can_fuse = bool(queries.is_cuda and triton is not None)
    if queries.is_cuda and require_fused_cuda and not can_fuse:
        raise RuntimeError("optimized SpecStream cohort requires Triton on CUDA")
    if not can_fuse:
        updated = []
        for index, item_state in enumerate(split_packed_history_cohort_state(state)):
            valid = int(valid_tokens[index].item())
            item, _ = update_packed_history_state(
                item_state,
                queries[index],
                packed_kv[index, :valid],
                scale=scale,
                require_fused_cuda=False,
            )
            updated.append(item)
        return stack_packed_history_cohort_states(updated), False

    if queries.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("fused cohort requires FP16 or BF16 queries")
    if queries.stride(-1) != 1 or not packed_kv.is_contiguous():
        raise RuntimeError(
            "fused cohort requires contiguous packed KV and unit-stride Q"
        )
    if valid_tokens.device != queries.device:
        raise ValueError("cohort valid-token metadata must be on the query device")
    if valid_tokens.dtype != torch.int32:
        valid_tokens = valid_tokens.to(dtype=torch.int32)
    if not valid_tokens.is_contiguous():
        valid_tokens = valid_tokens.contiguous()
    if not queries.is_contiguous():
        queries = queries.contiguous()

    block_d = triton.next_power_of_2(int(head_dim))
    block_n = 32 if int(head_dim) >= 128 else 64
    block_m = max(
        16,
        min(64, triton.next_power_of_2(int(query_count) * int(groups))),
    )
    attention_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    grid = (
        triton.cdiv(int(query_count) * int(groups), block_m),
        int(num_kv_heads),
        int(cohort_size),
    )
    _packed_history_cohort_update_kernel[grid](
        queries,
        packed_kv,
        valid_tokens,
        state.max_score,
        state.normalizer,
        state.weighted_value,
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        queries.stride(3),
        packed_kv.stride(0),
        packed_kv.stride(1),
        packed_kv.stride(2),
        packed_kv.stride(3),
        packed_kv.stride(4),
        query_count=query_count,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        groups=groups,
        chunk_tokens=int(packed_kv.shape[1]),
        head_dim=int(head_dim),
        scale=attention_scale,
        USE_BF16=queries.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return state, True


def update_packed_history_cohort(
    states: list[OnlineSoftmaxState],
    queries: torch.Tensor,
    packed_kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    scale: float | None = None,
    require_fused_cuda: bool = True,
) -> tuple[list[OnlineSoftmaxState], bool]:
    """Compatibility wrapper for one-shot cohort updates."""

    if len(states) != int(queries.shape[0]):
        raise ValueError("cohort state count does not match query batch")
    batched = stack_packed_history_cohort_states(states)
    batched, fused = update_packed_history_cohort_batched(
        batched,
        queries,
        packed_kv,
        valid_tokens,
        scale=scale,
        require_fused_cuda=require_fused_cuda,
    )
    return split_packed_history_cohort_state(batched), fused
