# SpecStream dual KV offload with STANDALONE

The Target and Draft use separate GPU KV pools. When the Target pool fills, committed history moves to CPU memory and returns in prefetched chunks for GPU verification. When the Draft pool fills, the Draft computes attention over its CPU and GPU history separately and merges the results. Draft CPU decoding can reuse history across multiple queries. Dynamic verification selects a width from 2, 4, 6, and 8.

This preset enables cross-round Draft lookahead, timing-based admission, and reuse of partially completed candidate sequences. The included admission profile was calibrated on the C24 and C32 LongBench v2 workload on a dual A800 node. Its timing estimates are specific to that setup.

Use an editable install from the `python` directory, Linux with CUDA and a C++ compiler, two A800 GPUs, and local BF16 Qwen3-32B and Qwen3-0.6B model paths. Run the commands below from the repository root with `TARGET_MODEL` and `DRAFT_MODEL` set to those paths. The two GPU KV pools and Target staging buffers together use about 43 GiB per GPU, excluding model weights and other runtime allocations.

```bash
export SPECSTREAM_NATIVE_DYNAMIC_Q=1
export SPECSTREAM_NATIVE_LOGICAL_SLOTS=524288
export SPECSTREAM_NATIVE_TARGET_GPU_SLOTS=261888
export SPECSTREAM_NATIVE_TARGET_STAGE_TOKENS=8192
export SPECSTREAM_NATIVE_DRAFT_GPU_SLOTS=205970
export SPECSTREAM_CPU_ATTN_THREADS=12
export SPECSTREAM_CPU_ATTN_TIMING=1
export SPECSTREAM_GROUPED_CPU_ATTN=1
export SPECSTREAM_GROUPED_DECODE_ONLY=1
export SPECSTREAM_KEEP_PREFILL_GPU=1
export SPECSTREAM_NATIVE_ROUND_PARALLEL=1
export SPECSTREAM_NATIVE_LOOKAHEAD_ADMISSION="$PWD/docs/specstream_spectre/native_lookahead_a800_profile.json"
export SPECSTREAM_NATIVE_SEGMENT_TRACE="$PWD/native_segment.jsonl"
export SPECSTREAM_NATIVE_SEGMENT_STRIDE=3
export SPECSTREAM_NATIVE_LOOKAHEAD_SM_LIMIT=0

python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 7 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 8 \
    --tp-size 2 --dtype bfloat16 --attention-backend fa3 \
    --context-length 16384 --page-size 1 --chunked-prefill-size 8192 \
    --max-total-tokens 261888 --max-running-requests 32 \
    --cuda-graph-max-bs 32 --mem-fraction-static 0.95 \
    --disable-radix-cache --disable-overlap-schedule --skip-server-warmup
```

Segment timing samples every third decode round and feeds the KV transfer slowdown check. Set `SPECSTREAM_NATIVE_LOOKAHEAD_SM_LIMIT=32` for the optional Draft SM ablation with a Green Context capable `sgl-kernel`. For the serial control, set `SPECSTREAM_NATIVE_ROUND_PARALLEL=0` and unset `SPECSTREAM_NATIVE_LOOKAHEAD_ADMISSION`.
