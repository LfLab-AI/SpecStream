# SpecStream dual KV offload with STANDALONE

The Target and Draft use separate GPU KV pools. When the Target pool fills, committed history moves to CPU memory and returns in prefetched chunks for GPU verification. When the Draft pool fills, the Draft computes attention over its CPU and GPU history separately and merges the results. Draft CPU decoding can reuse history across multiple queries. Dynamic verification selects a width from 2, 4, 6, and 8.

The following launch reproduces the serving configuration used for the 32-client LongBench v2 run. Use an editable install from the `python` directory, Linux with CUDA and a C++ compiler, two A800 GPUs, and local BF16 Qwen3-32B and Qwen3-0.6B model paths. Set `TARGET_MODEL` and `DRAFT_MODEL` to those paths. The two GPU KV pools and Target staging buffers together use about 43 GiB per GPU, excluding model weights and other runtime allocations.

```bash
export SPECSTREAM_NATIVE_DYNAMIC_Q=1
export SPECSTREAM_NATIVE_LOGICAL_SLOTS=524288
export SPECSTREAM_NATIVE_TARGET_GPU_SLOTS=261888
export SPECSTREAM_NATIVE_TARGET_STAGE_TOKENS=8192
export SPECSTREAM_NATIVE_DRAFT_GPU_SLOTS=205970
export SPECSTREAM_CPU_ATTN_THREADS=12
export SPECSTREAM_GROUPED_CPU_ATTN=1
export SPECSTREAM_GROUPED_DECODE_ONLY=1
export SPECSTREAM_KEEP_PREFILL_GPU=1
export SPECSTREAM_NATIVE_ROUND_PARALLEL=0

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

The measured launch runs Draft generation and Target verification serially within each round. For the optional timing breakdown, set `SPECSTREAM_NATIVE_SEGMENT_TRACE` to a writable JSONL path and `SPECSTREAM_NATIVE_SEGMENT_STRIDE=3` to sample every third decode round.
