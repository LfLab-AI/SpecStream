"""FA3 prefill and host-resident paged attention for ordinary Qwen3 Draft decoding."""

import json
import os
import time

import torch


class CpuDraftBackend:
    phase_reader = None
    tail_kv_only = False

    @staticmethod
    def _contiguous_prefill_start(slots):
        # Called once per complete prefill, while the scheduler owns every slot.
        # Comparing only the two endpoints would miss holes or permutations.
        if slots.numel() < 2:
            return None
        first = int(slots[0])
        if first <= 0 or int(slots[-1]) != first + slots.numel() - 1:
            return None
        return first if torch.equal(slots[1:], slots[:-1] + 1) else None

    def __init__(self, runner, flash_prefill):
        pool = runner.token_to_kv_pool
        self.history = pool.cpu_draft_history
        if (self.history.keys.device.type != "cpu" or self.history.values.device.type != "cpu"
                or pool.get_kv_size_bytes() != (0, 0)):
            raise RuntimeError("CPU Draft requires a host-only history pool")
        self.first_layer = pool.start_layer
        self.flash_prefill = flash_prefill
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.req_pool = runner.req_to_token_pool
        self.rank = runner.tp_rank
        self.buffers = {}
        self.download_done = torch.cuda.Event()
        self.upload_done = torch.cuda.Event()
        self.upload_in_flight = False
        self.metrics_path = os.environ.get("SPECSTREAM_DRAFT_CPU_TRACE_PATH")
        self.causal_cpu_attention = os.environ.get("SPECSTREAM_ROUND_CPU_ATTN") == "1"
        self.tail_kv_only = os.environ.get("SPECSTREAM_DRAFT_TAIL_KV_ONLY") == "1"
        if os.environ.get("SPECSTREAM_DRAFT_LAYER_WINDOW_DIR"):
            from .draft_layer_phase import make_window_reader

            self.phase_reader = make_window_reader(runner.tp_size)
        self.layer_phase = None

    def prepare(self, batch):
        self.layer_phase = None
        if batch.forward_mode.is_idle():
            return
        decode = batch.forward_mode.is_decode()
        if self.tail_kv_only:
            from .draft_layer_phase import in_draft_tail_forward

            batch._cpu_draft_tail_writer = self if decode and in_draft_tail_forward() else None
            batch._cpu_draft_tail_kv_complete = False
        if not decode and not batch.forward_mode.is_extend():
            raise RuntimeError("CPU Draft attention only accepts ordinary extend and decode batches")
        counts = ([1] * batch.batch_size if decode else list(map(int, batch.extend_seq_lens_cpu)))
        full = batch.seq_lens_cpu
        if full is None:
            full = batch.seq_lens.to("cpu")
        full = torch.as_tensor(full, dtype=torch.int64, device="cpu").contiguous()
        lengths = full.tolist()
        if len(counts) != len(lengths) or any(not 0 < n <= l for n, l in zip(counts, lengths)):
            raise ValueError("Invalid Draft query lengths for CPU attention")
        self.direct_prefill = not decode and counts == lengths
        mirrored = getattr(self.req_pool, "_draft_cpu_forward_metadata", None)
        if mirrored is None:
            self.locations = batch.out_cache_loc.to("cpu", dtype=torch.int64).contiguous()
        else:
            host_rows, self.locations = mirrored
            if len(host_rows) != len(counts) or any(row.numel() != l for row, l in zip(host_rows, lengths)):
                raise RuntimeError("CPU Draft slot mirror does not match the scheduled sequence lengths")
        # The scheduler materializes the fresh allocation IDs on the CPU once,
        # even when ready-to-use host rows avoid the full table transfer.
        metadata_bytes = self.locations.numel() * batch.out_cache_loc.element_size()
        if sum(counts) != self.locations.numel():
            raise ValueError("CPU Draft query count differs from allocated slot count")

        self.prefill_first_slot = self._contiguous_prefill_start(self.locations) if self.direct_prefill else None
        if self.direct_prefill:
            self.max_new = max(counts)
            self.cu = torch.nn.functional.pad(
                torch.tensor(counts, dtype=torch.int32, device="cpu").cumsum(0, dtype=torch.int32), (1, 0)
            ).to(batch.input_ids.device)
            mode = "prefill"
        else:
            # Capture the newly allocated slots along with the old slots.  The scheduler
            # owns the row and may change it after this forward (including on rollback).
            if mirrored is None:
                device_rows = self.req_to_token[batch.req_pool_indices.long(), :max(lengths)]
                rows = device_rows.cpu().long().contiguous()
                metadata_bytes += device_rows.numel() * device_rows.element_size()
            elif len(host_rows) == 1:
                rows = host_rows[0].long().view(1, -1)
            else:
                rows = torch.nn.utils.rnn.pad_sequence(host_rows, batch_first=True).long().contiguous()
            if self.causal_cpu_attention:
                self.slots, self.lengths = rows, full
                self.query_offsets = torch.nn.functional.pad(
                    torch.tensor(counts, dtype=torch.int64, device="cpu").cumsum(0), (1, 0),
                )
            elif all(n == 1 for n in counts):
                self.slots, self.lengths = rows, full
            else:
                # Catch-up/replay queries are request-major and causal. Replicating the
                # index row never duplicates the underlying KV. The common B=1 case
                # uses a zero stride and allocates no repeated index table.
                if len(counts) == 1:
                    self.slots = rows.expand(counts[0], -1)
                else:
                    self.slots = rows.repeat_interleave(torch.tensor(counts, device="cpu"), dim=0)
                self.lengths = torch.cat([
                    torch.arange(l - n + 1, l + 1, dtype=torch.int64, device="cpu") for n, l in zip(counts, lengths)
                ])
            mode = "decode" if decode else "replay"
        if self.metrics_path:
            self.metrics = dict(mode=mode, rank=self.rank, requests=len(counts), queries=sum(counts),
                                optimized_cpu_attention=self.causal_cpu_attention and not self.direct_prefill,
                                start_ts=time.time(), max_history=max(lengths), mirrored=mirrored is not None,
                                prefill_contiguous=int(self.prefill_first_slot is not None),
                                history_host_bytes=self.history.keys.nbytes + self.history.values.nbytes,
                                history_gpu_bytes=0, metadata_d2h_bytes=metadata_bytes,
                                qkv_d2h_bytes=0, out_h2d_bytes=0, cpu_attention_ms=0.0,
                                cpu_store_ms=0.0, download_event_wait_ms=0.0)
        if self.phase_reader is not None and not self.direct_prefill:
            self.layer_phase = self.phase_reader.start_forward()
            if self.layer_phase is not None:
                if self.metrics_path:
                    self.metrics.update(target_physical_round=self.layer_phase.round_id,
                                        cpu_attention_monotonic_us=[], gpu_admissions_monotonic_us=[])
                self._admit_gpu_burst()  # First Q/K/V, before the first attention call.

    def _admit_gpu_burst(self):
        admitted, deadline = self.layer_phase.admit_gpu()
        if self.metrics_path:
            # Host submission markers, not GPU kernel start/end timestamps.
            self.metrics["gpu_admissions_monotonic_us"].append((admitted, deadline))

    def _pinned(self, name, shape):
        # Cache a high-water buffer for each tensor. Only a single layer's current
        # Q/K/V/O is staged; all long-lived KV pages are regular CPU memory.
        size = shape.numel()
        buffer = self.buffers.get(name)
        if buffer is None or size > buffer.numel():
            buffer = torch.empty(size, dtype=torch.bfloat16, device="cpu", pin_memory=True)
            self.buffers[name] = buffer
        return buffer[:size].view(shape)

    def forward(self, q, k, v, layer, batch, save_kv_cache, *, last_kv_only=False):
        if (not save_kv_cache or k is None or v is None or layer.is_cross_attention
                or layer.logit_cap or layer.sliding_window_size not in (None, -1)):
            raise RuntimeError("CPU Draft attention expects causal self-attention with ordinary KV writes")
        index = layer.layer_id - self.first_layer
        if last_kv_only and (getattr(batch, "_cpu_draft_tail_writer", None) is not self
                             or self.direct_prefill or index != self.history.num_layers - 1):
            raise RuntimeError("KV-only execution requires the last layer of a materialize-only Draft decode")
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.reshape_as(k)
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise TypeError("CPU Draft attention requires BF16 Q/K/V")
        k_host = self._pinned("k", k.shape)
        v_host = self._pinned("v", v.shape)
        k_host.copy_(k.contiguous(), non_blocking=True)
        v_host.copy_(v.contiguous(), non_blocking=True)
        if not self.direct_prefill and not last_kv_only:
            q_host = self._pinned("q", q.shape)
            q_host.copy_(q.contiguous(), non_blocking=True)
        self.download_done.record()

        if self.direct_prefill:
            # K/V already cover the complete causal prefix and are available on
            # the GPU for this layer. Enqueue attention before the host scatter.
            out = self.flash_prefill(
                q=q, k=k, v=v, cu_seqlens_q=self.cu, cu_seqlens_k=self.cu,
                max_seqlen_q=self.max_new, max_seqlen_k=self.max_new,
                softmax_scale=layer.scaling, causal=True,
            )

        if self.metrics_path:
            started_wait = time.perf_counter()
        self.download_done.synchronize()
        if self.metrics_path:
            started_store = time.perf_counter()
            self.metrics["download_event_wait_ms"] += 1000 * (started_store - started_wait)
        self.history.store(index, self.locations, k_host, v_host, contiguous_first=self.prefill_first_slot)
        if self.metrics_path:
            self.metrics["cpu_store_ms"] += 1000 * (time.perf_counter() - started_store)
        if not self.direct_prefill and not last_kv_only:
            # A previous layer may still read the shared pinned output on a CUDA
            # stream. Wait for that copy before allowing CPU threads to overwrite it.
            if self.upload_in_flight:
                self.upload_done.synchronize()
            out_host = self._pinned("out", q.shape)
            if self.metrics_path:
                started_attention = time.perf_counter()
                if self.layer_phase is not None:
                    cpu_start_us = time.monotonic_ns() // 1000
            if self.causal_cpu_attention:
                self.history.decode_causal(
                    index, q_host, self.slots, self.lengths, self.query_offsets, layer.scaling, out=out_host,
                )
            else:
                self.history.decode(index, q_host, self.slots, self.lengths, layer.scaling, out=out_host)
            if self.metrics_path:
                self.metrics["cpu_attention_ms"] += 1000 * (time.perf_counter() - started_attention)
                if self.layer_phase is not None:
                    self.metrics["cpu_attention_monotonic_us"].append((cpu_start_us, time.monotonic_ns() // 1000))
            if self.layer_phase is not None:
                # Attention may run through an entire Target GPU compute phase.
                # The following burst includes the output upload, O, MLP, and
                # the next layer's Q/K/V (the final layer precedes logits).
                self._admit_gpu_burst()
            out = out_host.to(q.device, non_blocking=True)
            self.upload_done.record()
            self.upload_in_flight = True

        if self.metrics_path:
            self.metrics["qkv_d2h_bytes"] += k.nbytes + v.nbytes
            if not self.direct_prefill and not last_kv_only:
                self.metrics["qkv_d2h_bytes"] += q.nbytes
                self.metrics["out_h2d_bytes"] += out.nbytes
            if index + 1 == self.history.num_layers:
                self.metrics["end_ts"] = time.time()
                if last_kv_only:
                    self.metrics.update(tail_last_layer_kv_only=True, cpu_attention_layers=index)
                if self.layer_phase is not None:
                    phase = self.layer_phase
                    self.metrics.update(gpu_bursts=phase.bursts, window_admissions=phase.hits,
                                        window_fallbacks=phase.bursts - phase.hits,
                                        window_poll_us=phase.waited_us)
                    if getattr(phase, "verify_envelope", False):
                        self.metrics.update(gpu_dispatch_mode="verify_envelope",
                                            observed_window_bursts=phase.hits,
                                            window_admissions=0, window_fallbacks=0)
                path = self.metrics_path.format(rank=self.rank, pid=os.getpid())
                with open(path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(self.metrics) + "\n")
        if last_kv_only:
            batch._cpu_draft_tail_kv_complete = True
            return None
        return out.reshape(q.shape[0], -1)
