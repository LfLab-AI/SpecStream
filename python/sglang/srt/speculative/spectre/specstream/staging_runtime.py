from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import wraps
from itertools import islice
import logging
import math
import os
from threading import RLock
import time

import torch


logger = logging.getLogger(__name__)


def _detect_pcie_bandwidth_ceiling_gbps(
    device: torch.device,
) -> tuple[float, str]:
    """Return a physical PCIe raw-bandwidth upper bound via NVML.

    The raw transfer rate intentionally ignores encoding/protocol overhead and
    adds 10% margin, so it is an upper bound rather than a throughput estimate.
    A smaller measured-copy value must never be treated as a physical ceiling.
    """

    try:
        import pynvml
    except ImportError:
        return 0.0, "nvml_unavailable"

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        properties = torch.cuda.get_device_properties(device)
        uuid = getattr(properties, "uuid", None)
        if uuid:
            if isinstance(uuid, bytes):
                uuid = uuid.decode("ascii")
            handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
        else:
            local_index = (
                int(device.index)
                if device.index is not None
                else int(torch.cuda.current_device())
            )
            visible = [
                item.strip()
                for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if item.strip()
            ]
            token = visible[local_index] if local_index < len(visible) else ""
            if token.startswith("GPU-"):
                handle = pynvml.nvmlDeviceGetHandleByUUID(token)
            elif token.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(token))
            elif not visible:
                handle = pynvml.nvmlDeviceGetHandleByIndex(local_index)
            else:
                return 0.0, "cuda_device_nvml_mapping_unavailable"
        generation = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
        width = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
        # Raw GT/s converted to decimal GB/s per lane.  Ignoring link encoding
        # makes these values safely larger than usable payload bandwidth.
        raw_gbps_per_lane = {
            1: 0.3125,
            2: 0.625,
            3: 1.0,
            4: 2.0,
            5: 4.0,
            6: 8.0,
            7: 16.0,
        }.get(generation)
        if raw_gbps_per_lane is None or width <= 0:
            return 0.0, f"unsupported_pcie_link_gen{generation}_x{width}"
        ceiling = raw_gbps_per_lane * width * 1.10
        return ceiling, f"nvml_pcie_gen{generation}_x{width}_raw_upper_bound"
    except Exception as exc:
        return 0.0, f"nvml_pcie_query_failed:{type(exc).__name__}"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


@dataclass(frozen=True)
class StagingTransfer:
    slot: int
    tensor: torch.Tensor
    nbytes: int
    submitted_ns: int
    source_count: int = 1
    valid_tokens: torch.Tensor | None = None
    valid_lengths: tuple[int, ...] = ()
    h2d_window_id: int = 0
    source_nbytes: int = 0
    padding_nbytes: int = 0
    dma_count: int = 0
    host_wait_ms: float = 0.0
    host_pack_ms: float = 0.0
    metadata_cache_hit: bool = False


@dataclass
class _ImmutableMetadata:
    host: torch.Tensor
    device: torch.Tensor
    ready_event: object | None = None


def _locked_windows(method):
    """Serialize descriptor publication/recycling, never a host CUDA wait."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._h2d_window_lock:
            return method(self, *args, **kwargs)

    return wrapped


def _coalesce_adjacent_sources(sources):
    """Merge adjacent views without copying or reading unrelated storage."""
    merged = []
    for source in sources:
        if not source.numel():
            continue
        if merged:
            previous = merged[-1]
            if (
                previous.dtype == source.dtype
                and previous.device == source.device
                and tuple(previous.shape[1:]) == tuple(source.shape[1:])
                and previous.is_contiguous()
                and source.is_contiguous()
                and previous.untyped_storage().data_ptr()
                == source.untyped_storage().data_ptr()
                and previous.storage_offset() + previous.numel()
                == source.storage_offset()
            ):
                merged[-1] = previous.as_strided(
                    (previous.shape[0] + source.shape[0], *previous.shape[1:]),
                    previous.stride(),
                )
                continue
        merged.append(source)
    return tuple(merged)


@dataclass(frozen=True)
class H2DWindowObservation:
    """Nonblocking view of the H2D copy currently executing on the GPU.

    ``remaining_us`` is deliberately a conservative lower-bound estimate.  Its
    age upper bound starts at the host's CUDA-event submission timestamp and
    tightens whenever polling observes an incomplete begin event.
    """

    round_id: int
    window_id: int = 0
    active: bool = False
    remaining_us: float = 0.0
    # Absolute lower-bound completion time in the same CLOCK_MONOTONIC domain
    # used by TargetGrantRuntime deadlines.  Consumers must preserve this value
    # instead of re-anchoring ``remaining_us`` after host-side processing.
    window_end_us: int = 0
    nbytes: int = 0
    timing_source: str = ""
    reason: str = ""


@dataclass(frozen=True)
class H2DTimingSample:
    round_id: int
    window_id: int
    nbytes: int
    elapsed_ms: float
    target_wait_ms: float | None = None


@dataclass
class _TrackedH2DWindow:
    round_id: int
    window_id: int
    nbytes: int
    begin_event: object
    end_event: object
    enqueued_ns: int
    last_not_started_ns: int | None = None
    target_wait_begin_event: object | None = None
    target_wait_end_event: object | None = None
    target_wait_gate_recorded: bool = False
    consumer_pending: bool = False


class StagingWindowPool:
    """Bounded copy/compute double-buffer state machine."""

    def __init__(
        self,
        num_buffers: int,
        device: torch.device | str,
        *,
        track_h2d_windows: bool = False,
        serialize_h2d: bool = False,
        metadata_cache_entries: int = 128,
    ) -> None:
        if num_buffers < 1:
            raise ValueError("num_buffers must be positive")
        self.num_buffers = int(num_buffers)
        self.device = torch.device(device)
        self.serialize_h2d = bool(serialize_h2d)
        if metadata_cache_entries < 1:
            raise ValueError("metadata_cache_entries must be positive")
        self._metadata_cache_entries = int(metadata_cache_entries)
        self._metadata_cache: OrderedDict[tuple[int, ...], _ImmutableMetadata] = (
            OrderedDict()
        )
        self._h2d_window_lock = RLock()
        self.copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        self._buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._views: list[torch.Tensor | None] = [None] * self.num_buffers
        self._host_buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._host_views: list[torch.Tensor | None] = [None] * self.num_buffers
        self._valid_buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._valid_host_buffers: list[torch.Tensor | None] = [None] * self.num_buffers
        self._ready_events = [
            torch.cuda.Event() if self.copy_stream is not None else None
            for _ in range(self.num_buffers)
        ]
        self._has_ready_event = [False] * self.num_buffers
        self._free_events = [
            torch.cuda.Event() if self.copy_stream is not None else None
            for _ in range(self.num_buffers)
        ]
        self._has_free_event = [False] * self.num_buffers
        # The same CUDA events serve two purposes: PCIe-slack admission can
        # inspect the current copy nonblockingly, and ordinary streaming runs
        # can report isolated memcpy time plus the exact Target-stream wait.
        self._track_h2d_windows = bool(
            track_h2d_windows and self.copy_stream is not None
        )
        self._h2d_windows: deque[_TrackedH2DWindow] = deque()
        self._h2d_windows_by_id: dict[int, _TrackedH2DWindow] = {}
        self._h2d_timing_samples: deque[H2DTimingSample] = deque()
        # Seed a small event pool outside the measured path.  If a Target
        # forward queues more than eight transfers before they can be
        # harvested, the pool grows instead of overwriting an unfinished
        # window.  Complete coverage is required for honest overlap metrics.
        self._max_tracked_h2d_windows = 8
        self._h2d_event_pool: list[tuple[object, object, object, object]] = (
            [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(self._max_tracked_h2d_windows)
            ]
            if self._track_h2d_windows
            else []
        )
        self._next_h2d_window_id = 1
        (
            self._h2d_bandwidth_ceiling_gbps,
            self._h2d_bandwidth_ceiling_source,
        ) = (
            _detect_pcie_bandwidth_ceiling_gbps(self.device)
            if self._track_h2d_windows
            else (0.0, "tracking_disabled")
        )
        self._h2d_tracking_error = ""
        if self._track_h2d_windows and self._h2d_bandwidth_ceiling_gbps <= 0.0:
            logger.warning(
                "SpecStream current-H2D grants disabled: no reliable physical "
                "PCIe bandwidth ceiling (%s)",
                self._h2d_bandwidth_ceiling_source,
            )

    def _serialize_after_target_compute(self) -> None:
        """Order this copy after all Target work queued before submission.

        K1/K2 use this dependency as a strict no-overlap control.  The H2D
        still runs on the measured copy stream, but it cannot borrow compute
        time from the retained GPU-History prefix or any earlier Target kernel.
        K3+ omit the dependency and therefore retain true copy/compute overlap.
        """

        if self.copy_stream is None or not self.serialize_h2d:
            return
        target_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(target_stream)

    def _disable_h2d_tracking(self, reason: str) -> None:
        # Fail closed: an event-query failure must remove overlap opportunity,
        # not silently fall back to an unbounded historical window.
        self._track_h2d_windows = False
        self._h2d_tracking_error = str(reason)

    @_locked_windows
    def _begin_h2d_window(
        self, *, round_id: int | None, nbytes: int, asynchronous: bool
    ) -> _TrackedH2DWindow | None:
        if (
            not self._track_h2d_windows
            or round_id is None
            or not asynchronous
            or self.copy_stream is None
        ):
            return None
        try:
            # Never query or synchronize CUDA events on the per-layer submit
            # path.  Reuse a harvested tuple when possible; otherwise grow the
            # pool so an unfinished timing interval is never overwritten.
            if not self._h2d_event_pool:
                events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            else:
                events = self._h2d_event_pool.pop()
            (
                begin_event,
                end_event,
                target_wait_begin_event,
                target_wait_end_event,
            ) = events
            window = _TrackedH2DWindow(
                round_id=int(round_id),
                window_id=self._next_h2d_window_id,
                nbytes=max(int(nbytes), 0),
                begin_event=begin_event,
                end_event=end_event,
                enqueued_ns=time.perf_counter_ns(),
                target_wait_begin_event=target_wait_begin_event,
                target_wait_end_event=target_wait_end_event,
                consumer_pending=True,
            )
            self._next_h2d_window_id += 1
            # This is recorded after any slot-free dependency and immediately
            # before the first memcpy in the same copy stream.
            begin_event.record(self.copy_stream)
            return window
        except RuntimeError as exc:
            self._disable_h2d_tracking(f"cuda_event_record_failed:{exc}")
            return None

    @_locked_windows
    def _end_h2d_window(self, window: _TrackedH2DWindow | None) -> None:
        if window is None or self.copy_stream is None:
            return
        try:
            # Every memcpy belonging to this staging transfer is already
            # queued above this marker on the dedicated copy stream.
            window.end_event.record(self.copy_stream)
            # A recycled end event can still report its previous completed
            # generation until record(). Publish only the complete pair.
            self._h2d_windows.append(window)
            self._h2d_windows_by_id[window.window_id] = window
        except RuntimeError as exc:
            self._disable_h2d_tracking(f"cuda_event_record_failed:{exc}")

    @_locked_windows
    def _harvest_finished_h2d_windows(self) -> None:
        """Collect completed timings without synchronizing any CUDA stream."""

        while self._h2d_windows:
            window = self._h2d_windows[0]
            if window.consumer_pending:
                return
            try:
                if not bool(window.end_event.query()):
                    return
                elapsed_ms = float(window.begin_event.elapsed_time(window.end_event))
                target_wait_ms = None
                if window.target_wait_gate_recorded:
                    if not bool(window.target_wait_end_event.query()):
                        # The memcpy may be complete while the Target stream is
                        # still queued behind its ready-event dependency.  Keep
                        # this window until the synchronized forward epilogue
                        # can read both timestamps.
                        return
                    target_wait_ms = float(
                        window.target_wait_begin_event.elapsed_time(
                            window.target_wait_end_event
                        )
                    )
            except RuntimeError as exc:
                self._disable_h2d_tracking(f"cuda_event_query_failed:{exc}")
                return
            self._h2d_windows.popleft()
            self._h2d_windows_by_id.pop(window.window_id, None)
            if elapsed_ms > 0.0 and math.isfinite(elapsed_ms):
                self._h2d_timing_samples.append(
                    H2DTimingSample(
                        round_id=window.round_id,
                        window_id=window.window_id,
                        nbytes=window.nbytes,
                        elapsed_ms=elapsed_ms,
                        target_wait_ms=target_wait_ms,
                    )
                )
                measured_gbps = window.nbytes / elapsed_ms / 1e6
                if measured_gbps > 0.0 and math.isfinite(measured_gbps):
                    # Never lower the NVML physical ceiling.  A measurement
                    # above it (clock/timing noise or an unusual path) raises
                    # the ceiling further and therefore shortens, rather than
                    # inflates, the schedulable duration floor.
                    if self._h2d_bandwidth_ceiling_gbps > 0.0:
                        self._h2d_bandwidth_ceiling_gbps = max(
                            self._h2d_bandwidth_ceiling_gbps,
                            measured_gbps * 1.25,
                        )
            self._h2d_event_pool.append(
                (
                    window.begin_event,
                    window.end_event,
                    window.target_wait_begin_event,
                    window.target_wait_end_event,
                )
            )

    @_locked_windows
    def observe_h2d_window(self, round_id: int) -> H2DWindowObservation:
        """Query the current round's real CUDA copy window without blocking.

        The copy-stream queue is ordered, so only its oldest unfinished event
        pair can be active.  A window is exposed slack only when the Target
        compute stream has completed the pre-wait marker but not the post-wait
        marker around ``wait_event(ready)``.  Polling a pending copy-begin event
        also establishes an upper bound on copy age without GPU synchronization.
        """

        round_id = int(round_id)
        if not self._track_h2d_windows:
            reason = self._h2d_tracking_error or "cuda_event_tracking_disabled"
            return H2DWindowObservation(round_id=round_id, reason=reason)
        self._harvest_finished_h2d_windows()
        if not self._track_h2d_windows:
            return H2DWindowObservation(
                round_id=round_id,
                reason=self._h2d_tracking_error or "cuda_event_query_failed",
            )
        if not self._h2d_windows:
            return H2DWindowObservation(
                round_id=round_id, reason="no_current_h2d_window"
            )

        # Age accounting stays in perf_counter's elapsed-time domain.  Sample
        # the serialized-deadline clock *first*, then project the conservative
        # remaining duration onto it.  Python does not promise that perf_counter
        # and monotonic share an epoch, even when an implementation happens to
        # use the same underlying clock.
        deadline_now_us = time.monotonic_ns() // 1000
        now_ns = time.perf_counter_ns()
        try:
            # Prime timing anchors for future rolling-ring entries while an
            # earlier copy is active.  At most eight nonblocking queries are
            # issued, only in the explicit PCIe-slack mode.  Without this,
            # later copies would first be noticed after they started and their
            # old host-enqueue timestamp could conservatively collapse every
            # remaining budget to zero.
            for pending_window in islice(self._h2d_windows, 8):
                if not bool(pending_window.begin_event.query()):
                    pending_window.last_not_started_ns = now_ns
            # A later begin may have completed during the loop, which also
            # implies completion of earlier same-stream transfers.
            self._harvest_finished_h2d_windows()
            if not self._track_h2d_windows:
                return H2DWindowObservation(
                    round_id=round_id,
                    reason=self._h2d_tracking_error or "cuda_event_query_failed",
                )
            if not self._h2d_windows:
                return H2DWindowObservation(
                    round_id=round_id, reason="no_current_h2d_window"
                )
            window = self._h2d_windows[0]
            started = bool(window.begin_event.query())
            if not started:
                window.last_not_started_ns = now_ns
                return H2DWindowObservation(
                    round_id=round_id,
                    window_id=window.window_id,
                    nbytes=window.nbytes,
                    timing_source="current_cuda_event",
                    reason="current_h2d_pending",
                )
            # Close the small query race where the copy completes between the
            # first end-event query in harvest and the begin-event query above.
            if bool(window.end_event.query()):
                self._harvest_finished_h2d_windows()
                if self._h2d_windows and self._h2d_windows[0] is window:
                    return H2DWindowObservation(
                        round_id=round_id,
                        window_id=window.window_id,
                        nbytes=window.nbytes,
                        timing_source="current_cuda_event+target_wait_gate",
                        reason="target_h2d_wait_completing",
                    )
                return self.observe_h2d_window(round_id)
        except RuntimeError as exc:
            self._disable_h2d_tracking(f"cuda_event_query_failed:{exc}")
            return H2DWindowObservation(
                round_id=round_id, reason=self._h2d_tracking_error
            )

        if window.round_id != round_id:
            return H2DWindowObservation(
                round_id=round_id,
                window_id=window.window_id,
                nbytes=window.nbytes,
                timing_source="current_cuda_event",
                reason="different_round_h2d_active",
            )
        if (
            not window.target_wait_gate_recorded
            or window.target_wait_begin_event is None
            or window.target_wait_end_event is None
        ):
            return H2DWindowObservation(
                round_id=round_id,
                window_id=window.window_id,
                nbytes=window.nbytes,
                timing_source="current_cuda_event",
                reason="target_wait_gate_unavailable",
            )
        try:
            target_reached_wait = bool(window.target_wait_begin_event.query())
            target_left_wait = bool(window.target_wait_end_event.query())
            # Recheck the copy end after the Target-stream queries.  A grant is
            # valid only at the instant both predicates hold: Target is inside
            # wait_event(ready), and this exact transfer is still in progress.
            copy_finished = bool(window.end_event.query())
        except RuntimeError as exc:
            self._disable_h2d_tracking(f"cuda_event_query_failed:{exc}")
            return H2DWindowObservation(
                round_id=round_id, reason=self._h2d_tracking_error
            )
        if copy_finished:
            self._harvest_finished_h2d_windows()
            if self._h2d_windows and self._h2d_windows[0] is window:
                return H2DWindowObservation(
                    round_id=round_id,
                    window_id=window.window_id,
                    nbytes=window.nbytes,
                    timing_source="current_cuda_event+target_wait_gate",
                    reason="target_h2d_wait_completing",
                )
            return self.observe_h2d_window(round_id)
        if not target_reached_wait:
            return H2DWindowObservation(
                round_id=round_id,
                window_id=window.window_id,
                nbytes=window.nbytes,
                timing_source="current_cuda_event+target_wait_gate",
                reason="target_compute_before_h2d_wait",
            )
        if target_left_wait:
            return H2DWindowObservation(
                round_id=round_id,
                window_id=window.window_id,
                nbytes=window.nbytes,
                timing_source="current_cuda_event+target_wait_gate",
                reason="target_h2d_wait_completed",
            )
        if self._h2d_bandwidth_ceiling_gbps <= 0.0:
            return H2DWindowObservation(
                round_id=round_id,
                window_id=window.window_id,
                active=True,
                nbytes=window.nbytes,
                timing_source=(
                    "current_cuda_event+target_wait_gate+"
                    f"{self._h2d_bandwidth_ceiling_source}"
                ),
                reason="h2d_physical_ceiling_unavailable",
            )

        duration_floor_us = window.nbytes / self._h2d_bandwidth_ceiling_gbps / 1000.0
        # Event-record submission precedes actual GPU execution, hence it is a
        # safe (possibly loose) upper bound on the copy age.  A recent observed
        # incomplete begin event is a tighter bound when available.
        age_anchor_ns = max(
            window.enqueued_ns,
            window.last_not_started_ns or window.enqueued_ns,
        )
        age_upper_us = max((now_ns - age_anchor_ns) / 1000.0, 0.0)
        remaining_us = float(max(int(duration_floor_us - age_upper_us), 0))
        # Floor to a whole microsecond and anchor to the monotonic sample taken
        # before any event-query work.  Observation, per-request iteration and
        # grant serialization therefore can only consume this deadline; they
        # can never move it later.
        window_end_us = deadline_now_us + int(remaining_us)
        return H2DWindowObservation(
            round_id=round_id,
            window_id=window.window_id,
            active=True,
            remaining_us=remaining_us,
            window_end_us=window_end_us,
            nbytes=window.nbytes,
            timing_source=(
                "current_cuda_event+target_wait_gate+physical_copy_floor:"
                f"{self._h2d_bandwidth_ceiling_source}"
            ),
            reason=(
                "current_exposed_h2d_stall"
                if remaining_us > 0.0
                else "current_h2d_budget_exhausted"
            ),
        )

    @_locked_windows
    def drain_h2d_timing_samples(self) -> list[H2DTimingSample]:
        self._harvest_finished_h2d_windows()
        samples = list(self._h2d_timing_samples)
        self._h2d_timing_samples.clear()
        return samples

    @property
    def h2d_bandwidth_ceiling_gbps(self) -> float:
        return self._h2d_bandwidth_ceiling_gbps

    @property
    def h2d_bandwidth_ceiling_source(self) -> str:
        return self._h2d_bandwidth_ceiling_source

    def _wait_host_slot_writable(self, slot: int) -> float:
        """Do not overwrite a pinned pack buffer while its H2D is in flight."""

        if self.copy_stream is not None and self._has_ready_event[slot]:
            # The host pack buffer becomes writable as soon as H2D has consumed
            # it.  Waiting for the later compute/free event unnecessarily
            # serialized CPU packing with attention execution.
            if not self._ready_events[slot].query():
                started = time.perf_counter_ns()
                self._ready_events[slot].synchronize()
                return (time.perf_counter_ns() - started) / 1e6
        return 0.0

    def _immutable_valid_lengths(self, lengths: tuple[int, ...]):
        """Cache immutable metadata across layers, independently of KV slots.

        A cache hit writes neither host nor device memory. At bounded cache
        overflow, only the evicted metadata's first upload may require a wait;
        Target consumers retain device storage through record_stream().
        """
        entry = self._metadata_cache.get(lengths)
        if entry is not None:
            self._metadata_cache.move_to_end(lengths)
            return entry, True, 0.0
        wait_ms = 0.0
        if len(self._metadata_cache) >= self._metadata_cache_entries:
            _, old = self._metadata_cache.popitem(last=False)
            if old.ready_event is not None and not old.ready_event.query():
                started = time.perf_counter_ns()
                old.ready_event.synchronize()
                wait_ms = (time.perf_counter_ns() - started) / 1e6
        try:
            host = torch.tensor(
                lengths,
                dtype=torch.int32,
                device="cpu",
                pin_memory=self.copy_stream is not None,
            )
        except RuntimeError:
            host = torch.tensor(lengths, dtype=torch.int32, device="cpu")
        if self.copy_stream is None:
            device = host.clone()
        else:
            with torch.cuda.stream(self.copy_stream):
                device = torch.empty(
                    len(lengths), dtype=torch.int32, device=self.device
                )
        entry = _ImmutableMetadata(host, device)
        self._metadata_cache[lengths] = entry
        return entry, False, wait_ms

    def _ensure_host_shape(
        self,
        slot: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not 0 <= slot < self.num_buffers:
            raise IndexError("invalid SpecStream staging slot")
        current = self._host_buffers[slot]
        required = 1
        for size in shape:
            required *= int(size)
        if current is None or current.dtype != dtype or current.numel() < required:
            try:
                current = torch.empty(
                    required,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=bool(torch.cuda.is_available()),
                )
            except RuntimeError:
                current = torch.empty(required, dtype=dtype, device="cpu")
            self._host_buffers[slot] = current
        view = current[:required].view(shape)
        self._host_views[slot] = view
        return view

    def _ensure_valid_shape(
        self, slot: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        host = self._valid_host_buffers[slot]
        device = self._valid_buffers[slot]
        if host is None or host.numel() < count:
            try:
                host = torch.empty(
                    count,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=bool(torch.cuda.is_available()),
                )
            except RuntimeError:
                host = torch.empty(count, dtype=torch.int32, device="cpu")
            self._valid_host_buffers[slot] = host
        if device is None or device.numel() < count:
            device = torch.empty(count, dtype=torch.int32, device=self.device)
            self._valid_buffers[slot] = device
        return host[:count], device[:count]

    def _ensure_shape(
        self,
        slot: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not 0 <= slot < self.num_buffers:
            raise IndexError("invalid SpecStream staging slot")
        current = self._buffers[slot]
        required = 1
        for size in shape:
            required *= int(size)
        if current is None or current.dtype != dtype or current.numel() < required:
            if self.copy_stream is not None:
                torch.cuda.current_stream(self.device).synchronize()
            current = torch.empty(required, dtype=dtype, device=self.device)
            self._buffers[slot] = current
        view = current[:required].view(shape)
        self._views[slot] = view
        return view

    def reserve(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Allocate every bounded slot before the first verification round."""

        for slot in range(self.num_buffers):
            self._ensure_shape(slot, shape, dtype)

    def reserve_cohort_pack(self, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Preallocate reusable pinned pack and valid-length metadata buffers."""

        if len(shape) < 2 or int(shape[0]) < 1:
            raise ValueError("cohort reserve shape must begin with batch size")
        for slot in range(self.num_buffers):
            self._ensure_host_shape(slot, shape, dtype)
            self._ensure_valid_shape(slot, int(shape[0]))

    def calibrate_h2d_gbps(
        self, *, sample_bytes: int = 64 * 1024 * 1024, repeats: int = 4
    ) -> float:
        """Measure this process' pinned H2D bandwidth before serving starts.

        Python submit-to-consume intervals include queue residence and async
        kernel enqueue time; treating them as memcpy duration corrupts the I/O
        cost model.  This bounded startup calibration uses CUDA events on the
        actual SpecStream copy stream and never runs on the request path.
        """

        if self.copy_stream is None or repeats < 1:
            return 0.0
        destination = next(
            (buffer for buffer in self._buffers if buffer is not None), None
        )
        if destination is None or destination.numel() == 0:
            return 0.0
        element_size = int(destination.element_size())
        elements = min(
            int(destination.numel()),
            max(1, int(sample_bytes) // max(element_size, 1)),
        )
        try:
            source = torch.empty(
                elements,
                dtype=destination.dtype,
                device="cpu",
                pin_memory=True,
            )
        except RuntimeError:
            return 0.0
        target = destination.view(-1)[:elements]
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.copy_stream):
            target.copy_(source, non_blocking=True)
            # Materialize the fixed telemetry pool during startup, before the
            # measured interval.  No cudaEventCreate is then charged to the
            # first long-context request.
            for (
                begin_event,
                end_event,
                target_wait_begin_event,
                target_wait_end_event,
            ) in self._h2d_event_pool:
                begin_event.record(self.copy_stream)
                end_event.record(self.copy_stream)
                # These two events will later be recorded on the Target compute
                # stream around wait_event(ready).  Recording them once here
                # materializes their CUDA handles outside the request path.
                target_wait_begin_event.record(self.copy_stream)
                target_wait_end_event.record(self.copy_stream)
            started.record(self.copy_stream)
            for _ in range(int(repeats)):
                target.copy_(source, non_blocking=True)
            finished.record(self.copy_stream)
        finished.synchronize()
        elapsed_ms = float(started.elapsed_time(finished)) / int(repeats)
        if elapsed_ms <= 0 or not math.isfinite(elapsed_ms):
            return 0.0
        measured_gbps = elements * element_size / elapsed_ms / 1e6
        # See observe_h2d_window(): scheduling uses a duration floor derived
        # from a deliberately optimistic bandwidth ceiling.
        if self._h2d_bandwidth_ceiling_gbps > 0.0:
            self._h2d_bandwidth_ceiling_gbps = max(
                self._h2d_bandwidth_ceiling_gbps,
                measured_gbps * 1.25,
            )
        return measured_gbps

    def submit(
        self, source: torch.Tensor, slot: int, *, round_id: int | None = None
    ) -> StagingTransfer:
        if source.device.type != "cpu":
            raise ValueError("SpecStream H2D source must be a CPU tensor")
        if not source.is_contiguous():
            raise ValueError("SpecStream H2D source must be contiguous")
        destination = self._ensure_shape(slot, tuple(source.shape), source.dtype)
        submitted_ns = time.perf_counter_ns()
        if self.copy_stream is None:
            destination.copy_(source)
            h2d_window = None
        else:
            self._serialize_after_target_compute()
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                h2d_window = self._begin_h2d_window(
                    round_id=round_id,
                    nbytes=source.nbytes,
                    asynchronous=bool(source.is_pinned()),
                )
                destination.copy_(source, non_blocking=bool(source.is_pinned()))
                self._end_h2d_window(h2d_window)
                self._ready_events[slot].record(self.copy_stream)
                self._has_ready_event[slot] = True
        return StagingTransfer(
            slot,
            destination,
            source.nbytes,
            submitted_ns,
            h2d_window_id=(h2d_window.window_id if h2d_window is not None else 0),
            source_nbytes=int(source.nbytes),
            dma_count=1,
        )

    def submit_many(
        self,
        sources: list[torch.Tensor] | tuple[torch.Tensor, ...],
        slot: int,
        *,
        round_id: int | None = None,
    ) -> StagingTransfer:
        """Enqueue several adjacent CPU chunks into one contiguous GPU window.

        The sources remain separately allocated pinned slabs.  Their copies are
        issued back-to-back on the copy stream, followed by a *single* ready
        event.  This removes per-chunk event, attention-launch and Python-loop
        overhead without introducing an extra CPU concatenation.
        """

        if not sources:
            raise ValueError("SpecStream submit_many requires at least one source")
        first = sources[0]
        if first.device.type != "cpu" or not first.is_contiguous():
            raise ValueError("SpecStream H2D sources must be contiguous CPU tensors")
        trailing_shape = tuple(first.shape[1:])
        total_tokens = 0
        total_bytes = 0
        for source in sources:
            if source.device.type != "cpu" or not source.is_contiguous():
                raise ValueError(
                    "SpecStream H2D sources must be contiguous CPU tensors"
                )
            if source.dtype != first.dtype or tuple(source.shape[1:]) != trailing_shape:
                raise ValueError("grouped SpecStream chunks must have one KV geometry")
            total_tokens += int(source.shape[0])
            total_bytes += int(source.nbytes)

        copy_regions = _coalesce_adjacent_sources(sources)

        destination = self._ensure_shape(
            slot, (total_tokens, *trailing_shape), first.dtype
        )
        submitted_ns = time.perf_counter_ns()

        def copy_sources() -> None:
            cursor = 0
            for source in copy_regions:
                length = int(source.shape[0])
                destination[cursor : cursor + length].copy_(
                    source,
                    non_blocking=bool(source.is_pinned()),
                )
                cursor += length

        if self.copy_stream is None:
            copy_sources()
            h2d_window = None
        else:
            self._serialize_after_target_compute()
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                h2d_window = self._begin_h2d_window(
                    round_id=round_id,
                    nbytes=total_bytes,
                    asynchronous=all(source.is_pinned() for source in sources),
                )
                copy_sources()
                self._end_h2d_window(h2d_window)
                self._ready_events[slot].record(self.copy_stream)
                self._has_ready_event[slot] = True
        return StagingTransfer(
            slot,
            destination,
            total_bytes,
            submitted_ns,
            source_count=len(sources),
            h2d_window_id=(h2d_window.window_id if h2d_window is not None else 0),
            source_nbytes=total_bytes,
            dma_count=len(copy_regions),
        )

    def submit_cohort_groups(
        self,
        source_groups: (
            list[list[torch.Tensor] | tuple[torch.Tensor, ...]]
            | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], ...]
        ),
        slot: int,
        *,
        round_id: int | None = None,
    ) -> StagingTransfer:
        """Pack one cohort chunk-group and issue one contiguous H2D copy.

        CPU History slabs are request-private and therefore not adjacent.  A
        reusable pinned pack buffer turns all slabs for ``B`` requests and
        several adjacent chunks into one ``[B,N,2,H,D]`` DMA.  Padding is not
        initialized: ``valid_tokens`` masks it in the cohort kernel.
        """

        if not source_groups or not any(source_groups):
            raise ValueError("SpecStream cohort requires at least one source")
        first = next(group[0] for group in source_groups if group)
        if first.device.type != "cpu" or not first.is_contiguous():
            raise ValueError("SpecStream cohort sources must be contiguous CPU tensors")
        trailing_shape = tuple(first.shape[1:])
        valid_lengths: list[int] = []
        source_count = 0
        for group in source_groups:
            length = 0
            for source in group:
                if source.device.type != "cpu" or not source.is_contiguous():
                    raise ValueError(
                        "SpecStream cohort sources must be contiguous CPU tensors"
                    )
                if (
                    source.dtype != first.dtype
                    or tuple(source.shape[1:]) != trailing_shape
                ):
                    raise ValueError("cohort chunks must have one KV geometry")
                length += int(source.shape[0])
                source_count += 1
            valid_lengths.append(length)

        host_wait_ms = self._wait_host_slot_writable(slot)
        host_pack_started_ns = time.perf_counter_ns()
        max_tokens = max(valid_lengths)
        host = self._ensure_host_shape(
            slot,
            (len(source_groups), max_tokens, *trailing_shape),
            first.dtype,
        )
        for item_index, group in enumerate(source_groups):
            cursor = 0
            for source in group:
                length = int(source.shape[0])
                host[item_index, cursor : cursor + length].copy_(source)
                cursor += length

        host_pack_ms = (time.perf_counter_ns() - host_pack_started_ns) / 1e6

        valid_host, valid_device = self._ensure_valid_shape(slot, len(source_groups))
        for index, length in enumerate(valid_lengths):
            valid_host[index] = length

        destination = self._ensure_shape(slot, tuple(host.shape), host.dtype)
        source_nbytes = sum(
            source.nbytes for group in source_groups for source in group
        )
        # Equal-length cohorts still use one large copy. Ragged cohorts copy
        # each valid row only, so masked padding never consumes PCIe bandwidth.
        full_rectangle = all(length == max_tokens for length in valid_lengths)

        def copy_packed() -> None:
            if full_rectangle:
                destination.copy_(host, non_blocking=bool(host.is_pinned()))
            else:
                for index, length in enumerate(valid_lengths):
                    if length:
                        destination[index, :length].copy_(
                            host[index, :length], non_blocking=bool(host.is_pinned())
                        )
            valid_device.copy_(valid_host, non_blocking=bool(valid_host.is_pinned()))

        submitted_ns = time.perf_counter_ns()
        if self.copy_stream is None:
            copy_packed()
            h2d_window = None
        else:
            self._serialize_after_target_compute()
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                h2d_window = self._begin_h2d_window(
                    round_id=round_id,
                    nbytes=source_nbytes + valid_device.nbytes,
                    asynchronous=bool(host.is_pinned() and valid_host.is_pinned()),
                )
                copy_packed()
                self._end_h2d_window(h2d_window)
                self._ready_events[slot].record(self.copy_stream)
                self._has_ready_event[slot] = True
        return StagingTransfer(
            slot=slot,
            tensor=destination,
            nbytes=source_nbytes + valid_device.nbytes,
            submitted_ns=submitted_ns,
            source_count=source_count,
            valid_tokens=valid_device,
            valid_lengths=tuple(valid_lengths),
            h2d_window_id=(h2d_window.window_id if h2d_window is not None else 0),
            source_nbytes=source_nbytes,
            dma_count=(
                1 if full_rectangle else sum(length > 0 for length in valid_lengths)
            )
            + 1,
            host_wait_ms=host_wait_ms,
            host_pack_ms=host_pack_ms,
        )

    def submit_cohort_groups_direct_async(
        self,
        source_groups: (
            list[list[torch.Tensor] | tuple[torch.Tensor, ...]]
            | tuple[list[torch.Tensor] | tuple[torch.Tensor, ...], ...]
        ),
        slot: int,
        *,
        round_id: int | None = None,
    ) -> StagingTransfer:
        """Stream a cohort without repacking CPU data or copying padding.

        ``submit_cohort_groups`` is the steady-state packed-DMA path.  Calling
        it at the end of layer L, however, performs a potentially hundreds-of-
        MiB host-to-host pack before Python can enqueue L's Tail/MLP.  At high
        concurrency that CPU work becomes a new critical path and defeats
        cross-layer overlap.

        This path writes the request-private pinned slabs
        directly into slices of one cohort staging tensor on the copy stream.
        It uses more DMA descriptors, but no bulk synchronous CPU copy.  The
        same ready/free event and the same batched cohort kernel consume the
        resulting window in layer L+1.
        """

        if not source_groups or not any(source_groups):
            raise ValueError("SpecStream cohort requires at least one source")
        first = next(group[0] for group in source_groups if group)
        if first.device.type != "cpu" or not first.is_contiguous():
            raise ValueError("SpecStream cohort sources must be contiguous CPU tensors")
        trailing_shape = tuple(first.shape[1:])
        valid_lengths: list[int] = []
        source_count = 0
        total_bytes = 0
        for group in source_groups:
            length = 0
            for source in group:
                if source.device.type != "cpu" or not source.is_contiguous():
                    raise ValueError(
                        "SpecStream cohort sources must be contiguous CPU tensors"
                    )
                if (
                    source.dtype != first.dtype
                    or tuple(source.shape[1:]) != trailing_shape
                ):
                    raise ValueError("cohort chunks must have one KV geometry")
                length += int(source.shape[0])
                total_bytes += int(source.nbytes)
                source_count += 1
            valid_lengths.append(length)

        max_tokens = max(valid_lengths)
        destination = self._ensure_shape(
            slot,
            (len(source_groups), max_tokens, *trailing_shape),
            first.dtype,
        )
        metadata, metadata_cache_hit, host_wait_ms = self._immutable_valid_lengths(
            tuple(valid_lengths)
        )
        valid_host, valid_device = metadata.host, metadata.device
        metadata_bytes = 0 if metadata_cache_hit else int(valid_device.nbytes)
        copy_groups = tuple(
            _coalesce_adjacent_sources(group) for group in source_groups
        )

        submitted_ns = time.perf_counter_ns()

        def copy_sources() -> None:
            for item_index, group in enumerate(copy_groups):
                cursor = 0
                for source in group:
                    length = int(source.shape[0])
                    destination[item_index, cursor : cursor + length].copy_(
                        source,
                        non_blocking=bool(source.is_pinned()),
                    )
                    cursor += length
            if not metadata_cache_hit:
                valid_device.copy_(
                    valid_host, non_blocking=bool(valid_host.is_pinned())
                )
                if self.copy_stream is not None:
                    metadata.ready_event = torch.cuda.Event()
                    metadata.ready_event.record(self.copy_stream)

        if self.copy_stream is None:
            copy_sources()
            h2d_window = None
        else:
            self._serialize_after_target_compute()
            with torch.cuda.stream(self.copy_stream):
                if self._has_free_event[slot]:
                    self.copy_stream.wait_event(self._free_events[slot])
                h2d_window = self._begin_h2d_window(
                    round_id=round_id,
                    nbytes=total_bytes + metadata_bytes,
                    asynchronous=bool(
                        (metadata_cache_hit or valid_host.is_pinned())
                        and all(
                            source.is_pinned()
                            for group in source_groups
                            for source in group
                        )
                    ),
                )
                copy_sources()
                self._end_h2d_window(h2d_window)
                self._ready_events[slot].record(self.copy_stream)
                self._has_ready_event[slot] = True

        return StagingTransfer(
            slot=slot,
            tensor=destination,
            nbytes=total_bytes + metadata_bytes,
            submitted_ns=submitted_ns,
            source_count=source_count,
            valid_tokens=valid_device,
            valid_lengths=tuple(valid_lengths),
            h2d_window_id=(h2d_window.window_id if h2d_window is not None else 0),
            source_nbytes=total_bytes,
            dma_count=sum(len(group) for group in copy_groups)
            + int(not metadata_cache_hit),
            host_wait_ms=host_wait_ms,
            metadata_cache_hit=metadata_cache_hit,
        )

    @_locked_windows
    def wait_ready(self, transfer: StagingTransfer) -> torch.Tensor:
        if self.copy_stream is not None:
            target_stream = torch.cuda.current_stream(self.device)
            tracked_window = self._h2d_windows_by_id.get(transfer.h2d_window_id)
            if (
                tracked_window is not None
                and tracked_window.target_wait_begin_event is not None
                and tracked_window.target_wait_end_event is not None
            ):
                # Hardware-visible exposed-stall gate.  A copy is a usable
                # bubble only while the Target compute stream has reached this
                # dependency but has not passed it.  Cross-layer prefetch that
                # overlaps attention/MLP therefore remains Target-exclusive.
                tracked_window.target_wait_begin_event.record(target_stream)
                target_stream.wait_event(self._ready_events[transfer.slot])
                tracked_window.target_wait_end_event.record(target_stream)
                tracked_window.target_wait_gate_recorded = True
                tracked_window.consumer_pending = False
            else:
                target_stream.wait_event(self._ready_events[transfer.slot])
                if tracked_window is not None:
                    tracked_window.consumer_pending = False
            # Metadata is immutable and may be evicted while this consumer is
            # queued. Associate its allocator lifetime with the consumer stream.
            valid_tokens = getattr(transfer, "valid_tokens", None)
            if valid_tokens is not None and valid_tokens.is_cuda:
                valid_tokens.record_stream(target_stream)
        return transfer.tensor

    @_locked_windows
    def discard(self, transfer: StagingTransfer) -> None:
        """Cancel an unconsumed prefetch without recycling unfinished events.

        Copies are ordered on the single copy stream. Future writes to this
        slot cannot overtake its canceled transfer, so no host wait is needed.
        """
        window = self._h2d_windows_by_id.get(transfer.h2d_window_id)
        if window is not None:
            window.consumer_pending = False

    def mark_consumed(self, transfer: StagingTransfer) -> None:
        if self.copy_stream is not None:
            self._free_events[transfer.slot].record(
                torch.cuda.current_stream(self.device)
            )
            self._has_free_event[transfer.slot] = True

    @property
    def allocated_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._buffers if tensor is not None)

    @property
    def allocated_host_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self._host_buffers if tensor is not None)
