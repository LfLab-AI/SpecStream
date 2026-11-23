"""GPU burst admission for same-group speculative Draft forwards.

Target and Draft run on one host with the same TP size. Use a fresh local
directory per service pair. No model state, request IDs or collectives cross
this channel. Strict admission uses the Draft TP CPU group for a common
continue/cancel decision before the next GPU burst.
"""

import contextlib
import contextvars
import os
import struct
import time
from pathlib import Path


_round = contextvars.ContextVar("draft_tail_physical_round", default=None)
_strict_vote = contextvars.ContextVar("draft_tail_strict_vote", default=None)
_verify_envelope = contextvars.ContextVar("draft_tail_verify_envelope", default=False)
_sample = struct.Struct("=qqqq")  # TP size, physical round, observed_us, deadline_us
_max_age_us = 2000


@contextlib.contextmanager
def draft_tail_forward(round_id, *, strict_vote=None, verify_envelope=False):
    if verify_envelope and strict_vote is not None:
        raise ValueError("strict layer admission and the verification-envelope diagnostic are mutually exclusive")
    token = _round.set(int(round_id))
    vote_token = _strict_vote.set(strict_vote)
    envelope_token = _verify_envelope.set(verify_envelope)
    try:
        yield
    finally:
        _verify_envelope.reset(envelope_token)
        _strict_vote.reset(vote_token)
        _round.reset(token)


def in_draft_tail_forward():
    return _round.get() is not None


class DraftTailWindowExpired(RuntimeError):
    """Every Draft TP rank abandoned the same still-private forward."""

    def __init__(self, phase):
        super().__init__("no common Target transfer wait available for the next Draft GPU burst")
        self.burst = phase.bursts
        self.admissions = phase.hits
        self.waited_us = phase.waited_us


def make_strict_vote(rank, size, cpu_group, leader):
    if size == 1:
        return lambda poll: poll()
    import torch
    import torch.distributed as dist

    if cpu_group is None or dist.get_backend(cpu_group) != "gloo":
        raise ValueError("strict Draft tail phase requires a dedicated Gloo TP CPU group")
    decision = torch.zeros(2, dtype=torch.int64, device="cpu")

    def vote(poll):
        # CPU attention finishes independently on each Draft TP rank. All
        # ranks must reach this boundary before reading a short-lived window.
        dist.barrier(group=cpu_group)
        if rank == 0:
            try:
                observed, deadline = poll()
            except OSError:
                # Mailbox disappearance is recoverable: cancel on all ranks.
                observed, deadline = _now_us(), 0
            decision[0], decision[1] = observed, deadline
        dist.broadcast(decision, src=leader, group=cpu_group)
        return tuple(decision.tolist())

    return vote


def _now_us():
    return time.monotonic_ns() // 1000


def _rank_file(directory, rank):
    return directory / f"draft-layer-target-{rank}.window"


class WindowPublisher:
    def __init__(self, directory, rank, size):
        import fcntl

        self.fcntl = fcntl
        self.size = size
        directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(_rank_file(directory, rank), os.O_RDWR | os.O_CREAT, 0o600)
        self.clear()

    def _write(self, round_id, deadline, blocking):
        f = self.fcntl
        try:
            f.flock(self.fd, f.LOCK_EX if blocking else f.LOCK_EX | f.LOCK_NB)
        except BlockingIOError:
            # The reader holds a lock only around pread; never stall a sample.
            return
        try:
            os.pwrite(self.fd, _sample.pack(self.size, round_id, _now_us(), deadline), 0)
        finally:
            f.flock(self.fd, f.LOCK_UN)

    def publish(self, observation):
        deadline = int(observation.window_end_us) if observation.active else 0
        self._write(int(observation.round_id), deadline, blocking=False)

    def clear(self):
        # End of the Target forward (pump joined). Invalidate even if a reader
        # was in its short read section at the last sample.
        self._write(-1, 0, blocking=True)

    def close(self):
        self.clear()
        os.close(self.fd)


def make_window_publisher(tp_rank, tp_size):
    directory = os.getenv("SPECSTREAM_DRAFT_LAYER_WINDOW_DIR")
    return WindowPublisher(Path(directory), tp_rank, tp_size) if directory else None


class WindowReader:
    def __init__(self, directory, size, burst_us, wait_us):
        import fcntl

        if size < 1 or burst_us < 1 or wait_us < 0:
            raise ValueError("Invalid TP size or Draft layer admission budget")
        self.fcntl = fcntl
        self.directory = directory
        self.size = size
        self.burst_us = burst_us
        self.wait_us = wait_us
        self.peers = {}

    def start_forward(self):
        era = _round.get()
        return None if era is None else ForwardPhase(self, era)

    def _rank_snapshots(self, round_id, *, require_window=False):
        f = self.fcntl
        samples = []
        for rank in range(self.size):
            try:
                if rank not in self.peers:
                    self.peers[rank] = os.open(_rank_file(self.directory, rank), os.O_RDONLY)
                fd = self.peers[rank]
                f.flock(fd, f.LOCK_SH | f.LOCK_NB)
                try:
                    data = os.pread(fd, _sample.size, 0)
                finally:
                    f.flock(fd, f.LOCK_UN)
            except (FileNotFoundError, BlockingIOError):
                return ()
            if len(data) != _sample.size:
                return ()
            size, era, observed, end = _sample.unpack(data)
            if size != self.size or era != round_id or (require_window and end <= observed):
                return ()
            samples.append((observed, end))
        return samples

    def verification_active(self, round_id):
        # The Target pump publishes on every sample, including when no DMA
        # wait is exposed, and clears its era at verify exit. A fresh matching
        # heartbeat says nothing about spare GPU capacity or an H2D deadline.
        samples = self._rank_snapshots(round_id)
        now = _now_us()
        return bool(samples) and all(0 <= now - seen <= _max_age_us for seen, _ in samples)

    def window_deadline(self, round_id):
        samples = self._rank_snapshots(round_id, require_window=True)
        if not samples:
            return 0
        oldest = min(seen for seen, _ in samples)
        deadline = min(end for _, end in samples)
        now = _now_us()
        if now - oldest > _max_age_us or deadline - now < self.burst_us:
            return 0
        return deadline

    def close(self):
        for fd in self.peers.values():
            os.close(fd)
        self.peers.clear()


def make_window_reader(tp_size):
    directory = os.getenv("SPECSTREAM_DRAFT_LAYER_WINDOW_DIR")
    if not directory:
        return None
    burst = int(os.getenv("SPECSTREAM_DRAFT_LAYER_BURST_US", "100"))
    wait = int(os.getenv("SPECSTREAM_DRAFT_LAYER_WAIT_US", "1000"))
    return WindowReader(Path(directory), tp_size, burst, wait)


class ForwardPhase:
    def __init__(self, reader, round_id):
        self.reader = reader
        self.round_id = round_id
        self.remaining_us = reader.wait_us
        self.waited_us = 0
        self.hits = 0
        self.bursts = 0
        self.vote = _strict_vote.get()
        self.verify_envelope = _verify_envelope.get()

    def admit_gpu(self):
        started = _now_us()
        if self.verify_envelope:
            # Passive sample for tracing only. Submit every burst immediately,
            # also if the Target has already left this verification round.
            deadline = self.reader.window_deadline(self.round_id)
            now = _now_us()
            hit = deadline - now >= self.reader.burst_us
            self.bursts += 1
            self.hits += int(hit)
            return now, deadline if hit else 0
        poll = lambda: self._poll(started + self.remaining_us)
        # In strict mode the Draft ranks rendezvous on their CPU group. The
        # coordinator samples *all* Target rank windows and sends one result
        # before any Draft rank takes a conditional model/collective path.
        observed, deadline = poll() if self.vote is None else self.vote(poll)
        elapsed = _now_us() - started
        self.remaining_us = max(0, self.remaining_us - elapsed)
        self.waited_us += elapsed
        self.bursts += 1
        self.hits += int(deadline != 0)
        if self.vote is not None and deadline == 0:
            raise DraftTailWindowExpired(self)
        return observed, deadline

    def _poll(self, stop):
        while True:
            deadline = self.reader.window_deadline(self.round_id)
            now = _now_us()
            hit = deadline - now >= self.reader.burst_us
            if hit or now >= stop:
                return now, deadline if hit else 0
            time.sleep(min(stop - now, 50) / 1e6)
