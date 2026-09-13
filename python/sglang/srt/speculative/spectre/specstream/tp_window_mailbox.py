"""Same-host, nonblocking H2D window exchange; never issues TP collectives."""

import os
import struct
import time
from dataclasses import replace
from pathlib import Path


class TPWindowMailbox:
    _record = struct.Struct("=qqqqq")  # round, window, end_us, observed_us, active

    def __init__(self, directory, rank, size):
        import fcntl

        self._fcntl = fcntl
        self.rank, self.size = rank, size
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.directory / f"rank{rank}.window", os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(self.fd, self._record.size)
        self.peers = {}
        self._key = None
        self._generation = 0
        self.clear()

    def _write(self, values):
        f = self._fcntl
        # Reader never holds a lock while doing CUDA work or waiting for peers.
        f.flock(self.fd, f.LOCK_EX)
        try:
            os.pwrite(self.fd, self._record.pack(*values), 0)
        finally:
            f.flock(self.fd, f.LOCK_UN)

    def publish(self, observation):
        self._write((observation.round_id, observation.window_id,
                     observation.window_end_us, time.monotonic_ns() // 1000,
                     int(observation.active)))

    def clear(self):
        self._write((-1, 0, 0, 0, 0))

    def intersect(self, observation):
        inactive = replace(observation, active=False, remaining_us=0.0,
                           window_end_us=0, reason="tp_window_not_ready")
        if not observation.active:
            return inactive
        end = observation.window_end_us
        key = [observation.window_id]
        now = time.monotonic_ns() // 1000
        for rank in range(self.size):
            if rank == self.rank:
                continue
            try:
                if rank not in self.peers:
                    self.peers[rank] = os.open(self.directory / f"rank{rank}.window", os.O_RDONLY)
                fd = self.peers[rank]
                self._fcntl.flock(fd, self._fcntl.LOCK_SH | self._fcntl.LOCK_NB)
                try:
                    data = os.pread(fd, self._record.size, 0)
                finally:
                    self._fcntl.flock(fd, self._fcntl.LOCK_UN)
            except (FileNotFoundError, BlockingIOError):
                return inactive
            if len(data) != self._record.size:
                return inactive
            round_id, window, deadline, observed, active = self._record.unpack(data)
            if not active or round_id != observation.round_id or now - observed > 2000:
                return inactive
            end = min(end, deadline)
            key.append(window)
        if end <= now:
            return inactive
        key = (observation.round_id, tuple(key))
        if key != self._key:
            self._key = key
            self._generation += 1
        return replace(observation, window_id=self._generation,
                       window_end_us=end, remaining_us=float(end - now),
                       reason="tp_common_h2d_window")

    def close(self):
        self.clear()
        os.close(self.fd)
        for fd in self.peers.values():
            os.close(fd)
        self.peers.clear()
