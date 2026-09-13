"""A bounded control-only executor for the Target's Python submission interval.

The callback must be nonblocking and must never issue TP collectives or receive
complete Draft results. stop() is a barrier: the scheduler may consume results,
release requests, or begin another round only after it returns successfully.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class BackgroundGrantPump:
    def __init__(
        self,
        step: Callable[[], bool | None],
        *,
        initialize: Callable[[], None] | None = None,
        finalize: Callable[[], None] | None = None,
        interval_s: float = 0.0005,
        name: str = "specstream_grant_pump",
    ):
        self._step = step
        self._initialize = initialize
        self._finalize = finalize
        self._interval_s = max(float(interval_s), 0.0001)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._error: BaseException | None = None
        self.iterations = 0
        self.started_ns = 0
        self.stopped_ns = 0

    def start(self) -> "BackgroundGrantPump":
        self._thread.start()
        return self

    def _run(self) -> None:
        self.started_ns = time.perf_counter_ns()
        try:
            if self._initialize is not None:
                self._initialize()
            while not self._stop.is_set():
                keep_running = self._step()
                self.iterations += 1
                if keep_running is False:
                    break
                self._stop.wait(self._interval_s)
        except BaseException as exc:
            self._error = exc
        finally:
            if self._finalize is not None:
                try:
                    self._finalize()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
            self.stopped_ns = time.perf_counter_ns()

    def stop(self, *, raise_errors: bool = True) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # Fail closed. Proceeding would permit old-round control to
                # operate on recycled state. Callbacks are required nonblocking.
                raise RuntimeError(
                    "SpecStream grant pump failed to stop within 5 seconds"
                )
        if raise_errors and self._error is not None:
            raise RuntimeError(
                "SpecStream background grant pump failed"
            ) from self._error

    @property
    def elapsed_ms(self) -> float:
        if not self.started_ns or not self.stopped_ns:
            return 0.0
        return (self.stopped_ns - self.started_ns) / 1e6
