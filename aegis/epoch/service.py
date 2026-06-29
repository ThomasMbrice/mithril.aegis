"""
Fault Epoch Service — monotonic counter, the consistency backbone for URC.

Every classified fault increments the epoch.  Checkpoints, collective ops,
and compute fallbacks are all tagged with the epoch they belong to.
Recovery consensus only considers state from a consistent epoch boundary.

This is the cross-layer generalization of TierCheck's
"global minimum recoverable version" reduction (§3.2).
"""

from __future__ import annotations

import threading


class FaultEpochService:
    """
    Thread-safe, monotonically-increasing fault epoch counter.

    Synchronous API so it can be called from both sync and async callers
    without risk of blocking the event loop (the critical section is
    a single integer increment — nanoseconds, not coroutine-await-worthy).

    Design invariant: strictly monotonic, no duplicate epochs under race.
    """

    def __init__(self, initial: int = 0) -> None:
        self._epoch = initial
        self._lock = threading.Lock()

    def current(self) -> int:
        """Return the current epoch without incrementing."""
        with self._lock:
            return self._epoch

    def next_epoch(self) -> int:
        """Atomically increment and return the new epoch."""
        with self._lock:
            self._epoch += 1
            return self._epoch

    def tag(self) -> int:
        """Alias for current() — tag a checkpoint or event with the live epoch."""
        return self.current()
