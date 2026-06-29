"""
UT-A3 — Fault Epoch Service.

Pass criteria (§5.1):
  - Strictly monotonic under sequential calls
  - No duplicate epochs under concurrent access (race condition safety)
  - current() never increments; next_epoch() always does
"""

from __future__ import annotations

import threading

from aegis.epoch.service import FaultEpochService


def test_initial_value() -> None:
    svc = FaultEpochService(initial=0)
    assert svc.current() == 0


def test_first_next_epoch_is_one() -> None:
    svc = FaultEpochService(initial=0)
    assert svc.next_epoch() == 1


def test_monotonic_sequential() -> None:
    svc = FaultEpochService()
    epochs = [svc.next_epoch() for _ in range(200)]
    for i in range(1, len(epochs)):
        assert epochs[i] > epochs[i - 1], (
            f"Non-monotonic at index {i}: {epochs[i-1]} → {epochs[i]}"
        )


def test_no_duplicates_sequential() -> None:
    svc = FaultEpochService()
    epochs = [svc.next_epoch() for _ in range(1000)]
    assert len(set(epochs)) == 1000, "Duplicate epoch values in sequential calls"


def test_no_duplicates_concurrent() -> None:
    """UT-A3: no duplicate epochs under race conditions across 20 threads."""
    svc = FaultEpochService()
    results: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(100):
            epoch = svc.next_epoch()
            with lock:
                results.append(epoch)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 2000
    assert len(set(results)) == 2000, (
        f"Duplicate epochs found: {len(results) - len(set(results))} duplicates"
    )


def test_current_does_not_increment() -> None:
    svc = FaultEpochService()
    svc.next_epoch()  # → 1
    assert svc.current() == 1
    assert svc.current() == 1  # still 1
    svc.next_epoch()            # → 2
    assert svc.current() == 2


def test_tag_alias() -> None:
    svc = FaultEpochService()
    svc.next_epoch()  # → 1
    assert svc.tag() == svc.current()


def test_custom_initial_value() -> None:
    svc = FaultEpochService(initial=42)
    assert svc.current() == 42
    assert svc.next_epoch() == 43
