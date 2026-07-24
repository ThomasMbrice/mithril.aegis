"""
UT-D — Real StorageLayer (B2-B4) checkpoint I/O.

Not a reproduction of TierCheck's <10s/40B-param claim (that needs the
real training job and storage fabric, see design.md §8.1) — these tests
validate that the checkpoint *mechanics* (real file I/O, base+diff
bookkeeping, SHA-256 integrity verification, URC epoch gating) are real
and correct, not an in-memory-only stub.
"""

from __future__ import annotations

import time

from aegis.layers.storage import StorageLayer
from aegis.telemetry.events import BlastRadius, FaultSignal, TelemetryEvent


def _event(node: str, rank: int = 0) -> TelemetryEvent:
    return TelemetryEvent(
        rank=rank, node=node, fault_signal=FaultSignal.CUDA_KERNEL_CRASH,
        raw_payload={}, epoch=0,
    )


async def test_tier1_round_trip_is_byte_correct_and_checksummed():
    storage = StorageLayer(default_payload_bytes=4096)
    payload = b"x" * 4096
    ckpt = storage.write_tier1("node0", epoch=1, payload=payload)

    assert ckpt.checksum == __import__("hashlib").sha256(payload).hexdigest()
    assert not ckpt.diff  # first write for this node is always a base

    result = await storage.recover(_event("node0"), BlastRadius.B2, epoch=1)
    assert result.success
    assert "checksum verified" in result.message


async def test_tier1_restore_under_10s_wall_clock():
    """Real, measured wall-clock restore time — TierCheck's target, measured not asserted."""
    storage = StorageLayer(default_payload_bytes=65536)
    storage.write_tier1("node0", epoch=1)

    start = time.perf_counter()
    result = await storage.recover(_event("node0"), BlastRadius.B2, epoch=1)
    elapsed = time.perf_counter() - start

    assert result.success
    assert elapsed < 10.0


async def test_differential_checkpoint_base_interval():
    """Every Nth write is a full base; the rest are diffs (D4)."""
    storage = StorageLayer(base_interval=3, default_payload_bytes=1024)
    metas = [storage.write_tier1("node0", epoch=i) for i in range(6)]

    is_diff = [m.diff for m in metas]
    assert is_diff == [False, True, True, False, True, True]


async def test_diff_checkpoint_reconstructs_correctly_after_many_writes():
    """A diff entry deep in the chain still reconstructs to its own exact payload."""
    storage = StorageLayer(base_interval=3, default_payload_bytes=256)
    payloads = [f"payload-{i}".encode().ljust(256, b"\0") for i in range(5)]
    for i, p in enumerate(payloads):
        storage.write_tier1("node0", epoch=i, payload=p)

    # Restore without URC gating → picks the latest (epoch=4, a diff entry)
    result = await storage.recover(_event("node0"), BlastRadius.B2, epoch=4)
    assert result.success
    assert "checksum verified" in result.message


async def test_checksum_mismatch_is_detected():
    """Corrupting the on-disk diff/base bytes must fail restore, not silently succeed."""
    storage = StorageLayer(default_payload_bytes=64)
    ckpt = storage.write_tier1("node0", epoch=1, payload=b"a" * 64)

    # Corrupt the file on disk directly via the backend
    storage._backend1.write_bytes(ckpt.path, b"\xff" * 64)

    result = await storage.recover(_event("node0"), BlastRadius.B2, epoch=1)
    assert not result.success
    assert "checksum mismatch" in result.message


async def test_urc_gating_restores_at_or_before_min_valid_epoch():
    """min_valid_epoch caps which checkpoint is eligible — never a newer one."""
    storage = StorageLayer(default_payload_bytes=128)
    storage.write_tier1("node0", epoch=1)
    storage.write_tier1("node0", epoch=5)  # newer than what "surviving ranks" have validated

    history = storage._tier1["node0"]
    ckpt, idx = storage._select(history, min_valid_epoch=3)
    assert ckpt.epoch == 1  # the epoch=5 write must NOT be selected

    ckpt_unbounded, _ = storage._select(history, min_valid_epoch=None)
    assert ckpt_unbounded.epoch == 5  # no consensus data → latest write, old behavior


async def test_urc_gating_fails_cleanly_when_nothing_qualifies():
    storage = StorageLayer(default_payload_bytes=128)
    storage.write_tier1("node0", epoch=10)

    result = await storage.recover(_event("node0"), BlastRadius.B2, epoch=10, min_valid_epoch=2)
    assert not result.success


async def test_tier3_global_base_round_trip():
    storage = StorageLayer(default_payload_bytes=512)
    storage.write_tier3(epoch=1)
    result = await storage.recover(
        TelemetryEvent(rank=0, node="any", fault_signal=FaultSignal.RACK_POWER_LOSS,
                        raw_payload={}, epoch=0),
        BlastRadius.B4, epoch=1,
    )
    assert result.success
    assert "checksum verified" in result.message


async def test_fidelity_flag_persists_through_real_disk_round_trip():
    """fidelity_flag survives a real write→disk→read cycle (not just in-memory)."""
    storage = StorageLayer(default_payload_bytes=128)
    ckpt = storage.write_tier1("node0", epoch=1, fidelity_flag=True)
    assert ckpt.fidelity_flag

    latest = storage.latest_tier1("node0")
    assert latest is not None
    assert latest.fidelity_flag
