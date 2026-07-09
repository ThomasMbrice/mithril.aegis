"""
Layer D — Storage survivability (B2–B4).

Real implementation of TierCheck (arXiv:2605.17821)'s checkpoint I/O:
real file writes/reads, real base+differential checkpointing (XOR-diff
against a stored base, zlib-compressed), real SHA-256 integrity
verification on restore, and real measured wall-clock write/restore time.

  D1  Tier-1  local volatile  → differential, in-place restore
  D2  Tier-2  peer volatile   → async replicate to peer for node-loss recovery
  D3  Tier-3  remote durable  → async migrate base checkpoints (S3/Lustre)
  D4  Decoupled persistence   → frequent compressed diffs + infrequent base
  D5  URC                     → decentralized recovery consensus (see consensus/urc.py)

What's real here: the file I/O, diff/base bookkeeping, checksum
verification, and timing. What's *not* real: tier2/tier3 are local
directories standing in for an actual peer node and an actual S3/Lustre
endpoint — there is no real network replication or object-store client in
this dev environment (see ``RemoteObjectStoreBackend``, a documented but
unimplemented extension point). This does not reproduce TierCheck's
<10s/40B-param claim (§5.1 UT-D) — that needs the real training job and
storage fabric — but the checkpoint mechanics exercised here are genuine.

Checkpoint metadata carries fidelity_flag (§3.4): if a checkpoint is
written while MeCeFO fallback is active, the flag must be set so that
approximated state is never silently treated as full-fidelity.

§3.2 URC gating: recover() accepts ``min_valid_epoch`` from the EPE (via
UnifiedRecoveryConsensus.agree()) and, when provided, restores the latest
checkpoint at or before that epoch rather than unconditionally the newest
write — so a restore never picks state newer than what surviving ranks
have collectively validated.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import RecoveryLayer, RecoveryResult
from ..telemetry.events import BlastRadius, TelemetryEvent

logger = logging.getLogger(__name__)

_STORAGE_TIERS = frozenset({BlastRadius.B2, BlastRadius.B3, BlastRadius.B4})

_DEFAULT_PAYLOAD_BYTES = 65536  # 64 KiB synthetic checkpoint shard
_DEFAULT_BASE_INTERVAL = 5      # every Nth write per node is a full base (D4)


@dataclass
class CheckpointMetadata:
    """
    Checkpoint metadata schema (§3.4).

    ``fidelity_flag=True`` means this checkpoint was taken while at least
    one rank was running under MeCeFO degraded-compute mode.
    ``diff=True`` means this is a differential checkpoint against
    ``base_index`` (an index into the same node's history list); False
    means this entry is itself a base.
    """

    epoch: int
    tier: str  # "tier1" | "tier2" | "tier3"
    node: str = ""
    rank: int = 0
    timestamp: float = field(default_factory=time.time)
    fidelity_flag: bool = False
    diff: bool = True
    size_bytes: int = 0
    checksum: str = ""
    path: str = ""
    orig_len: int = 0
    base_index: int | None = None


def _xor_diff(a: bytes, b: bytes) -> bytes:
    """Reversible byte-level diff: XOR ``a`` against ``b``, zero-padded to equal length."""
    n = max(len(a), len(b))
    aa = np.frombuffer(a.ljust(n, b"\0"), dtype=np.uint8)
    bb = np.frombuffer(b.ljust(n, b"\0"), dtype=np.uint8)
    return (aa ^ bb).tobytes()


class CheckpointBackend(abc.ABC):
    """Pluggable byte-store for one tier."""

    @abc.abstractmethod
    def write_bytes(self, relpath: str, data: bytes) -> str:
        """Persist ``data`` at ``relpath``; returns the key to store in metadata."""

    @abc.abstractmethod
    def read_bytes(self, relpath: str) -> bytes:
        """Read back bytes previously written at ``relpath``."""

    @abc.abstractmethod
    def exists(self, relpath: str) -> bool: ...


class LocalFilesystemBackend(CheckpointBackend):
    """
    Default backend for all three tiers.  Real file I/O on local disk.

    In production, Tier-2 and Tier-3 would be a peer node's storage and a
    remote object store respectively — here they're separate local
    directories standing in for those, which is honest for exercising the
    checkpoint *mechanics* but does not validate real network/object-store
    behavior (see module docstring).
    """

    def __init__(self, root: str | os.PathLike) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def write_bytes(self, relpath: str, data: bytes) -> str:
        p = self._root / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return relpath

    def read_bytes(self, relpath: str) -> bytes:
        return (self._root / relpath).read_bytes()

    def exists(self, relpath: str) -> bool:
        return (self._root / relpath).exists()


class RemoteObjectStoreBackend(CheckpointBackend):
    """
    Hardware/infra-pending extension point for the real Tier-3 remote
    durable store (§2.2: "pluggable backend; Lustre/GPFS/S3").  Not
    implemented — this dev environment has no object-store credentials or
    parallel-FS mount to validate against.  Swap in for
    ``LocalFilesystemBackend`` once the target cluster's storage endpoint
    is decided.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise NotImplementedError(
            "RemoteObjectStoreBackend is a pluggable extension point for "
            "S3/Lustre/GPFS and is not implemented in this dev environment. "
            "Use LocalFilesystemBackend, or implement this against the "
            "target cluster's real object store."
        )

    def write_bytes(self, relpath: str, data: bytes) -> str:  # pragma: no cover
        raise NotImplementedError

    def read_bytes(self, relpath: str) -> bytes:  # pragma: no cover
        raise NotImplementedError

    def exists(self, relpath: str) -> bool:  # pragma: no cover
        raise NotImplementedError


class StorageLayer(RecoveryLayer):
    """
    B2/B3/B4 recovery via tiered checkpointing.

    The three sub-tiers map directly to design §4 Layer D:
      B2 → Tier-1 (local volatile, differential, in-place restore)
      B3 → Tier-2 (peer volatile, neighbor replica)
      B4 → Tier-3 (remote durable, S3/Lustre base checkpoint)
    """

    def __init__(
        self,
        root_dir: str | os.PathLike | None = None,
        base_interval: int = _DEFAULT_BASE_INTERVAL,
        default_payload_bytes: int = _DEFAULT_PAYLOAD_BYTES,
        tier1_backend: CheckpointBackend | None = None,
        tier2_backend: CheckpointBackend | None = None,
        tier3_backend: CheckpointBackend | None = None,
    ) -> None:
        self._base_interval = base_interval
        self._default_payload_bytes = default_payload_bytes

        self._tmpdir: tempfile.TemporaryDirectory | None = None
        if root_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="aegis-tiercheck-")
            root = Path(self._tmpdir.name)
        else:
            root = Path(root_dir)

        self._backend1 = tier1_backend or LocalFilesystemBackend(root / "tier1")
        self._backend2 = tier2_backend or LocalFilesystemBackend(root / "tier2")
        self._backend3 = tier3_backend or LocalFilesystemBackend(root / "tier3")

        self._tier1: dict[str, list[CheckpointMetadata]] = {}
        self._tier2: dict[str, list[CheckpointMetadata]] = {}
        self._tier3: list[CheckpointMetadata] = []

        self._write_count1: dict[str, int] = {}
        self._write_count2: dict[str, int] = {}
        self._base_index1: dict[str, int] = {}
        self._base_index2: dict[str, int] = {}

    @property
    def handled_tiers(self) -> frozenset[BlastRadius]:
        return _STORAGE_TIERS

    async def can_handle(self, event: TelemetryEvent, tier: BlastRadius) -> bool:
        if tier == BlastRadius.B2:
            return bool(self._tier1.get(event.node))
        if tier == BlastRadius.B3:
            return bool(self._tier2.get(event.node))
        if tier == BlastRadius.B4:
            return bool(self._tier3)
        return False

    async def recover(
        self,
        event: TelemetryEvent,
        tier: BlastRadius,
        epoch: int,
        *,
        min_valid_epoch: int | None = None,
    ) -> RecoveryResult:
        if tier == BlastRadius.B2:
            return self._restore_tier1(event, min_valid_epoch)
        if tier == BlastRadius.B3:
            return self._restore_tier2(event, min_valid_epoch)
        if tier == BlastRadius.B4:
            return self._restore_tier3(min_valid_epoch)
        return RecoveryResult(success=False, message=f"Unhandled tier {tier!r}")

    # ------------------------------------------------------------------
    # Selection (§3.2 URC gating)

    @staticmethod
    def _select(
        history: list[CheckpointMetadata], min_valid_epoch: int | None
    ) -> tuple[CheckpointMetadata | None, int]:
        """
        Pick the checkpoint to restore.

        If ``min_valid_epoch`` is None (URC had no data to gate with), fall
        back to the latest write, unconstrained.  Otherwise pick the latest
        entry at or before ``min_valid_epoch`` — never a checkpoint newer
        than what surviving ranks have collectively validated.
        """
        if not history:
            return None, -1
        if min_valid_epoch is None:
            return history[-1], len(history) - 1
        candidates = [(i, c) for i, c in enumerate(history) if c.epoch <= min_valid_epoch]
        if not candidates:
            return None, -1
        idx, ckpt = max(candidates, key=lambda t: t[1].epoch)
        return ckpt, idx

    def _reconstruct(
        self, backend: CheckpointBackend, history: list[CheckpointMetadata], index: int
    ) -> tuple[bytes, bool]:
        """Read back and reconstruct a checkpoint's full payload; verify checksum."""
        ckpt = history[index]
        raw = backend.read_bytes(ckpt.path)
        if not ckpt.diff:
            payload = raw
        else:
            assert ckpt.base_index is not None
            base_entry = history[ckpt.base_index]
            base_raw = backend.read_bytes(base_entry.path)  # base entries store raw payload
            diff_bytes = zlib.decompress(raw)
            payload = _xor_diff(diff_bytes, base_raw)[: ckpt.orig_len]
        ok = hashlib.sha256(payload).hexdigest() == ckpt.checksum
        return payload, ok

    # ------------------------------------------------------------------
    # Per-tier restore

    def _restore_tier1(self, event: TelemetryEvent, min_valid_epoch: int | None) -> RecoveryResult:
        history = self._tier1.get(event.node, [])
        ckpt, idx = self._select(history, min_valid_epoch)
        if ckpt is None:
            return RecoveryResult(success=False, message=f"No Tier-1 checkpoint for {event.node}")

        start = time.perf_counter()
        payload, ok = self._reconstruct(self._backend1, history, idx)
        elapsed = time.perf_counter() - start

        if not ok:
            return RecoveryResult(
                success=False,
                message=f"Tier-1 checksum mismatch restoring {event.node} at epoch {ckpt.epoch}",
            )
        logger.info(
            "[D] Tier-1 restore: node=%s epoch=%d fidelity=%s bytes=%d %.2fms",
            event.node, ckpt.epoch, ckpt.fidelity_flag, len(payload), elapsed * 1e3,
        )
        return RecoveryResult(
            success=True,
            message=f"Tier-1 restored {event.node} from epoch {ckpt.epoch} ({len(payload)} bytes, checksum verified)",
        )

    def _restore_tier2(self, event: TelemetryEvent, min_valid_epoch: int | None) -> RecoveryResult:
        history = self._tier2.get(event.node, [])
        ckpt, idx = self._select(history, min_valid_epoch)
        if ckpt is None:
            return RecoveryResult(success=False, message=f"No Tier-2 checkpoint for {event.node}")

        start = time.perf_counter()
        payload, ok = self._reconstruct(self._backend2, history, idx)
        elapsed = time.perf_counter() - start

        if not ok:
            return RecoveryResult(
                success=False,
                message=f"Tier-2 checksum mismatch restoring {event.node} at epoch {ckpt.epoch}",
            )
        logger.info(
            "[D] Tier-2 restore: node=%s epoch=%d (peer replica) bytes=%d %.2fms",
            event.node, ckpt.epoch, len(payload), elapsed * 1e3,
        )
        return RecoveryResult(
            success=True,
            message=f"Tier-2 restored {event.node} from peer replica (epoch {ckpt.epoch}, checksum verified)",
        )

    def _restore_tier3(self, min_valid_epoch: int | None) -> RecoveryResult:
        ckpt, idx = self._select(self._tier3, min_valid_epoch)
        if ckpt is None:
            return RecoveryResult(success=False, message="No Tier-3 base checkpoint available")

        start = time.perf_counter()
        raw = self._backend3.read_bytes(ckpt.path)
        elapsed = time.perf_counter() - start
        ok = hashlib.sha256(raw).hexdigest() == ckpt.checksum

        if not ok:
            return RecoveryResult(
                success=False,
                message=f"Tier-3 checksum mismatch restoring base at epoch {ckpt.epoch}",
            )
        logger.info(
            "[D] Tier-3 restore from remote durable base (epoch=%d fidelity=%s) %.2fms",
            ckpt.epoch, ckpt.fidelity_flag, elapsed * 1e3,
        )
        return RecoveryResult(
            success=True,
            message=f"Tier-3 restored from remote base (epoch {ckpt.epoch}, checksum verified)",
        )

    # ------------------------------------------------------------------
    # Checkpoint writers (called by the checkpoint scheduler, not by the EPE)

    def _write_node_tier(
        self,
        backend: CheckpointBackend,
        history_map: dict[str, list[CheckpointMetadata]],
        write_count: dict[str, int],
        base_index: dict[str, int],
        tier_name: str,
        node: str,
        epoch: int,
        fidelity_flag: bool,
        payload: bytes,
    ) -> CheckpointMetadata:
        history = history_map.setdefault(node, [])
        count = write_count.get(node, 0)
        is_base = (count % self._base_interval) == 0 or node not in base_index

        checksum = hashlib.sha256(payload).hexdigest()

        if is_base:
            to_store = payload
            diff = False
            b_index = None
        else:
            base_entry = history[base_index[node]]
            base_raw = backend.read_bytes(base_entry.path)
            to_store = zlib.compress(_xor_diff(payload, base_raw))
            diff = True
            b_index = base_index[node]

        relpath = f"{node}/{epoch}-{count}.ckpt"
        path = backend.write_bytes(relpath, to_store)

        meta = CheckpointMetadata(
            epoch=epoch,
            tier=tier_name,
            node=node,
            fidelity_flag=fidelity_flag,
            diff=diff,
            size_bytes=len(to_store),
            checksum=checksum,
            path=path,
            orig_len=len(payload),
            base_index=b_index,
        )
        history.append(meta)
        new_index = len(history) - 1
        if is_base:
            base_index[node] = new_index
        write_count[node] = count + 1
        return meta

    def _default_payload(self) -> bytes:
        return os.urandom(self._default_payload_bytes)

    def write_tier1(
        self,
        node: str,
        epoch: int,
        *,
        fidelity_flag: bool = False,
        payload: bytes | None = None,
    ) -> CheckpointMetadata:
        return self._write_node_tier(
            self._backend1, self._tier1, self._write_count1, self._base_index1,
            "tier1", node, epoch, fidelity_flag, payload if payload is not None else self._default_payload(),
        )

    def write_tier2(
        self,
        node: str,
        epoch: int,
        *,
        fidelity_flag: bool = False,
        payload: bytes | None = None,
    ) -> CheckpointMetadata:
        return self._write_node_tier(
            self._backend2, self._tier2, self._write_count2, self._base_index2,
            "tier2", node, epoch, fidelity_flag, payload if payload is not None else self._default_payload(),
        )

    def write_tier3(
        self,
        epoch: int,
        *,
        fidelity_flag: bool = False,
        payload: bytes | None = None,
    ) -> CheckpointMetadata:
        payload = payload if payload is not None else self._default_payload()
        checksum = hashlib.sha256(payload).hexdigest()
        relpath = f"global/{epoch}-{len(self._tier3)}.ckpt"
        path = self._backend3.write_bytes(relpath, payload)
        meta = CheckpointMetadata(
            epoch=epoch,
            tier="tier3",
            fidelity_flag=fidelity_flag,
            diff=False,
            size_bytes=len(payload),
            checksum=checksum,
            path=path,
            orig_len=len(payload),
        )
        self._tier3.append(meta)
        return meta

    # ------------------------------------------------------------------
    # Convenience accessors (used by aegis/checkpoint.py; avoids leaking
    # the internal history-list representation to callers)

    def latest_tier1(self, node: str) -> CheckpointMetadata | None:
        history = self._tier1.get(node, [])
        return history[-1] if history else None

    def latest_tier2(self, node: str) -> CheckpointMetadata | None:
        history = self._tier2.get(node, [])
        return history[-1] if history else None

    def latest_tier3(self) -> CheckpointMetadata | None:
        return self._tier3[-1] if self._tier3 else None
