"""
Layer D — Storage survivability (B2–B4).

Stub for TierCheck (arXiv:2605.17821).  In production this manages:
  D1  Tier-1  local volatile  → differential, in-place restore (<10 s)
  D2  Tier-2  peer volatile   → async replicate to peer for node-loss recovery
  D3  Tier-3  remote durable  → async migrate base checkpoints (S3/Lustre)
  D4  Decoupled persistence   → frequent compressed diffs + infrequent base
  D5  URC                     → decentralized recovery consensus (see consensus/urc.py)

Checkpoint metadata carries fidelity_flag (§3.4): if a checkpoint is written
while MeCeFO fallback is active, the flag must be set so that approximated
state is never silently treated as full-fidelity.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .base import RecoveryLayer, RecoveryResult
from ..telemetry.events import BlastRadius, TelemetryEvent

logger = logging.getLogger(__name__)

_STORAGE_TIERS = frozenset({BlastRadius.B2, BlastRadius.B3, BlastRadius.B4})


@dataclass
class CheckpointMetadata:
    """
    Checkpoint metadata schema (§3.4).

    ``fidelity_flag=True`` means this checkpoint was taken while at least
    one rank was running under MeCeFO degraded-compute mode.
    ``diff=True`` means this is a differential checkpoint; False means a base.
    """

    epoch: int
    tier: str  # "tier1" | "tier2" | "tier3"
    node: str = ""
    rank: int = 0
    timestamp: float = field(default_factory=time.time)
    fidelity_flag: bool = False
    diff: bool = True
    size_bytes: int = 0  # populated by real implementation


class StorageLayer(RecoveryLayer):
    """
    B2/B3/B4 recovery via tiered checkpointing.

    The three sub-tiers map directly to design §4 Layer D:
      B2 → Tier-1 (local volatile, sub-10s restore)
      B3 → Tier-2 (peer volatile, neighbor replica)
      B4 → Tier-3 (remote durable, S3/Lustre base checkpoint)
    """

    def __init__(self) -> None:
        self._tier1: dict[str, CheckpointMetadata] = {}   # node → latest
        self._tier2: dict[str, CheckpointMetadata] = {}   # node → latest
        self._tier3: CheckpointMetadata | None = None      # global base

    @property
    def handled_tiers(self) -> frozenset[BlastRadius]:
        return _STORAGE_TIERS

    async def can_handle(self, event: TelemetryEvent, tier: BlastRadius) -> bool:
        if tier == BlastRadius.B2:
            return event.node in self._tier1
        if tier == BlastRadius.B3:
            return event.node in self._tier2
        if tier == BlastRadius.B4:
            return self._tier3 is not None
        return False

    async def recover(
        self, event: TelemetryEvent, tier: BlastRadius, epoch: int
    ) -> RecoveryResult:
        if tier == BlastRadius.B2:
            return await self._restore_tier1(event, epoch)
        if tier == BlastRadius.B3:
            return await self._restore_tier2(event, epoch)
        if tier == BlastRadius.B4:
            return await self._restore_tier3(event, epoch)
        return RecoveryResult(success=False, message=f"Unhandled tier {tier!r}")

    # ------------------------------------------------------------------
    # Per-tier restore stubs

    async def _restore_tier1(self, event: TelemetryEvent, epoch: int) -> RecoveryResult:
        ckpt = self._tier1.get(event.node)
        if ckpt is None:
            return RecoveryResult(success=False, message=f"No Tier-1 checkpoint for {event.node}")
        logger.info(
            "[D] Tier-1 restore: node=%s epoch=%d fidelity=%s",
            event.node, ckpt.epoch, ckpt.fidelity_flag,
        )
        await asyncio.sleep(0)  # real: restore from local volatile memory (<10 s)
        return RecoveryResult(
            success=True,
            message=f"Tier-1 restored {event.node} from epoch {ckpt.epoch}",
        )

    async def _restore_tier2(self, event: TelemetryEvent, epoch: int) -> RecoveryResult:
        ckpt = self._tier2.get(event.node)
        if ckpt is None:
            return RecoveryResult(success=False, message=f"No Tier-2 checkpoint for {event.node}")
        logger.info(
            "[D] Tier-2 restore: node=%s epoch=%d (peer replica)", event.node, ckpt.epoch
        )
        await asyncio.sleep(0)  # real: restore from peer volatile store
        return RecoveryResult(
            success=True,
            message=f"Tier-2 restored {event.node} from peer replica (epoch {ckpt.epoch})",
        )

    async def _restore_tier3(self, event: TelemetryEvent, epoch: int) -> RecoveryResult:
        if self._tier3 is None:
            return RecoveryResult(success=False, message="No Tier-3 base checkpoint available")
        logger.info(
            "[D] Tier-3 restore from remote durable base (epoch=%d fidelity=%s)",
            self._tier3.epoch, self._tier3.fidelity_flag,
        )
        await asyncio.sleep(0)  # real: restore from remote durable store (S3/Lustre)
        return RecoveryResult(
            success=True,
            message=f"Tier-3 restored from remote base (epoch {self._tier3.epoch})",
        )

    # ------------------------------------------------------------------
    # Checkpoint writers (called by the checkpoint scheduler, not by the EPE)

    def write_tier1(
        self, node: str, epoch: int, *, fidelity_flag: bool = False
    ) -> CheckpointMetadata:
        ckpt = CheckpointMetadata(epoch=epoch, tier="tier1", node=node,
                                  fidelity_flag=fidelity_flag, diff=True)
        self._tier1[node] = ckpt
        return ckpt

    def write_tier2(
        self, node: str, epoch: int, *, fidelity_flag: bool = False
    ) -> CheckpointMetadata:
        ckpt = CheckpointMetadata(epoch=epoch, tier="tier2", node=node,
                                  fidelity_flag=fidelity_flag, diff=True)
        self._tier2[node] = ckpt
        return ckpt

    def write_tier3(
        self, epoch: int, *, fidelity_flag: bool = False
    ) -> CheckpointMetadata:
        ckpt = CheckpointMetadata(epoch=epoch, tier="tier3",
                                  fidelity_flag=fidelity_flag, diff=False)
        self._tier3 = ckpt
        return ckpt
