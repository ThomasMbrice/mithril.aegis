"""
RecoveryLayer — abstract base for all blast-radius recovery handlers.

Each concrete layer owns one band of the blast-radius spectrum and must
implement two coroutines:

  can_handle(event, tier) → bool  — quick pre-check (topology, checkpoint presence, …)
  recover(event, tier, epoch, *, min_valid_epoch=None) → RecoveryResult

The EPE calls these in order, escalating to the next tier if can_handle
returns False or recover returns success=False.

``min_valid_epoch`` (§3.2 URC): the EPE computes this from
``UnifiedRecoveryConsensus.agree()`` before calling storage-tier recovery,
so a restore never picks a checkpoint newer than what surviving ranks have
collectively validated.  ``None`` means URC had no data to gate with (e.g.
no other ranks have reported yet) — layers must treat that as "no
constraint", not as a failure, so the absence of consensus data never
itself blocks recovery.  Transport/compute layers may ignore the parameter.
"""

from __future__ import annotations

import abc

from ..telemetry.events import BlastRadius, TelemetryEvent


class RecoveryResult:
    """
    Result returned by a layer after attempting recovery.

    ``degraded=True`` means the layer completed recovery but the model
    state is approximate (MeCeFO fallback active).  Downstream components
    must surface this via checkpoint fidelity_flag (§3.4).
    """

    __slots__ = ("success", "message", "degraded")

    def __init__(
        self,
        success: bool,
        message: str = "",
        degraded: bool = False,
    ) -> None:
        self.success = success
        self.message = message
        self.degraded = degraded

    def __repr__(self) -> str:
        return (
            f"RecoveryResult(success={self.success}, "
            f"degraded={self.degraded}, message={self.message!r})"
        )


class RecoveryLayer(abc.ABC):
    """Abstract recovery layer."""

    @property
    @abc.abstractmethod
    def handled_tiers(self) -> frozenset[BlastRadius]:
        """Which blast-radius tiers this layer can handle."""

    @abc.abstractmethod
    async def can_handle(self, event: TelemetryEvent, tier: BlastRadius) -> bool:
        """
        Pre-flight check.

        Should be fast (no I/O).  Returns True only if this layer has
        the resources to absorb the fault right now (backup NIC available,
        neighbor registered, checkpoint present, …).
        """

    @abc.abstractmethod
    async def recover(
        self,
        event: TelemetryEvent,
        tier: BlastRadius,
        epoch: int,
        *,
        min_valid_epoch: int | None = None,
    ) -> RecoveryResult:
        """
        Attempt recovery for the given event at the given tier.

        Must be idempotent: the EPE may call recover() more than once if
        a concurrent fault races with the first attempt.
        """
