"""
Layer C — Compute survivability (B1).

Stub for MeCeFO (arXiv:2510.16415).  In production this implements:
  (i)  Skip-connection on MHA backprop (dead node's attention path skipped)
  (ii) Selective activation recompute in FFN (neighbor recomputes dead node's FFN)
  (iii) Low-rank gradient approximation (reduces memory pressure on absorbing node)

Reported numbers: 4.18% throughput drop under high-frequency failures;
baseline convergence rate preserved; works with TP/PP (node-local design).

§3.4 fidelity: iterations computed under fallback are approximate.
RecoveryResult.degraded=True flags this; callers must propagate to checkpoint
metadata fidelity_flag so there's no silent approximation.

§6 risk: bound consecutive fallbacks — force a full-fidelity checkpoint after
MAX_CONSECUTIVE_FALLBACKS to avoid compounding approximation error.
"""

from __future__ import annotations

import asyncio
import logging

from .base import RecoveryLayer, RecoveryResult
from ..telemetry.events import BlastRadius, TelemetryEvent

logger = logging.getLogger(__name__)

_B1_TIERS = frozenset({BlastRadius.B1})

# After this many back-to-back fallback windows on a single node,
# emit a warning recommending a full-fidelity checkpoint (§6).
MAX_CONSECUTIVE_FALLBACKS = 10


class ComputeLayer(RecoveryLayer):
    """
    B1 recovery via MeCeFO neighbor absorption.

    ~4% throughput drop (paper: 4.18%).  Model state is degraded during
    fallback; callers receive degraded=True and must set fidelity_flag on
    any checkpoint written during this window.
    """

    def __init__(self) -> None:
        # node → neighbor node (registered by the runtime)
        self._topology: dict[str, str] = {}
        # count of consecutive fallback activations per node
        self._consecutive_fallbacks: dict[str, int] = {}

    @property
    def handled_tiers(self) -> frozenset[BlastRadius]:
        return _B1_TIERS

    async def can_handle(self, event: TelemetryEvent, tier: BlastRadius) -> bool:
        if tier != BlastRadius.B1:
            return False
        return event.node in self._topology

    async def recover(
        self, event: TelemetryEvent, tier: BlastRadius, epoch: int
    ) -> RecoveryResult:
        """
        Stub: trigger neighbor-absorb for dead node.

        Real: (i) skip-connection on MHA backprop, (ii) selective activation
        recompute in FFN, (iii) low-rank gradient approx (MeCeFO §3).
        """
        neighbor = self._topology.get(event.node)
        if not neighbor:
            return RecoveryResult(
                success=False,
                message=f"No neighbor registered for node {event.node}",
            )

        count = self._consecutive_fallbacks.get(event.node, 0) + 1
        self._consecutive_fallbacks[event.node] = count

        logger.info(
            "[B1] Compute recovery: node %s absorbed by %s (epoch %d, fallback #%d)",
            event.node,
            neighbor,
            epoch,
            count,
        )

        if count >= MAX_CONSECUTIVE_FALLBACKS:
            logger.warning(
                "[B1] Node %s has %d consecutive fallbacks — "
                "full-fidelity checkpoint strongly recommended (§6)",
                event.node,
                count,
            )

        await asyncio.sleep(0)  # real: redistribute workload to neighbor GPU

        msg = f"Neighbor {neighbor} absorbing {event.node} (fallback #{count})"
        return RecoveryResult(success=True, message=msg, degraded=True)

    def register_neighbor(self, node: str, neighbor: str) -> None:
        """Register that *neighbor* can absorb *node*'s workload."""
        self._topology[node] = neighbor

    def exit_fallback(self, node: str) -> None:
        """
        Signal that a node has recovered from fallback.

        Resets the consecutive-fallback counter so the degradation bound
        restarts from zero.
        """
        self._consecutive_fallbacks.pop(node, None)
