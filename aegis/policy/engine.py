"""
Escalation Policy Engine (EPE) — the integration keystone (§4 Layer E2).

Consumes Failure Classifier output + operator policy → routes to the
appropriate recovery layer (B/C/D).  Enforces two system-wide invariants:

  1. ONE-DIRECTIONAL ESCALATION: a fault is only ever routed to a higher
     tier, never de-escalated mid-fault.  Verified by the escalation_audit()
     method and tested by IT-6.

  2. DEBOUNCE / CORRELATION WINDOW: a classified B1 followed by ≥N correlated
     node failures in the same rack within window W is re-classified upward to
     B4 *before* committing to the expensive neighbor-absorb path (§3.3).

The EPE is scaffolded early as a stub (per §4 build-order note) so the
interface contract is fixed while B/C/D layers are being built.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from ..classifier.classifier import FailureClassifier
from ..consensus.urc import UnifiedRecoveryConsensus
from ..epoch.service import FaultEpochService
from ..layers.base import RecoveryLayer, RecoveryResult
from ..policy.dsl import OperatorPolicy
from ..telemetry.bus import UnifiedTelemetryPlane
from ..telemetry.events import BlastRadius, TelemetryEvent

logger = logging.getLogger(__name__)


@dataclass
class FaultRecord:
    """Internal audit record for one processed fault."""

    event: TelemetryEvent
    original_tier: BlastRadius
    final_tier: BlastRadius
    epoch: int
    result: RecoveryResult | None = None
    escalated: bool = False
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def escalation_valid(self) -> bool:
        """True iff the one-directional escalation invariant holds for this record."""
        return self.final_tier >= self.original_tier


class EscalationPolicyEngine:
    """
    Route every fault from the UTP to the cheapest layer that can absorb it.

    Subscribes to the UTP in __init__ so that events start flowing as soon as
    the UTP's dispatch loop is started (via AegisRuntime.start()).
    """

    def __init__(
        self,
        utp: UnifiedTelemetryPlane,
        classifier: FailureClassifier,
        epoch_service: FaultEpochService,
        consensus: UnifiedRecoveryConsensus,
        policy: OperatorPolicy,
        layers: list[RecoveryLayer],
    ) -> None:
        self._utp = utp
        self._classifier = classifier
        self._epochs = epoch_service
        self._consensus = consensus
        self._policy = policy

        # Build tier → layer map.  Layers declare which tiers they handle.
        self._tier_to_layer: dict[BlastRadius, RecoveryLayer] = {}
        for layer in layers:
            for tier in layer.handled_tiers:
                self._tier_to_layer[tier] = layer

        # Correlation window state (§3.3): recent B1s by (timestamp, node, rack)
        self._recent_b1s: deque[tuple[float, str, str]] = deque()

        # Immutable audit log
        self._history: list[FaultRecord] = []

        # Subscribe — EPE is the sole UTP consumer; FC/layers are called by EPE
        self._utp.subscribe(self._on_event)

    # ------------------------------------------------------------------
    # UTP callback

    async def _on_event(self, event: TelemetryEvent) -> None:
        """
        Main dispatch entry point, called by the UTP for every event.

        Steps:
          1. Rule-based classification → BlastRadius tier
          2. Correlation window check: maybe escalate B1 → B4 (§3.3)
          3. Increment fault epoch
          4. Route to layer(s), escalating if needed
          5. Append to audit log
        """
        tier = self._classifier.classify(event)

        if tier == BlastRadius.B0 and self._policy.allow_b0_fast_path:
            logger.debug(
                "[EPE] B0 fast-path: %s on %s — transport may have already begun migration",
                event.fault_signal.value, event.node,
            )

        # Correlation window re-classification (§3.3)
        tier = self._apply_correlation_window(event, tier)

        epoch = self._epochs.next_epoch()

        record = FaultRecord(
            event=event,
            original_tier=tier,
            final_tier=tier,
            epoch=epoch,
        )

        logger.info(
            "[EPE] %s → %s (epoch %d, node %s, rank %d)",
            event.fault_signal.value, tier.name, epoch, event.node, event.rank,
        )

        result = await self._route(record)
        record.result = result

        if not result.success:
            logger.error(
                "[EPE] All layers exhausted for %s at epoch %d: %s",
                event.fault_signal.value, epoch, result.message,
            )

        self._history.append(record)

    # ------------------------------------------------------------------
    # Correlation window (§3.3)

    def _apply_correlation_window(
        self, event: TelemetryEvent, tier: BlastRadius
    ) -> BlastRadius:
        """
        If a B1 is the leading edge of a rack-level outage, re-classify to B4
        *before* committing to the expensive neighbor-absorb path.

        Tunable W (correlation_window_secs) and N (correlation_node_threshold)
        per §6 risk note.
        """
        if tier != BlastRadius.B1:
            return tier

        now = time.monotonic()
        rack = event.rack_id or "unknown"
        self._recent_b1s.append((now, event.node, rack))

        # Prune entries outside the window
        cutoff = now - self._policy.correlation_window_secs
        while self._recent_b1s and self._recent_b1s[0][0] < cutoff:
            self._recent_b1s.popleft()

        same_rack = sum(1 for _, _, r in self._recent_b1s if r == rack)

        if same_rack >= self._policy.correlation_node_threshold:
            logger.warning(
                "[EPE] Correlation window: %d B1s in rack %r within %.1fs → escalating to B4",
                same_rack, rack, self._policy.correlation_window_secs,
            )
            return BlastRadius.B4

        return BlastRadius.B1

    # ------------------------------------------------------------------
    # Routing

    async def _route(self, record: FaultRecord) -> RecoveryResult:
        """
        Route to the appropriate layer, escalating upward on failure.

        Invariant enforced here: tier only increases inside this loop.
        De-escalation (tier decreasing) is structurally impossible because
        we always take `tier = BlastRadius(tier + 1)` when escalating.
        """
        tier = record.final_tier

        while True:
            layer = self._tier_to_layer.get(tier)

            if layer is not None:
                can_handle = await layer.can_handle(record.event, tier)
                if can_handle:
                    # Update final_tier before recovery so the audit log is
                    # correct even if recover() raises.
                    if tier != record.original_tier:
                        record.escalated = True
                    record.final_tier = tier

                    result = await layer.recover(record.event, tier, record.epoch)
                    if result.success:
                        return result

                    logger.warning(
                        "[EPE] Layer %s failed at %s (epoch %d) — escalating",
                        type(layer).__name__, tier.name, record.epoch,
                    )
                else:
                    logger.debug(
                        "[EPE] Layer %s cannot handle %s at %s — escalating",
                        type(layer).__name__, record.event.fault_signal.value, tier.name,
                    )
            else:
                logger.error("[EPE] No layer registered for tier %s", tier.name)

            # Escalate (only upward — invariant enforced)
            next_value = tier.value + 1
            if next_value > BlastRadius.B4.value:
                break
            tier = BlastRadius(next_value)

        return RecoveryResult(
            success=False,
            message=(
                f"All layers exhausted for {record.event.fault_signal.value} "
                f"at epoch {record.epoch}"
            ),
        )

    # ------------------------------------------------------------------
    # Audit

    @property
    def history(self) -> list[FaultRecord]:
        """Immutable copy of the processed-fault audit log."""
        return list(self._history)

    def escalation_audit(self) -> list[dict]:
        """
        Serialisable audit log — used by tests and the operator dashboard.

        Each entry includes ``escalation_valid`` (True iff the one-directional
        invariant holds for that fault) so property tests can assert over it.
        """
        return [
            {
                "epoch": r.epoch,
                "signal": r.event.fault_signal.value,
                "node": r.event.node,
                "rank": r.event.rank,
                "original_tier": r.original_tier.name,
                "final_tier": r.final_tier.name,
                "escalated": r.escalated,
                "success": r.result.success if r.result else False,
                "degraded": r.result.degraded if r.result else False,
                "escalation_valid": r.escalation_valid,
            }
            for r in self._history
        ]
