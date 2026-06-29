"""
AegisRuntime — top-level orchestrator.

Wires all AEGIS components following the build order specified in §4:
  A (UTP, FC, Epoch) → B ∥ C ∥ D (layers) → E (EPE + policy)

Intended usage:

    async with AegisRuntime() as rt:
        rt.transport.register_node_nics("node0", ["nic0", "nic1"])
        rt.compute.register_neighbor("node0", "node1")
        rt.storage.write_tier3(epoch=0)
        # publish events via rt.utp.publish(event) or chaos harness
"""

from __future__ import annotations

import logging
from types import TracebackType

from .classifier.classifier import FailureClassifier
from .consensus.urc import UnifiedRecoveryConsensus
from .epoch.service import FaultEpochService
from .kpi import KPIMeter
from .layers.compute import ComputeLayer
from .layers.storage import StorageLayer
from .layers.transport import TransportLayer
from .policy.dsl import OperatorPolicy
from .policy.engine import EscalationPolicyEngine
from .telemetry.bus import UnifiedTelemetryPlane
from .telemetry.sensors import SyntheticSensor

logger = logging.getLogger(__name__)


class AegisRuntime:
    """
    Top-level runtime.

    All public attributes are the individual components — callers may
    configure them directly (e.g., register nodes, write seed checkpoints)
    before starting the runtime.
    """

    def __init__(self, policy: OperatorPolicy | None = None) -> None:
        policy = policy or OperatorPolicy()

        # ---- Layer A: foundation -----------------------------------------
        self.utp = UnifiedTelemetryPlane()
        self.classifier = FailureClassifier()
        self.epoch_service = FaultEpochService()

        # ---- Layers B / C / D: recovery stubs ---------------------------
        self.transport = TransportLayer()
        self.compute = ComputeLayer()
        self.storage = StorageLayer()

        # ---- URC: cross-layer consensus ----------------------------------
        self.consensus = UnifiedRecoveryConsensus()

        # ---- KPI meter ---------------------------------------------------
        self.kpi = KPIMeter(gpu_hr_cost=policy.gpu_hr_cost)

        # ---- Layer E: EPE (integration keystone) -------------------------
        # EPE subscribes to UTP in its __init__; events are dispatched once
        # utp.start() is called.
        self.epe = EscalationPolicyEngine(
            utp=self.utp,
            classifier=self.classifier,
            epoch_service=self.epoch_service,
            consensus=self.consensus,
            policy=policy,
            layers=[self.transport, self.compute, self.storage],
        )

        # Synthetic sensor for testing / chaos injection
        self.sensor = SyntheticSensor(self.utp)

        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle

    async def start(self) -> None:
        if self._started:
            return
        logger.info("[AEGIS] Runtime starting — UTP, EPE, epoch service active")
        await self.utp.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        logger.info("[AEGIS] Runtime stopping")
        await self.utp.stop()
        self._started = False
        logger.info("[AEGIS] Runtime stopped")

    async def __aenter__(self) -> "AegisRuntime":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        await self.stop()
        return False  # never suppress exceptions
