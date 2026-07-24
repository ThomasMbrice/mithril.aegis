"""
AegisAdapter — uses the real ChaosHarness + EPE to handle faults.

This is the only adapter that exercises the actual AEGIS runtime rather
than a cost-model simulation. It injects faults via the chaos harness,
waits for the EPE to process them, then reads the audit log to determine
which tier was used and maps it to the cost table.
"""

from __future__ import annotations

import asyncio

from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from aegis.telemetry.events import BlastRadius
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness

from bench.sim.cost_model import AEGIS_RECOVERY_SECS
from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig

from .base import BaseSystemAdapter, RecoveryOutcome


def _configure_runtime(rt: AegisRuntime) -> None:
    """Populate a runtime with the standard 4-node, 2-rack topology."""
    nodes = ["node0", "node1", "node2", "node3"]

    for node in nodes:
        rt.transport.register_node_nics(node, ["nic0", "nic1"])

    for i, node in enumerate(nodes):
        neighbor = nodes[(i + 1) % len(nodes)]
        rt.compute.register_neighbor(node, neighbor)

    for node in nodes:
        rt.storage.write_tier1(node, epoch=0)
        rt.storage.write_tier2(node, epoch=0)

    rt.storage.write_tier3(epoch=0)

    for i in range(len(nodes)):
        rt.consensus.report_epoch(rank=i, epoch=0)


_TIER_MAP: dict[str, BlastRadius] = {
    "B0": BlastRadius.B0,
    "B1": BlastRadius.B1,
    "B2": BlastRadius.B2,
    "B3": BlastRadius.B3,
    "B4": BlastRadius.B4,
}


class AegisAdapter(BaseSystemAdapter):
    """
    Adapter that drives the real AEGIS runtime.

    Each call to handle_fault spins up a fresh runtime, injects the fault
    via ChaosHarness, lets the EPE settle, then reads the audit to
    determine the actual tier used and maps it to AEGIS_RECOVERY_SECS.
    """

    system_name = "AEGIS"

    def __init__(self, policy: OperatorPolicy | None = None) -> None:
        self._policy = policy or OperatorPolicy(
            correlation_window_secs=1.0,
            correlation_node_threshold=3,
        )

    async def handle_fault_async(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float = 2.35,
    ) -> RecoveryOutcome:
        """
        Inject fault into a real runtime and return a RecoveryOutcome.

        Steps:
        1. Build a fresh runtime + chaos harness with standard topology
        2. Inject the fault via chaos.inject(FaultSpec(...))
        3. Sleep briefly to let the EPE process the event
        4. Read runtime.epe.escalation_audit() for final_tier
        5. Map final_tier → AEGIS_RECOVERY_SECS[tier]
        6. Return RecoveryOutcome
        """
        rt = AegisRuntime(policy=self._policy)
        _configure_runtime(rt)

        async with rt:
            chaos = ChaosHarness(rt.utp, rt.epoch_service)

            await chaos.inject(
                FaultSpec(
                    fault_signal=fault.fault_signal,
                    rank=fault.target_rank,
                    node=fault.target_node,
                    nic_id=fault.target_nic,
                    rack_id=fault.target_rack,
                )
            )

            # Allow the EPE's async dispatch loop to process the event
            await asyncio.sleep(0.15)

            audit = rt.epe.escalation_audit()

        if not audit:
            # No events processed — fall back to vanilla cost
            last_ckpt = (
                fault.step // workload.checkpoint_interval_steps
            ) * workload.checkpoint_interval_steps
            recovery_secs = workload.vanilla_rollback_secs(fault.step, last_ckpt)
            return RecoveryOutcome.from_recovery_time(
                recovery_time_secs=recovery_secs,
                gpu_count=workload.gpu_count,
                gpu_hr_cost=gpu_hr_cost,
                tier_handled="full_restart",
                succeeded=False,
                forced_restart=True,
                notes="EPE produced no audit entries",
            )

        # Use the last audit entry (most recent routing decision)
        last_entry = audit[-1]
        final_tier_name = last_entry["final_tier"]
        tier = _TIER_MAP.get(final_tier_name, BlastRadius.B4)
        recovery_secs = AEGIS_RECOVERY_SECS[tier]
        succeeded = last_entry["success"]

        return RecoveryOutcome.from_recovery_time(
            recovery_time_secs=recovery_secs,
            gpu_count=workload.gpu_count,
            gpu_hr_cost=gpu_hr_cost,
            tier_handled=final_tier_name,
            succeeded=succeeded,
            forced_restart=False,
            notes=f"EPE routed {last_entry['signal']} → {final_tier_name}",
        )

    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        """Synchronous wrapper: runs handle_fault_async in a new event loop."""
        return asyncio.run(self.handle_fault_async(fault, workload, gpu_hr_cost))
