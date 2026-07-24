"""
R2CCLAdapter — transport-only baseline (B-R2CCL).

R²CCL handles B0 (NIC/link failures) via QP migration to a backup NIC.
For any other fault class (B1+), it has no recovery mechanism and falls
back to full checkpoint-and-restart.

This isolates the "transport only" point on the composition spectrum and
directly tests the claim that AEGIS's composition beats a single layer.
"""

from __future__ import annotations

from aegis.telemetry.events import BlastRadius

from bench.sim.cost_model import R2CCL_RECOVERY_SECS
from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig

from .base import BaseSystemAdapter, RecoveryOutcome


class R2CCLAdapter(BaseSystemAdapter):
    """
    R²CCL transport-only adapter.

    Handles B0 with the R²CCL cost; falls back to vanilla for B1+.
    """

    system_name = "B-R2CCL"

    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        tier = fault.expected_tier

        if tier in R2CCL_RECOVERY_SECS:
            recovery_secs = R2CCL_RECOVERY_SECS[tier]
            return RecoveryOutcome.from_recovery_time(
                recovery_time_secs=recovery_secs,
                gpu_count=workload.gpu_count,
                gpu_hr_cost=gpu_hr_cost,
                tier_handled=tier.name,
                succeeded=True,
                forced_restart=False,
                notes=f"R²CCL handled {tier.name} via NIC migration",
            )

        # Falls back to vanilla checkpoint-and-restart for B1+
        return self.vanilla_fallback(fault, workload, gpu_hr_cost)
