"""
MeCeFOAdapter — compute-only baseline (B-MECEFO).

MeCeFO handles B1 (node/GPU death) via neighbor-absorb — a surviving
neighbor node absorbs the dead rank's workload using approximate state.
For any other fault class (B0, B2+), it has no recovery mechanism and
falls back to full checkpoint-and-restart.
"""

from __future__ import annotations

from aegis.telemetry.events import BlastRadius

from bench.sim.cost_model import MECEFO_RECOVERY_SECS
from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig

from .base import BaseSystemAdapter, RecoveryOutcome


class MeCeFOAdapter(BaseSystemAdapter):
    """
    MeCeFO compute-only adapter.

    Handles B1 with the MeCeFO neighbor-absorb cost; falls back to vanilla
    for B0, B2, B3, B4.
    """

    system_name = "B-MECEFO"

    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        tier = fault.expected_tier

        if tier in MECEFO_RECOVERY_SECS:
            recovery_secs = MECEFO_RECOVERY_SECS[tier]
            return RecoveryOutcome.from_recovery_time(
                recovery_time_secs=recovery_secs,
                gpu_count=workload.gpu_count,
                gpu_hr_cost=gpu_hr_cost,
                tier_handled=tier.name,
                succeeded=True,
                forced_restart=False,
                notes=f"MeCeFO handled {tier.name} via neighbor-absorb",
            )

        # Falls back to vanilla checkpoint-and-restart for B0, B2+
        return self.vanilla_fallback(fault, workload, gpu_hr_cost)
