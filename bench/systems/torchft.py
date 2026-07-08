"""
TorchFTAdapter — elastic training baseline (B-TORCHFT).

TorchFT is the direct OSS competitor: elastic training that handles all
fault tiers, but more slowly than AEGIS's composed runtime. Beating TorchFT
proves that composition with tier-specific recovery primitives is faster than
a generic elastic training approach.
"""

from __future__ import annotations

from aegis.telemetry.events import BlastRadius

from bench.sim.cost_model import TORCHFT_RECOVERY_SECS
from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig

from .base import BaseSystemAdapter, RecoveryOutcome


class TorchFTAdapter(BaseSystemAdapter):
    """
    TorchFT elastic training adapter.

    Handles all fault tiers, but more slowly than AEGIS for every tier.
    Uses the TORCHFT_RECOVERY_SECS cost table.
    """

    system_name = "B-TORCHFT"

    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        tier = fault.expected_tier
        recovery_secs = TORCHFT_RECOVERY_SECS.get(tier, TORCHFT_RECOVERY_SECS[BlastRadius.B4])

        return RecoveryOutcome.from_recovery_time(
            recovery_time_secs=recovery_secs,
            gpu_count=workload.gpu_count,
            gpu_hr_cost=gpu_hr_cost,
            tier_handled=tier.name,
            succeeded=True,
            forced_restart=False,
            notes=f"TorchFT elastic training handled {tier.name}",
        )
