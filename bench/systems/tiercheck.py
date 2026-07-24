"""
TierCheckAdapter — storage-only baseline (B-TIERCHECK).

TierCheck handles B2/B3/B4 via tiered checkpoints:
  - T1 (local restore): <10s for recoverable software crashes (B2)
  - T2 (peer restore): ~60s for node-level state loss (B3)
  - T3 (cluster restore): ~300s for rack-level outages (B4)

For B0 and B1 (transient NIC / node death), TierCheck has no native
mechanism and falls back to full checkpoint-and-restart.
"""

from __future__ import annotations

from aegis.telemetry.events import BlastRadius

from bench.sim.cost_model import TIERCHECK_RECOVERY_SECS
from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig

from .base import BaseSystemAdapter, RecoveryOutcome


class TierCheckAdapter(BaseSystemAdapter):
    """
    TierCheck storage-only adapter.

    Handles B2/B3/B4 with tiered checkpoint restore; falls back to vanilla
    for B0, B1.
    """

    system_name = "B-TIERCHECK"

    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        tier = fault.expected_tier

        if tier in TIERCHECK_RECOVERY_SECS:
            recovery_secs = TIERCHECK_RECOVERY_SECS[tier]
            tier_label = {
                BlastRadius.B2: "T1-local",
                BlastRadius.B3: "T2-peer",
                BlastRadius.B4: "T3-cluster",
            }[tier]
            return RecoveryOutcome.from_recovery_time(
                recovery_time_secs=recovery_secs,
                gpu_count=workload.gpu_count,
                gpu_hr_cost=gpu_hr_cost,
                tier_handled=tier.name,
                succeeded=True,
                forced_restart=False,
                notes=f"TierCheck handled {tier.name} via {tier_label} restore",
            )

        # Falls back to vanilla checkpoint-and-restart for B0, B1
        return self.vanilla_fallback(fault, workload, gpu_hr_cost)
