"""
VanillaAdapter — checkpoint-and-restart baseline (B-VANILLA).

Represents the status-quo: on any fault, restart the entire job from the
most recent periodic checkpoint. This is what AEGIS is measured against
as the core thesis — per-step/per-tier recovery beats full rollback.
"""

from __future__ import annotations

from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig

from .base import BaseSystemAdapter, RecoveryOutcome


class VanillaAdapter(BaseSystemAdapter):
    """
    Checkpoint-and-restart adapter.

    On every fault, rolls back to the most recent checkpoint and restarts
    the full training job. Recovery time is:
        lost_steps × step_duration + checkpoint_restore_secs
    """

    system_name = "B-VANILLA"

    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        last_ckpt = (
            fault.step // workload.checkpoint_interval_steps
        ) * workload.checkpoint_interval_steps
        recovery_secs = workload.vanilla_rollback_secs(fault.step, last_ckpt)
        lost_steps = fault.step - last_ckpt

        return RecoveryOutcome.from_recovery_time(
            recovery_time_secs=recovery_secs,
            gpu_count=workload.gpu_count,
            gpu_hr_cost=gpu_hr_cost,
            tier_handled="full_restart",
            succeeded=True,
            forced_restart=True,
            notes=(
                f"Checkpoint-and-restart: rolled back {lost_steps} steps "
                f"({recovery_secs:.1f}s total)"
            ),
        )
