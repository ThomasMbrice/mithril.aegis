"""
RecoveryOutcome dataclass and BaseSystemAdapter ABC.

All system adapters implement the same interface so the simulation engine
can drive them interchangeably.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from bench.traces.per_tier import TraceFaultEvent
from bench.workloads.configs import WorkloadConfig


@dataclass
class RecoveryOutcome:
    """
    Result of a single fault recovery for one system under test.

    idle_gpu_hours and dollars_wasted are derived from recovery_time_secs
    and can be compared directly across systems.
    """

    recovery_time_secs: float
    idle_gpu_hours: float          # = recovery_time_secs × gpu_count / 3600
    dollars_wasted: float          # = idle_gpu_hours × gpu_hr_cost
    succeeded: bool
    tier_handled: str              # 'B0', 'B1', ... or 'full_restart'
    forced_restart: bool = False
    notes: str = ""

    @classmethod
    def from_recovery_time(
        cls,
        recovery_time_secs: float,
        gpu_count: int,
        gpu_hr_cost: float,
        tier_handled: str,
        succeeded: bool = True,
        forced_restart: bool = False,
        notes: str = "",
    ) -> "RecoveryOutcome":
        """Construct from a recovery time, computing derived fields."""
        idle_gpu_hours = recovery_time_secs * gpu_count / 3600.0
        dollars_wasted = idle_gpu_hours * gpu_hr_cost
        return cls(
            recovery_time_secs=recovery_time_secs,
            idle_gpu_hours=idle_gpu_hours,
            dollars_wasted=dollars_wasted,
            succeeded=succeeded,
            tier_handled=tier_handled,
            forced_restart=forced_restart,
            notes=notes,
        )


class BaseSystemAdapter(abc.ABC):
    """
    Abstract adapter for a fault-tolerance system under test.

    Each adapter wraps one system (AEGIS, B-VANILLA, B-R2CCL, etc.) and
    exposes a uniform interface for the simulation engine to call.
    """

    #: Human-readable system name, used in reports
    system_name: str = "unknown"

    def handle_fault(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float = 2.35,
    ) -> RecoveryOutcome:
        """
        Handle a fault event and return a recovery outcome.

        Synchronous wrapper — calls _handle_fault_impl().
        """
        return self._handle_fault_impl(fault, workload, gpu_hr_cost)

    @abc.abstractmethod
    def _handle_fault_impl(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
    ) -> RecoveryOutcome:
        """Implement the actual fault handling logic."""

    def vanilla_fallback(
        self,
        fault: TraceFaultEvent,
        workload: WorkloadConfig,
        gpu_hr_cost: float,
        notes: str = "",
    ) -> RecoveryOutcome:
        """
        Compute a vanilla checkpoint-and-restart recovery outcome.

        Used by partial-coverage adapters (R2CCL, MeCeFO, TierCheck) when
        they encounter a fault class they cannot handle natively.
        """
        last_ckpt = (
            fault.step // workload.checkpoint_interval_steps
        ) * workload.checkpoint_interval_steps
        recovery_secs = workload.vanilla_rollback_secs(fault.step, last_ckpt)
        return RecoveryOutcome.from_recovery_time(
            recovery_time_secs=recovery_secs,
            gpu_count=workload.gpu_count,
            gpu_hr_cost=gpu_hr_cost,
            tier_handled="full_restart",
            succeeded=True,
            forced_restart=True,
            notes=notes or f"Forced full restart: {self.system_name} cannot handle {fault.expected_tier.name}",
        )
