"""
SimulationEngine — async engine that runs (system × workload × trace) simulations.

Each simulation produces a SimulationResult with aggregate metrics:
total recovery time, idle GPU-hours, dollars wasted, and goodput.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from bench.systems.base import BaseSystemAdapter, RecoveryOutcome
from bench.traces.per_tier import FaultTrace
from bench.workloads.configs import WorkloadConfig


@dataclass
class SimulationResult:
    """
    Aggregated result of one (system × workload × trace) simulation.

    Contains both the per-fault outcomes and aggregate metrics for
    comparison and reporting.
    """

    system_name: str
    workload_name: str
    trace_name: str
    fault_outcomes: list[RecoveryOutcome] = field(default_factory=list)
    total_recovery_secs: float = 0.0
    total_idle_gpu_hours: float = 0.0
    total_dollars_wasted: float = 0.0
    goodput: float = 1.0  # productive_steps / total_steps, clamped to [0, 1]


class SimulationEngine:
    """
    Async simulation engine.

    Drives a system adapter through a fault trace and accumulates metrics.
    The engine is stateless — each call to run() is independent.
    """

    async def run(
        self,
        system: BaseSystemAdapter,
        workload: WorkloadConfig,
        trace: FaultTrace,
        gpu_hr_cost: float = 2.35,
        total_steps: int = 50_000,
    ) -> SimulationResult:
        """
        Run one (system × workload × trace) simulation.

        For each fault in the trace:
          1. Call system.handle_fault() (or handle_fault_async if available)
          2. Accumulate RecoveryOutcome into aggregate metrics
          3. Compute goodput from productive vs. total steps

        Args:
            system: The system adapter to evaluate.
            workload: The training workload configuration.
            trace: The fault trace to replay.
            gpu_hr_cost: GPU hourly cost in $/GPU-hr.
            total_steps: Total training steps (denominator for goodput).

        Returns:
            SimulationResult with per-fault outcomes and aggregate metrics.
        """
        result = SimulationResult(
            system_name=system.system_name,
            workload_name=workload.name,
            trace_name=trace.name,
        )

        for fault in trace.events:
            # Use async variant if available (AegisAdapter), otherwise sync
            if hasattr(system, "handle_fault_async"):
                outcome = await system.handle_fault_async(fault, workload, gpu_hr_cost)
            else:
                # Run sync adapter in executor to avoid blocking event loop
                outcome = await asyncio.get_event_loop().run_in_executor(
                    None, system.handle_fault, fault, workload, gpu_hr_cost
                )

            result.fault_outcomes.append(outcome)

        # Aggregate metrics
        result.total_recovery_secs = sum(o.recovery_time_secs for o in result.fault_outcomes)
        result.total_idle_gpu_hours = sum(o.idle_gpu_hours for o in result.fault_outcomes)
        result.total_dollars_wasted = sum(o.dollars_wasted for o in result.fault_outcomes)

        # Goodput: productive_steps / total_steps, clamped to [0, 1]
        # Each recovery_time_secs means the training was stalled for that long,
        # which translates to wasted steps at the workload's step rate.
        recovery_steps_wasted = sum(
            o.recovery_time_secs * workload.steps_per_second
            for o in result.fault_outcomes
        )
        productive_steps = max(0.0, total_steps - recovery_steps_wasted)
        result.goodput = min(1.0, max(0.0, productive_steps / total_steps))

        return result
