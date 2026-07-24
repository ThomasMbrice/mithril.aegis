"""
Benchmark runner — orchestrates system × workload × trace × repeat runs.

The runner produces a dict of SimulationResult objects keyed by system name,
suitable for passing to report/generate.py.
"""

from __future__ import annotations

import asyncio

from bench.sim.engine import SimulationEngine, SimulationResult
from bench.systems.base import BaseSystemAdapter
from bench.traces.per_tier import FaultTrace
from bench.workloads.configs import WorkloadConfig


async def run_benchmark(
    systems: list[BaseSystemAdapter],
    workload: WorkloadConfig,
    trace: FaultTrace,
    gpu_hr_cost: float = 2.35,
    total_steps: int = 50_000,
    n_repeats: int = 1,
) -> dict[str, SimulationResult]:
    """
    Run a full benchmark: all systems against one workload + trace.

    For each system, runs n_repeats simulations and returns the result
    from the final repeat (deterministic traces mean repeats are identical
    for non-AEGIS adapters; AEGIS adapters are also deterministic via the
    EPE's fixed routing logic).

    Args:
        systems: List of system adapters to evaluate.
        workload: The training workload configuration.
        trace: The fault trace to replay.
        gpu_hr_cost: GPU hourly cost in $/GPU-hr.
        total_steps: Total training steps for goodput calculation.
        n_repeats: Number of repeat runs (for variance measurement).

    Returns:
        Mapping of system_name → SimulationResult (last repeat's result).
    """
    engine = SimulationEngine()
    results: dict[str, SimulationResult] = {}

    for system in systems:
        last_result = None
        for _ in range(n_repeats):
            result = await engine.run(
                system=system,
                workload=workload,
                trace=trace,
                gpu_hr_cost=gpu_hr_cost,
                total_steps=total_steps,
            )
            last_result = result

        if last_result is not None:
            results[system.system_name] = last_result

    return results
