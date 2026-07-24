"""
Metrics computation and comparison functions for the benchmark suite.

The hero metric is $/GPU-hr saved vs. each baseline. Supporting metrics
are goodput and per-tier recovery time.
"""

from __future__ import annotations

from dataclasses import dataclass

from bench.sim.engine import SimulationResult


# Alias for readability in external code
SimResult = SimulationResult


def compute_metrics(result: SimulationResult) -> dict:
    """
    Compute summary metrics for a single simulation result.

    Returns a dict with:
      - system_name, workload_name, trace_name
      - total_recovery_secs, total_idle_gpu_hours, total_dollars_wasted
      - goodput
      - fault_count
      - forced_restart_count
    """
    forced_restarts = sum(1 for o in result.fault_outcomes if o.forced_restart)
    return {
        "system_name": result.system_name,
        "workload_name": result.workload_name,
        "trace_name": result.trace_name,
        "total_recovery_secs": result.total_recovery_secs,
        "total_idle_gpu_hours": result.total_idle_gpu_hours,
        "total_dollars_wasted": result.total_dollars_wasted,
        "goodput": result.goodput,
        "fault_count": len(result.fault_outcomes),
        "forced_restart_count": forced_restarts,
    }


def compute_savings_vs_baseline(
    system_result: SimulationResult,
    baseline_result: SimulationResult,
) -> dict:
    """
    Compute savings of system_result vs. baseline_result.

    Returns:
        {
            'dollars_saved': float,
            'gpu_hours_saved': float,
            'goodput_improvement': float,
            'recovery_time_reduction_pct': float,
        }
    """
    dollars_saved = (
        baseline_result.total_dollars_wasted - system_result.total_dollars_wasted
    )
    gpu_hours_saved = (
        baseline_result.total_idle_gpu_hours - system_result.total_idle_gpu_hours
    )
    goodput_improvement = system_result.goodput - baseline_result.goodput

    if baseline_result.total_recovery_secs > 0:
        recovery_reduction_pct = (
            (baseline_result.total_recovery_secs - system_result.total_recovery_secs)
            / baseline_result.total_recovery_secs
            * 100.0
        )
    else:
        recovery_reduction_pct = 0.0

    return {
        "dollars_saved": dollars_saved,
        "gpu_hours_saved": gpu_hours_saved,
        "goodput_improvement": goodput_improvement,
        "recovery_time_reduction_pct": recovery_reduction_pct,
    }


def compare_systems(
    results: dict[str, SimulationResult],
    baseline_name: str = "B-VANILLA",
) -> list[dict]:
    """
    Compare all systems against a baseline and return sorted comparison rows.

    Args:
        results: Mapping of system_name → SimulationResult.
        baseline_name: Name of the baseline system to compare against.

    Returns:
        List of dicts, one per system, sorted by dollars_wasted ascending.
        Each dict contains compute_metrics() output plus savings vs. baseline.
    """
    baseline = results.get(baseline_name)
    rows = []

    for name, result in results.items():
        row = compute_metrics(result)

        if baseline is not None and name != baseline_name:
            savings = compute_savings_vs_baseline(result, baseline)
            row.update(savings)
        else:
            row.update({
                "dollars_saved": 0.0,
                "gpu_hours_saved": 0.0,
                "goodput_improvement": 0.0,
                "recovery_time_reduction_pct": 0.0,
            })

        rows.append(row)

    rows.sort(key=lambda r: r["total_dollars_wasted"])
    return rows
