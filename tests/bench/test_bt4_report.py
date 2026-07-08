"""
BT-4: Report generation and correctness tests.

Tests that the report generation infrastructure works correctly and that
the headline $/GPU-hr number is positive and traceable.
"""

from __future__ import annotations

import pytest

from bench.metrics.compute import compare_systems, compute_savings_vs_baseline
from bench.report.generate import generate_comparison_matrix, print_report
from bench.sim.engine import SimulationEngine, SimulationResult
from bench.systems.aegis_adapter import AegisAdapter
from bench.systems.mecefo import MeCeFOAdapter
from bench.systems.r2ccl import R2CCLAdapter
from bench.systems.tiercheck import TierCheckAdapter
from bench.systems.torchft import TorchFTAdapter
from bench.systems.vanilla import VanillaAdapter
from bench.traces.mixed import make_ft_poisson, make_ft_production
from bench.traces.per_tier import FT_B0, FT_B1, FT_B2, FT_B3, FT_B4
from bench.workloads.configs import W1


engine = SimulationEngine()


def test_bt4_report_generates_matrix():
    """
    Report generation produces a non-empty ASCII matrix with all systems.

    The matrix must include at least the system names in the output.
    """
    # Build minimal synthetic results (no async needed for report generation)
    from bench.systems.base import RecoveryOutcome

    def _make_result(name: str, dollars: float, goodput: float) -> SimulationResult:
        outcome = RecoveryOutcome.from_recovery_time(
            recovery_time_secs=dollars * 3600 / (W1.gpu_count * 2.35),
            gpu_count=W1.gpu_count,
            gpu_hr_cost=2.35,
            tier_handled="B1",
        )
        r = SimulationResult(
            system_name=name,
            workload_name="W1",
            trace_name="FT-B1",
            fault_outcomes=[outcome],
        )
        r.total_dollars_wasted = dollars
        r.total_idle_gpu_hours = dollars / 2.35
        r.total_recovery_secs = dollars * 3600 / (W1.gpu_count * 2.35)
        r.goodput = goodput
        return r

    results = {
        "AEGIS": _make_result("AEGIS", 0.05, 0.998),
        "B-VANILLA": _make_result("B-VANILLA", 1.20, 0.970),
        "B-R2CCL": _make_result("B-R2CCL", 1.20, 0.970),
        "B-MECEFO": _make_result("B-MECEFO", 0.40, 0.990),
        "B-TIERCHECK": _make_result("B-TIERCHECK", 1.20, 0.970),
        "B-TORCHFT": _make_result("B-TORCHFT", 0.65, 0.985),
    }

    matrix = generate_comparison_matrix(results, baseline_name="B-VANILLA")

    # Matrix must be a non-empty string
    assert isinstance(matrix, str)
    assert len(matrix) > 0

    # All system names must appear in the matrix
    for name in results:
        assert name in matrix, f"System '{name}' should appear in the comparison matrix"

    # Matrix must contain the column headers
    assert "$/GPU-hr wasted" in matrix or "Goodput" in matrix


def test_bt4_report_matrix_baseline_marker():
    """The baseline row is marked '(baseline)' in the savings column."""
    from bench.systems.base import RecoveryOutcome

    def _make_result(name: str, dollars: float) -> SimulationResult:
        outcome = RecoveryOutcome.from_recovery_time(
            recovery_time_secs=10.0,
            gpu_count=8,
            gpu_hr_cost=2.35,
            tier_handled="B0",
        )
        r = SimulationResult(
            system_name=name,
            workload_name="W1",
            trace_name="FT-B0",
            fault_outcomes=[outcome],
        )
        r.total_dollars_wasted = dollars
        r.total_idle_gpu_hours = dollars / 2.35
        r.total_recovery_secs = 10.0
        r.goodput = 0.99
        return r

    results = {
        "AEGIS": _make_result("AEGIS", 0.05),
        "B-VANILLA": _make_result("B-VANILLA", 1.20),
    }

    matrix = generate_comparison_matrix(results, baseline_name="B-VANILLA")
    assert "(baseline)" in matrix, "Baseline row should be marked '(baseline)'"


async def test_bt4_report_headline_figure_is_positive():
    """
    The $/GPU-hr saved headline number is positive for AEGIS vs B-VANILLA
    on the per-tier B1 trace. AEGIS (~25s) saves money vs vanilla (~120s+).
    """
    aegis_result = await engine.run(AegisAdapter(), W1, FT_B1)
    vanilla_result = await engine.run(VanillaAdapter(), W1, FT_B1)

    savings = compute_savings_vs_baseline(aegis_result, vanilla_result)

    assert savings["dollars_saved"] > 0, (
        f"Headline $/GPU-hr saved must be positive: got {savings['dollars_saved']:.4f}"
    )
    assert savings["gpu_hours_saved"] > 0, (
        f"GPU hours saved must be positive: got {savings['gpu_hours_saved']:.4f}"
    )
    assert savings["goodput_improvement"] >= 0, (
        f"Goodput improvement must be non-negative: got {savings['goodput_improvement']:.4f}"
    )
    assert savings["recovery_time_reduction_pct"] > 0, (
        f"Recovery time reduction must be positive: got {savings['recovery_time_reduction_pct']:.1f}%"
    )


def test_bt4_trace_determinism():
    """
    Same trace generates identically across calls (seeded RNG).

    FT-POISSON and FT-PRODUCTION must produce identical event sequences
    when called with the same seed.
    """
    trace1 = make_ft_poisson(seed=42)
    trace2 = make_ft_poisson(seed=42)

    assert len(trace1.events) == len(trace2.events), (
        "Same seed must produce the same number of events"
    )
    for i, (e1, e2) in enumerate(zip(trace1.events, trace2.events)):
        assert e1 == e2, (
            f"Event {i} differs between identical seeds: {e1} != {e2}"
        )

    # Different seeds produce different traces
    trace3 = make_ft_poisson(seed=99)
    assert len(trace1.events) != len(trace3.events) or any(
        e1 != e2 for e1, e2 in zip(trace1.events, trace3.events)
    ), "Different seeds should produce different traces"


def test_bt4_production_trace_determinism():
    """FT-PRODUCTION trace is deterministic with same seed."""
    trace1 = make_ft_production(seed=2024)
    trace2 = make_ft_production(seed=2024)

    assert len(trace1.events) == len(trace2.events)
    for i, (e1, e2) in enumerate(zip(trace1.events, trace2.events)):
        assert e1 == e2, f"Production trace event {i} differs between seeds"


def test_bt4_compare_systems_sorted():
    """compare_systems() returns rows sorted by dollars_wasted ascending."""
    from bench.systems.base import RecoveryOutcome

    def _make_result(name: str, dollars: float) -> SimulationResult:
        r = SimulationResult(
            system_name=name,
            workload_name="W1",
            trace_name="FT-B1",
            fault_outcomes=[],
        )
        r.total_dollars_wasted = dollars
        r.total_idle_gpu_hours = dollars / 2.35
        r.total_recovery_secs = dollars * 3600 / (8 * 2.35)
        r.goodput = 1.0 - dollars / 10
        return r

    results = {
        "AEGIS": _make_result("AEGIS", 0.05),
        "B-VANILLA": _make_result("B-VANILLA", 1.20),
        "B-TORCHFT": _make_result("B-TORCHFT", 0.65),
    }

    rows = compare_systems(results, baseline_name="B-VANILLA")

    # Rows should be sorted by dollars_wasted ascending
    for i in range(len(rows) - 1):
        assert rows[i]["total_dollars_wasted"] <= rows[i + 1]["total_dollars_wasted"], (
            f"compare_systems() must sort ascending by dollars_wasted"
        )


async def test_bt4_full_production_report_prints():
    """
    Full production trace report prints without error and AEGIS is the winner.
    """
    trace = make_ft_production(seed=2024)

    systems = [
        AegisAdapter(),
        VanillaAdapter(),
        R2CCLAdapter(),
        MeCeFOAdapter(),
        TierCheckAdapter(),
        TorchFTAdapter(),
    ]

    results = {}
    for adapter in systems:
        results[adapter.system_name] = await engine.run(adapter, W1, trace)

    # Should not raise
    matrix = generate_comparison_matrix(results, baseline_name="B-VANILLA")
    assert "AEGIS" in matrix

    # AEGIS must be the cheapest system
    aegis_cost = results["AEGIS"].total_dollars_wasted
    for name, result in results.items():
        if name == "AEGIS":
            continue
        assert aegis_cost <= result.total_dollars_wasted, (
            f"AEGIS (${aegis_cost:.4f}) must be cheapest; {name} costs "
            f"${result.total_dollars_wasted:.4f}"
        )
