"""
BT-3: Mixed trace tests.

Tests using the realistic mixed fault traces (FT-POISSON, FT-BURST, FT-PRODUCTION)
that drive the $/GPU-hr hero number.

FT-BURST is the critical test: correlated B1→B4 escalation is exactly where
AEGIS should win biggest against single-primitive baselines.
"""

from __future__ import annotations

import pytest

from bench.metrics.compute import compute_savings_vs_baseline
from bench.report.generate import generate_comparison_matrix, print_report
from bench.sim.engine import SimulationEngine
from bench.systems.aegis_adapter import AegisAdapter
from bench.systems.mecefo import MeCeFOAdapter
from bench.systems.r2ccl import R2CCLAdapter
from bench.systems.tiercheck import TierCheckAdapter
from bench.systems.torchft import TorchFTAdapter
from bench.systems.vanilla import VanillaAdapter
from bench.traces.mixed import make_ft_burst, make_ft_poisson, make_ft_production
from bench.workloads.configs import W1


engine = SimulationEngine()


async def test_bt3_burst_trace_aegis_wins_biggest():
    """
    FT-BURST: correlated B1→B4 cascade.

    Single-primitive baselines handle worst:
    - B-R2CCL: forced full restart on all B1+ events (only handles B0)
    - AEGIS: correlation window fires, uses T3 restore for B4; always better
      than vanilla and R2CCL which must do many full restarts.

    Note: MeCeFO handles the B1 events natively (its home tier), so its
    recovery cost is competitive on the B1-heavy portion of the burst. The
    burst trace specifically tests R2CCL (can't handle B1/B4) and vanilla.
    """
    trace = make_ft_burst(seed=42)

    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()
    vanilla = VanillaAdapter()

    aegis_result = await engine.run(aegis, W1, trace)
    r2ccl_result = await engine.run(r2ccl, W1, trace)
    vanilla_result = await engine.run(vanilla, W1, trace)

    aegis_cost = aegis_result.total_dollars_wasted

    assert aegis_cost < r2ccl_result.total_dollars_wasted, (
        f"AEGIS (${aegis_cost:.4f}) must beat R2CCL "
        f"(${r2ccl_result.total_dollars_wasted:.4f}) on burst trace"
    )
    assert aegis_cost < vanilla_result.total_dollars_wasted, (
        f"AEGIS (${aegis_cost:.4f}) must beat B-VANILLA "
        f"(${vanilla_result.total_dollars_wasted:.4f}) on burst trace"
    )


async def test_bt3_burst_trace_r2ccl_mecefo_forced_restart_on_b4():
    """
    FT-BURST: R2CCL and MeCeFO are forced to full restart on B4 events.
    This is their worst case — the case AEGIS was designed for.
    """
    trace = make_ft_burst(seed=42)

    r2ccl = R2CCLAdapter()
    mecefo = MeCeFOAdapter()

    r2ccl_result = await engine.run(r2ccl, W1, trace)
    mecefo_result = await engine.run(mecefo, W1, trace)

    # At least one forced restart should happen for events they can't handle
    assert any(o.forced_restart for o in r2ccl_result.fault_outcomes), (
        "R2CCL should have at least one forced restart on the burst trace"
    )
    assert any(o.forced_restart for o in mecefo_result.fault_outcomes), (
        "MeCeFO should have at least one forced restart on the burst trace"
    )


async def test_bt3_production_trace_headline_number():
    """
    FT-PRODUCTION trace: compute the $/GPU-hr headline.

    Run all systems on FT-PRODUCTION × W1, print the comparison table,
    and assert:
    - AEGIS saves money vs. each baseline
    - dollars_saved_vs_vanilla > 0
    """
    trace = make_ft_production(seed=2024)

    systems = {
        "AEGIS": AegisAdapter(),
        "B-VANILLA": VanillaAdapter(),
        "B-R2CCL": R2CCLAdapter(),
        "B-MECEFO": MeCeFOAdapter(),
        "B-TIERCHECK": TierCheckAdapter(),
        "B-TORCHFT": TorchFTAdapter(),
    }

    results = {}
    for name, adapter in systems.items():
        results[name] = await engine.run(adapter, W1, trace)

    # Print the comparison table
    print_report(results, baseline_name="B-VANILLA")

    aegis_result = results["AEGIS"]
    vanilla_result = results["B-VANILLA"]

    savings = compute_savings_vs_baseline(aegis_result, vanilla_result)

    assert savings["dollars_saved"] > 0, (
        f"AEGIS must save money vs B-VANILLA on production trace: "
        f"AEGIS=${aegis_result.total_dollars_wasted:.4f}, "
        f"vanilla=${vanilla_result.total_dollars_wasted:.4f}"
    )

    # AEGIS should also beat each single-primitive baseline
    for name in ["B-R2CCL", "B-MECEFO", "B-TIERCHECK", "B-TORCHFT"]:
        baseline_result = results[name]
        assert aegis_result.total_dollars_wasted <= baseline_result.total_dollars_wasted, (
            f"AEGIS must have <= dollars_wasted than {name} on production trace"
        )


async def test_bt3_poisson_goodput_comparison():
    """
    FT-POISSON: goodput vs MTBF comparison across systems.

    On a Poisson fault trace with mixed fault types, AEGIS should achieve
    the highest goodput due to optimal per-tier routing.
    """
    trace = make_ft_poisson(seed=42)

    systems = {
        "AEGIS": AegisAdapter(),
        "B-VANILLA": VanillaAdapter(),
        "B-R2CCL": R2CCLAdapter(),
        "B-MECEFO": MeCeFOAdapter(),
    }

    results = {}
    for name, adapter in systems.items():
        results[name] = await engine.run(adapter, W1, trace)

    aegis_goodput = results["AEGIS"].goodput

    # AEGIS should have better or equal goodput than vanilla
    assert aegis_goodput >= results["B-VANILLA"].goodput, (
        f"AEGIS goodput ({aegis_goodput:.4f}) should be >= "
        f"B-VANILLA goodput ({results['B-VANILLA'].goodput:.4f})"
    )

    # AEGIS should have better or equal goodput than single-primitives
    for name in ["B-R2CCL", "B-MECEFO"]:
        assert aegis_goodput >= results[name].goodput, (
            f"AEGIS goodput ({aegis_goodput:.4f}) should be >= "
            f"{name} goodput ({results[name].goodput:.4f})"
        )


async def test_bt3_poisson_aegis_lower_cost_than_vanilla():
    """FT-POISSON: AEGIS total cost lower than B-VANILLA."""
    trace = make_ft_poisson(seed=42)

    aegis_result = await engine.run(AegisAdapter(), W1, trace)
    vanilla_result = await engine.run(VanillaAdapter(), W1, trace)

    assert aegis_result.total_dollars_wasted <= vanilla_result.total_dollars_wasted, (
        "AEGIS must cost <= vanilla on Poisson trace"
    )


async def test_bt3_production_trace_aegis_goodput_best():
    """FT-PRODUCTION: AEGIS achieves highest goodput among all systems."""
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

    aegis_goodput = results["AEGIS"].goodput

    for name, result in results.items():
        if name == "AEGIS":
            continue
        assert aegis_goodput >= result.goodput, (
            f"AEGIS goodput ({aegis_goodput:.4f}) must be >= {name} ({result.goodput:.4f})"
        )


async def test_bt3_burst_aegis_beats_torchft():
    """FT-BURST: AEGIS beats TorchFT even though TorchFT handles all tiers."""
    trace = make_ft_burst(seed=42)

    aegis_result = await engine.run(AegisAdapter(), W1, trace)
    torchft_result = await engine.run(TorchFTAdapter(), W1, trace)

    assert aegis_result.total_dollars_wasted < torchft_result.total_dollars_wasted, (
        f"AEGIS (${aegis_result.total_dollars_wasted:.4f}) must beat "
        f"TorchFT (${torchft_result.total_dollars_wasted:.4f}) on burst trace"
    )
