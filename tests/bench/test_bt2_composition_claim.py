"""
BT-2: Composition claim tests.

The central product claim: AEGIS beats each single primitive on the fault class
that primitive cannot handle. If a single-layer tool can only handle its own tier,
it must fall back to vanilla for all others — and vanilla is always more expensive
than AEGIS's tier-optimized routing.

These tests are the most important in the whole suite.
"""

from __future__ import annotations

import pytest

from bench.sim.engine import SimulationEngine
from bench.systems.aegis_adapter import AegisAdapter
from bench.systems.mecefo import MeCeFOAdapter
from bench.systems.r2ccl import R2CCLAdapter
from bench.systems.tiercheck import TierCheckAdapter
from bench.systems.torchft import TorchFTAdapter
from bench.systems.vanilla import VanillaAdapter
from bench.traces.per_tier import FT_B0, FT_B1, FT_B2, FT_B3, FT_B4
from bench.workloads.configs import W1


engine = SimulationEngine()


async def test_bt2_aegis_beats_r2ccl_on_node_death():
    """
    B-R2CCL can only handle B0 (NIC flap). On B1 (node death), it falls back
    to full checkpoint-and-restart.

    AEGIS routes to neighbor-absorb (~25s) vs R2CCL forced restart (~120s+ on W1).
    """
    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()

    # Use SimulationEngine for the FT-B1 trace
    aegis_result = await engine.run(aegis, W1, FT_B1)
    r2ccl_result = await engine.run(r2ccl, W1, FT_B1)

    aegis_cost = aegis_result.total_dollars_wasted
    r2ccl_cost = r2ccl_result.total_dollars_wasted

    assert aegis_cost < r2ccl_cost, (
        f"AEGIS (${aegis_cost:.4f}) should cost less than R2CCL (${r2ccl_cost:.4f}) "
        f"on B1 node death"
    )

    # Verify R2CCL was forced to restart
    assert all(o.forced_restart for o in r2ccl_result.fault_outcomes), (
        "R2CCL should be forced to full restart on B1 faults"
    )


async def test_bt2_aegis_beats_mecefo_on_nic_flap():
    """
    B-MECEFO can only handle B1 (node death). On B0 (NIC flap), it falls back
    to full checkpoint-and-restart.

    AEGIS routes to B0 fast-path (~5s) vs MeCeFO forced restart (~120s+ on W1).
    """
    aegis = AegisAdapter()
    mecefo = MeCeFOAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B0)
    mecefo_result = await engine.run(mecefo, W1, FT_B0)

    aegis_cost = aegis_result.total_dollars_wasted
    mecefo_cost = mecefo_result.total_dollars_wasted

    assert aegis_cost < mecefo_cost, (
        f"AEGIS (${aegis_cost:.4f}) should cost less than MeCeFO (${mecefo_cost:.4f}) "
        f"on B0 NIC flap"
    )

    assert all(o.forced_restart for o in mecefo_result.fault_outcomes), (
        "MeCeFO should be forced to full restart on B0 faults"
    )


async def test_bt2_aegis_beats_tiercheck_on_node_death():
    """
    B-TIERCHECK can only handle B2/B3/B4. On B1 (node death), it falls back
    to full checkpoint-and-restart.

    AEGIS routes to neighbor-absorb (~25s) vs TierCheck forced restart (~120s+ on W1).
    """
    aegis = AegisAdapter()
    tiercheck = TierCheckAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B1)
    tiercheck_result = await engine.run(tiercheck, W1, FT_B1)

    aegis_cost = aegis_result.total_dollars_wasted
    tiercheck_cost = tiercheck_result.total_dollars_wasted

    assert aegis_cost < tiercheck_cost, (
        f"AEGIS (${aegis_cost:.4f}) should cost less than TierCheck "
        f"(${tiercheck_cost:.4f}) on B1 node death"
    )

    assert all(o.forced_restart for o in tiercheck_result.fault_outcomes), (
        "TierCheck should be forced to full restart on B1 faults"
    )


async def test_bt2_aegis_beats_r2ccl_on_cuda_crash():
    """B-R2CCL cannot handle B2 (CUDA crash). AEGIS T1 restore wins."""
    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B2)
    r2ccl_result = await engine.run(r2ccl, W1, FT_B2)

    assert aegis_result.total_dollars_wasted < r2ccl_result.total_dollars_wasted


async def test_bt2_aegis_beats_mecefo_on_cuda_crash():
    """B-MECEFO cannot handle B2 (CUDA crash). AEGIS T1 restore wins."""
    aegis = AegisAdapter()
    mecefo = MeCeFOAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B2)
    mecefo_result = await engine.run(mecefo, W1, FT_B2)

    assert aegis_result.total_dollars_wasted < mecefo_result.total_dollars_wasted


async def test_bt2_aegis_beats_r2ccl_on_node_unrecoverable():
    """B-R2CCL cannot handle B3. AEGIS T2 peer restore wins."""
    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B3)
    r2ccl_result = await engine.run(r2ccl, W1, FT_B3)

    assert aegis_result.total_dollars_wasted < r2ccl_result.total_dollars_wasted


async def test_bt2_aegis_beats_mecefo_on_node_unrecoverable():
    """B-MECEFO cannot handle B3. AEGIS T2 peer restore wins."""
    aegis = AegisAdapter()
    mecefo = MeCeFOAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B3)
    mecefo_result = await engine.run(mecefo, W1, FT_B3)

    assert aegis_result.total_dollars_wasted < mecefo_result.total_dollars_wasted


async def test_bt2_aegis_beats_all_on_rack_outage():
    """
    FT-B4: Rack power loss.
    B-R2CCL, B-MECEFO cannot handle B4 — forced restart.
    AEGIS T3 cluster restore (~270s) beats forced restarts.
    """
    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()
    mecefo = MeCeFOAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B4)
    r2ccl_result = await engine.run(r2ccl, W1, FT_B4)
    mecefo_result = await engine.run(mecefo, W1, FT_B4)

    assert aegis_result.total_dollars_wasted < r2ccl_result.total_dollars_wasted, (
        "AEGIS should beat R2CCL on B4 rack outage"
    )
    assert aegis_result.total_dollars_wasted < mecefo_result.total_dollars_wasted, (
        "AEGIS should beat MeCeFO on B4 rack outage"
    )


async def test_bt2_composition_superiority_across_all_tiers():
    """
    The key composition claim: on a mixed multi-tier trace, AEGIS total
    dollars_wasted < each single-primitive baseline.

    This tests the thesis: no single paper is a product, but composed they are.
    Uses all per-tier traces combined to simulate a mixed fault scenario.
    """
    from bench.traces.per_tier import FaultTrace

    # Combine all per-tier events into one mixed trace
    all_events = (
        FT_B0.events
        + FT_B1.events
        + FT_B2.events
        + FT_B3.events
        + FT_B4.events
    )
    mixed_trace = FaultTrace(name="FT-MIXED-ALL", events=all_events)

    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()
    mecefo = MeCeFOAdapter()
    tiercheck = TierCheckAdapter()

    aegis_result = await engine.run(aegis, W1, mixed_trace)
    r2ccl_result = await engine.run(r2ccl, W1, mixed_trace)
    mecefo_result = await engine.run(mecefo, W1, mixed_trace)
    tiercheck_result = await engine.run(tiercheck, W1, mixed_trace)

    aegis_cost = aegis_result.total_dollars_wasted

    assert aegis_cost < r2ccl_result.total_dollars_wasted, (
        f"AEGIS (${aegis_cost:.4f}) must beat R2CCL "
        f"(${r2ccl_result.total_dollars_wasted:.4f}) across all tiers"
    )
    assert aegis_cost < mecefo_result.total_dollars_wasted, (
        f"AEGIS (${aegis_cost:.4f}) must beat MeCeFO "
        f"(${mecefo_result.total_dollars_wasted:.4f}) across all tiers"
    )
    assert aegis_cost < tiercheck_result.total_dollars_wasted, (
        f"AEGIS (${aegis_cost:.4f}) must beat TierCheck "
        f"(${tiercheck_result.total_dollars_wasted:.4f}) across all tiers"
    )


async def test_bt2_aegis_beats_torchft_on_nic_flap():
    """
    AEGIS B0 fast-path (~5s) beats TorchFT's elastic-training B0 recovery (~60s).
    TorchFT handles all tiers but more slowly.
    """
    aegis = AegisAdapter()
    torchft = TorchFTAdapter()

    aegis_result = await engine.run(aegis, W1, FT_B0)
    torchft_result = await engine.run(torchft, W1, FT_B0)

    assert aegis_result.total_dollars_wasted < torchft_result.total_dollars_wasted, (
        "AEGIS should beat TorchFT on B0 — TorchFT takes 60s vs AEGIS 5s"
    )


async def test_bt2_goodput_aegis_best_on_mixed_trace():
    """
    AEGIS achieves the highest goodput on a mixed trace, confirming that
    composition superiority translates to more productive training time.
    """
    from bench.traces.per_tier import FaultTrace

    all_events = (
        FT_B0.events
        + FT_B1.events
        + FT_B2.events
        + FT_B3.events
        + FT_B4.events
    )
    mixed_trace = FaultTrace(name="FT-MIXED-GOODPUT", events=all_events)

    adapters = {
        "AEGIS": AegisAdapter(),
        "B-VANILLA": VanillaAdapter(),
        "B-R2CCL": R2CCLAdapter(),
        "B-MECEFO": MeCeFOAdapter(),
        "B-TIERCHECK": TierCheckAdapter(),
    }

    results = {}
    for name, adapter in adapters.items():
        results[name] = await engine.run(adapter, W1, mixed_trace)

    aegis_goodput = results["AEGIS"].goodput
    for name, result in results.items():
        if name == "AEGIS":
            continue
        assert aegis_goodput >= result.goodput, (
            f"AEGIS goodput ({aegis_goodput:.4f}) should be >= {name} ({result.goodput:.4f})"
        )
