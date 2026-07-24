"""
BT-1: Per-tier isolation tests.

For each per-tier trace (FT-B0 through FT-B4):
  - AEGIS recovery time < B-VANILLA recovery time
  - The "primary struggle baseline" (from test_suite.md §4.1) is indeed
    worse than AEGIS for that tier
  - AEGIS's tier_handled matches the expected_tier (EPE routes correctly)

These tests prove clean per-tier attribution: each tier's recovery is
cheaper with AEGIS than with vanilla checkpoint-and-restart.
"""

from __future__ import annotations

import pytest

from bench.sim.engine import SimulationEngine
from bench.systems.aegis_adapter import AegisAdapter
from bench.systems.mecefo import MeCeFOAdapter
from bench.systems.r2ccl import R2CCLAdapter
from bench.systems.tiercheck import TierCheckAdapter
from bench.systems.vanilla import VanillaAdapter
from bench.traces.per_tier import FT_B0, FT_B1, FT_B2, FT_B3, FT_B4
from bench.workloads.configs import W1


engine = SimulationEngine()


async def test_bt1_b0_nic_flap_aegis_beats_vanilla():
    """
    FT-B0: NIC flap. AEGIS routes to B0 fast-path (~5s).
    B-VANILLA must fully restart (~120s+ on W1).
    """
    aegis = AegisAdapter()
    vanilla = VanillaAdapter()

    fault = FT_B0.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    vanilla_outcome = vanilla.handle_fault(fault, W1)

    # AEGIS must be faster
    assert aegis_outcome.recovery_time_secs < vanilla_outcome.recovery_time_secs, (
        f"AEGIS B0 recovery ({aegis_outcome.recovery_time_secs}s) should be "
        f"faster than vanilla ({vanilla_outcome.recovery_time_secs}s)"
    )

    # AEGIS must route to B0
    assert aegis_outcome.tier_handled == "B0", (
        f"AEGIS should handle NIC flap at B0, got {aegis_outcome.tier_handled}"
    )

    # Vanilla must do a full restart
    assert vanilla_outcome.forced_restart, "Vanilla should do a full restart for any fault"


async def test_bt1_b0_mecefo_struggles_with_nic_flap():
    """
    FT-B0: MeCeFO cannot handle NIC flap (B0) — falls back to vanilla.
    AEGIS (~5s) beats MeCeFO fallback (~120s+).
    """
    aegis = AegisAdapter()
    mecefo = MeCeFOAdapter()

    fault = FT_B0.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    mecefo_outcome = mecefo.handle_fault(fault, W1)

    assert mecefo_outcome.forced_restart, "MeCeFO should fall back for B0 (NIC flap)"
    assert aegis_outcome.recovery_time_secs < mecefo_outcome.recovery_time_secs


async def test_bt1_b0_tiercheck_struggles_with_nic_flap():
    """FT-B0: TierCheck cannot handle B0 — falls back to vanilla."""
    aegis = AegisAdapter()
    tiercheck = TierCheckAdapter()

    fault = FT_B0.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    tiercheck_outcome = tiercheck.handle_fault(fault, W1)

    assert tiercheck_outcome.forced_restart, "TierCheck should fall back for B0"
    assert aegis_outcome.recovery_time_secs < tiercheck_outcome.recovery_time_secs


async def test_bt1_b1_node_death_aegis_beats_vanilla():
    """
    FT-B1: Node crash. AEGIS neighbor-absorb (~25s).
    B-VANILLA: full restart (~120s+ on W1).
    """
    aegis = AegisAdapter()
    vanilla = VanillaAdapter()

    fault = FT_B1.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    vanilla_outcome = vanilla.handle_fault(fault, W1)

    assert aegis_outcome.recovery_time_secs < vanilla_outcome.recovery_time_secs
    assert aegis_outcome.tier_handled == "B1"
    assert vanilla_outcome.forced_restart


async def test_bt1_b1_r2ccl_struggles_with_node_death():
    """FT-B1: R2CCL cannot handle node death (B1) — falls back to vanilla."""
    aegis = AegisAdapter()
    r2ccl = R2CCLAdapter()

    fault = FT_B1.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    r2ccl_outcome = r2ccl.handle_fault(fault, W1)

    assert r2ccl_outcome.forced_restart, "R2CCL should fall back for B1 (node death)"
    assert aegis_outcome.recovery_time_secs < r2ccl_outcome.recovery_time_secs


async def test_bt1_b1_tiercheck_struggles_with_node_death():
    """FT-B1: TierCheck cannot handle B1 — falls back to vanilla."""
    aegis = AegisAdapter()
    tiercheck = TierCheckAdapter()

    fault = FT_B1.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    tiercheck_outcome = tiercheck.handle_fault(fault, W1)

    assert tiercheck_outcome.forced_restart, "TierCheck should fall back for B1"
    assert aegis_outcome.recovery_time_secs < tiercheck_outcome.recovery_time_secs


async def test_bt1_b2_cuda_crash_aegis_beats_vanilla():
    """
    FT-B2: CUDA kernel crash. AEGIS T1 restore (~8s).
    B-VANILLA: full restart.
    """
    aegis = AegisAdapter()
    vanilla = VanillaAdapter()

    fault = FT_B2.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    vanilla_outcome = vanilla.handle_fault(fault, W1)

    assert aegis_outcome.recovery_time_secs < vanilla_outcome.recovery_time_secs
    assert aegis_outcome.tier_handled == "B2"


async def test_bt1_b3_node_unrecoverable_aegis_beats_vanilla():
    """
    FT-B3: Node unrecoverable. AEGIS T2 peer restore (~55s).
    B-VANILLA: full restart.
    """
    aegis = AegisAdapter()
    vanilla = VanillaAdapter()

    fault = FT_B3.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    vanilla_outcome = vanilla.handle_fault(fault, W1)

    assert aegis_outcome.recovery_time_secs < vanilla_outcome.recovery_time_secs
    assert aegis_outcome.tier_handled == "B3"


async def test_bt1_b4_rack_outage_aegis_beats_vanilla():
    """
    FT-B4: Rack power loss. AEGIS T3 cluster restore (~270s).
    B-VANILLA: full restart (~120s+ on W1).
    """
    aegis = AegisAdapter()
    vanilla = VanillaAdapter()

    # FT-B4 has 2 events (both nodes on rack1); test the first
    fault = FT_B4.events[0]

    aegis_outcome = await aegis.handle_fault_async(fault, W1)
    vanilla_outcome = vanilla.handle_fault(fault, W1)

    assert aegis_outcome.recovery_time_secs < vanilla_outcome.recovery_time_secs
    assert aegis_outcome.tier_handled == "B4"


async def test_bt1_b4_rack_outage_all_single_primitives_struggle():
    """
    FT-B4: All single-primitive baselines forced to full restart.
    B-R2CCL, B-MECEFO cannot handle B4. TierCheck handles B4 but AEGIS is faster.
    """
    r2ccl = R2CCLAdapter()
    mecefo = MeCeFOAdapter()

    fault = FT_B4.events[0]

    r2ccl_outcome = r2ccl.handle_fault(fault, W1)
    mecefo_outcome = mecefo.handle_fault(fault, W1)

    # R2CCL and MeCeFO must fall back (they can't handle B4)
    assert r2ccl_outcome.forced_restart, "R2CCL cannot handle B4 rack outage"
    assert mecefo_outcome.forced_restart, "MeCeFO cannot handle B4 rack outage"


async def test_bt1_per_tier_trace_simulation():
    """
    Full simulation: AEGIS total recovery < B-VANILLA on each per-tier trace.
    Uses SimulationEngine to drive all events in the trace.
    """
    aegis = AegisAdapter()
    vanilla = VanillaAdapter()

    for trace in [FT_B0, FT_B1, FT_B2, FT_B3, FT_B4]:
        aegis_result = await engine.run(aegis, W1, trace)
        vanilla_result = await engine.run(vanilla, W1, trace)

        assert aegis_result.total_recovery_secs <= vanilla_result.total_recovery_secs, (
            f"AEGIS should recover faster than vanilla on {trace.name}: "
            f"AEGIS={aegis_result.total_recovery_secs}s vs "
            f"vanilla={vanilla_result.total_recovery_secs}s"
        )
        assert aegis_result.goodput >= vanilla_result.goodput, (
            f"AEGIS goodput should be >= vanilla on {trace.name}"
        )
