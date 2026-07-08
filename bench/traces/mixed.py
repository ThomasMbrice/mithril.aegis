"""
Mixed fault traces for realistic end-to-end evaluation.

These drive the $/GPU-hr hero number. Faults arrive as they do in
production: mixed classes, correlated bursts, drawn from realistic
distributions.

All traces are seeded for deterministic, reproducible replay.
"""

from __future__ import annotations

import random

from aegis.telemetry.events import BlastRadius, FaultSignal

from .per_tier import FaultTrace, TraceFaultEvent


def make_ft_poisson(
    *,
    seed: int = 42,
    total_steps: int = 100_000,
    mtbf_steps: int = 20_000,
) -> FaultTrace:
    """
    FT-POISSON: independent failures at Poisson-distributed intervals.

    Models independent hardware failures drawn from an exponential
    distribution parameterised by mean time between failures (MTBF).

    Fault types are drawn uniformly from B0/B1/B2/B3/B4 signals to give
    a realistic mix of transport, compute, and storage events.

    Args:
        seed: RNG seed for determinism.
        total_steps: Total training steps to simulate.
        mtbf_steps: Mean steps between failures (Poisson rate = 1/MTBF).

    Returns:
        FaultTrace with randomly placed faults.
    """
    rng = random.Random(seed)

    # Fault pool: (signal, expected_tier, node, nic, rack)
    fault_pool = [
        (FaultSignal.NIC_PORT_FLAP, BlastRadius.B0, "node0", "nic0", "rack0"),
        (FaultSignal.LINK_FLUCTUATION, BlastRadius.B0, "node1", "nic1", "rack0"),
        (FaultSignal.NODE_CRASH, BlastRadius.B1, "node1", None, "rack0"),
        (FaultSignal.GPU_FELL_OFF_BUS, BlastRadius.B1, "node0", None, "rack0"),
        (FaultSignal.CUDA_KERNEL_CRASH, BlastRadius.B2, "node2", None, "rack1"),
        (FaultSignal.HUNG_RANK, BlastRadius.B2, "node3", None, "rack1"),
        (FaultSignal.NODE_UNRECOVERABLE, BlastRadius.B3, "node2", None, "rack1"),
        (FaultSignal.RACK_POWER_LOSS, BlastRadius.B4, "node3", None, "rack1"),
    ]

    events: list[TraceFaultEvent] = []
    step = rng.expovariate(1.0 / mtbf_steps)  # first fault

    while step < total_steps:
        step_int = int(step)
        signal, tier, node, nic, rack = rng.choice(fault_pool)
        rank = rng.randint(0, 7)

        events.append(
            TraceFaultEvent(
                step=step_int,
                fault_signal=signal,
                target_rank=rank,
                target_node=node,
                target_nic=nic,
                target_rack=rack,
                expected_tier=tier,
            )
        )

        # Next fault: exponentially distributed inter-arrival
        step += rng.expovariate(1.0 / mtbf_steps)

    return FaultTrace(name="FT-POISSON", events=events)


def make_ft_burst(*, seed: int = 42) -> FaultTrace:
    """
    FT-BURST: correlated rack-event bursts (B1s leading into a B4).

    This is where AEGIS should win biggest — it tests the correlation
    window logic that re-classifies a B1 burst to B4 before committing
    to the expensive neighbor-absorb path (§3.3).

    A cascade of B1 node deaths on the same rack within a short window,
    then a rack-level B4 event follows, matching a realistic cascade failure.

    Args:
        seed: RNG seed for determinism.

    Returns:
        FaultTrace with correlated burst events.
    """
    rng = random.Random(seed)

    events: list[TraceFaultEvent] = []

    # Phase 1: Isolated B0 event (early warning, different rack)
    events.append(
        TraceFaultEvent(
            step=3000,
            fault_signal=FaultSignal.NIC_PORT_FLAP,
            target_rank=0,
            target_node="node0",
            target_nic="nic0",
            target_rack="rack0",
            expected_tier=BlastRadius.B0,
        )
    )

    # Phase 2: Correlated B1 burst on rack1 (3 nodes in quick succession)
    # These arrive close together to trigger the correlation window
    burst_nodes = [
        ("node2", 4, "rack1"),
        ("node3", 6, "rack1"),
        ("node2", 5, "rack1"),  # second rank on same node
    ]
    for i, (node, rank, rack) in enumerate(burst_nodes):
        events.append(
            TraceFaultEvent(
                step=8000 + i * 2,  # steps 8000, 8002, 8004
                fault_signal=FaultSignal.NODE_CRASH,
                target_rank=rank,
                target_node=node,
                target_nic=None,
                target_rack=rack,
                expected_tier=BlastRadius.B1,  # individual classification
            )
        )

    # Phase 3: Full rack B4 event (rack1 taken out)
    events.append(
        TraceFaultEvent(
            step=15000,
            fault_signal=FaultSignal.RACK_POWER_LOSS,
            target_rank=4,
            target_node="node2",
            target_nic=None,
            target_rack="rack1",
            expected_tier=BlastRadius.B4,
        )
    )

    return FaultTrace(name="FT-BURST", events=events)


def make_ft_production(*, seed: int = 2024) -> FaultTrace:
    """
    FT-PRODUCTION: realistic large-cluster failure profile.

    Replays a realistic mix of failures observed in production LLM training
    clusters: mostly transient B0/B1 events with occasional B2/B3 events
    and rare B4 rack outages. This is the trace the $/GPU-hr headline is
    computed on.

    Args:
        seed: RNG seed for determinism.

    Returns:
        FaultTrace matching a realistic production failure profile.
    """
    rng = random.Random(seed)

    events: list[TraceFaultEvent] = []

    # Production failure profile (based on published large-cluster data):
    # ~70% B0 (transient NIC), ~20% B1 (node crashes),
    # ~5% B2 (software), ~4% B3 (hw replacement), ~1% B4 (rack outage)
    # MTBF: ~15000 steps between faults
    fault_profile = [
        # (weight, signal, tier, node, nic, rack)
        (35, FaultSignal.NIC_PORT_FLAP, BlastRadius.B0, "node0", "nic0", "rack0"),
        (20, FaultSignal.LINK_FLUCTUATION, BlastRadius.B0, "node1", "nic1", "rack0"),
        (15, FaultSignal.RDMA_TIMEOUT, BlastRadius.B0, "node2", "nic0", "rack1"),
        (12, FaultSignal.NODE_CRASH, BlastRadius.B1, "node1", None, "rack0"),
        (8, FaultSignal.GPU_FELL_OFF_BUS, BlastRadius.B1, "node0", None, "rack0"),
        (4, FaultSignal.CUDA_KERNEL_CRASH, BlastRadius.B2, "node2", None, "rack1"),
        (3, FaultSignal.HUNG_RANK, BlastRadius.B2, "node3", None, "rack1"),
        (2, FaultSignal.NODE_UNRECOVERABLE, BlastRadius.B3, "node2", None, "rack1"),
        (1, FaultSignal.RACK_POWER_LOSS, BlastRadius.B4, "node3", None, "rack1"),
    ]

    weights = [w for w, *_ in fault_profile]
    fault_specs = [rest for _, *rest in fault_profile]

    total_steps = 200_000
    mtbf_steps = 15_000

    step = rng.expovariate(1.0 / mtbf_steps)

    while step < total_steps:
        step_int = int(step)
        (signal, tier, node, nic, rack) = rng.choices(fault_specs, weights=weights, k=1)[0]
        rank = rng.randint(0, 7)

        events.append(
            TraceFaultEvent(
                step=step_int,
                fault_signal=signal,
                target_rank=rank,
                target_node=node,
                target_nic=nic,
                target_rack=rack,
                expected_tier=tier,
            )
        )

        step += rng.expovariate(1.0 / mtbf_steps)

    return FaultTrace(name="FT-PRODUCTION", events=events)
