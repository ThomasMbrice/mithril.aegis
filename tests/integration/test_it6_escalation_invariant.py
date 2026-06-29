"""
IT-6 — Escalation invariant property test (fuzz).

Scenario: inject a large randomised set of faults covering all signal types,
nodes, and racks with varying inter-fault delays.
Pass criterion (§5.2):
  - The one-directional escalation invariant (final_tier >= original_tier)
    MUST hold for every single processed fault record, no exceptions.
  - Specifically: no path ever de-escalates mid-fault.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness


# All real fault signals (exclude UNKNOWN — it's a catch-all, not a real fault)
_ALL_SIGNALS = [s for s in FaultSignal if s != FaultSignal.UNKNOWN]
_NODES = [f"node{i}" for i in range(4)]
_RACKS = ["rack0", "rack1"]

_NIC_SIGNALS = frozenset({
    FaultSignal.NIC_PORT_FLAP,
    FaultSignal.LINK_FLUCTUATION,
    FaultSignal.RDMA_TIMEOUT,
})


@pytest.fixture
async def fuzz_runtime() -> AegisRuntime:
    policy = OperatorPolicy(
        correlation_window_secs=0.3,
        correlation_node_threshold=3,
    )
    rt = AegisRuntime(policy=policy)
    for node in _NODES:
        rt.transport.register_node_nics(node, ["nic0", "nic1"])
    for i, node in enumerate(_NODES):
        rt.compute.register_neighbor(node, _NODES[(i + 1) % len(_NODES)])
    for node in _NODES:
        rt.storage.write_tier1(node, epoch=0)
        rt.storage.write_tier2(node, epoch=0)
    rt.storage.write_tier3(epoch=0)
    async with rt:
        yield rt


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_escalation_invariant_never_violated_under_fuzz(
    fuzz_runtime: AegisRuntime,
) -> None:
    """
    Property test: inject 60 randomised faults, assert no de-escalation.

    This is the primary guard against regressions to the single most important
    design invariant in AEGIS (§1 design invariant, §5.2 IT-6).
    """
    rng = random.Random(0xAE615)
    N = 60
    harness = ChaosHarness(fuzz_runtime.utp, fuzz_runtime.epoch_service)

    for _ in range(N):
        signal = rng.choice(_ALL_SIGNALS)
        node = rng.choice(_NODES)
        rack = rng.choice(_RACKS)
        nic_id = "nic0" if signal in _NIC_SIGNALS else None

        await harness.inject(FaultSpec(
            fault_signal=signal,
            rank=rng.randint(0, 7),
            node=node,
            rack_id=rack,
            nic_id=nic_id,
        ))

    await _settle(fuzz_runtime, expected=N)

    audit = fuzz_runtime.epe.escalation_audit()
    violations = [r for r in audit if not r["escalation_valid"]]

    assert not violations, (
        f"Escalation invariant violated {len(violations)} time(s):\n"
        + "\n".join(
            f"  epoch={v['epoch']} signal={v['signal']} "
            f"{v['original_tier']} → {v['final_tier']}"
            for v in violations[:10]
        )
    )


async def test_all_faults_are_processed(fuzz_runtime: AegisRuntime) -> None:
    """Every injected fault must produce exactly one audit record."""
    rng = random.Random(0xC0FFEE)
    N = 30
    harness = ChaosHarness(fuzz_runtime.utp, fuzz_runtime.epoch_service)

    for _ in range(N):
        signal = rng.choice(_ALL_SIGNALS)
        node = rng.choice(_NODES)
        await harness.inject(FaultSpec(
            fault_signal=signal, rank=0, node=node,
        ))

    await _settle(fuzz_runtime, expected=N)
    assert len(fuzz_runtime.epe.history) == N, (
        f"Expected {N} audit records, got {len(fuzz_runtime.epe.history)}"
    )


async def test_epoch_strictly_monotonic_across_fuzz(fuzz_runtime: AegisRuntime) -> None:
    """Epoch values across all processed faults must be strictly increasing."""
    rng = random.Random(0xDEAD)
    N = 40
    harness = ChaosHarness(fuzz_runtime.utp, fuzz_runtime.epoch_service)

    for _ in range(N):
        signal = rng.choice(_ALL_SIGNALS)
        await harness.inject(FaultSpec(
            fault_signal=signal, rank=0, node=rng.choice(_NODES),
        ))

    await _settle(fuzz_runtime, expected=N)

    epochs = [r.epoch for r in fuzz_runtime.epe.history]
    for i in range(1, len(epochs)):
        assert epochs[i] > epochs[i - 1], (
            f"Epoch not monotonic at index {i}: {epochs[i-1]} → {epochs[i]}"
        )
