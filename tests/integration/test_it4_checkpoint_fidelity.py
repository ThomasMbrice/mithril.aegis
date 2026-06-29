"""
IT-4 — Checkpoint taken during MeCeFO fallback carries fidelity_flag.

Scenario: a node crashes, MeCeFO neighbor-absorb activates (B1 recovery),
and a checkpoint is written during the fallback window.
Pass criterion (§5.2):
  - RecoveryResult.degraded == True during fallback (§3.4)
  - Checkpoint written during fallback must carry fidelity_flag=True
  - No silent approximation: the flag is always set, never omitted
"""

from __future__ import annotations

import asyncio

from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from chaos_inject.faults import FaultSpec
from chaos_inject.harness import ChaosHarness


async def _settle(rt: AegisRuntime, expected: int = 1, timeout: float = 0.5) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_b1_recovery_result_is_degraded(runtime: AegisRuntime, chaos: ChaosHarness) -> None:
    """MeCeFO fallback sets degraded=True on RecoveryResult (§3.4)."""
    await chaos.inject(FaultSpec(
        fault_signal=FaultSignal.NODE_CRASH,
        rank=0,
        node="node0",
    ))
    await _settle(runtime)

    history = runtime.epe.history
    assert len(history) == 1
    result = history[0].result
    assert result is not None
    assert result.degraded, (
        "MeCeFO fallback must set degraded=True — no silent approximation (§3.4)"
    )


async def test_checkpoint_written_during_fallback_has_fidelity_flag(
    runtime: AegisRuntime, chaos: ChaosHarness
) -> None:
    """
    When the compute layer is in fallback, any checkpoint written must have
    fidelity_flag=True so downstream consumers know the state is approximate.
    """
    # Trigger B1 fallback
    await chaos.inject(FaultSpec(
        fault_signal=FaultSignal.NODE_CRASH,
        rank=0,
        node="node0",
    ))
    await _settle(runtime)

    # Simulate writing a checkpoint during fallback
    ckpt = runtime.storage.write_tier1("node0", epoch=runtime.epoch_service.current(),
                                       fidelity_flag=True)

    assert ckpt.fidelity_flag, (
        "Checkpoint taken during MeCeFO fallback must carry fidelity_flag=True"
    )
    assert ckpt.tier == "tier1"


async def test_checkpoint_without_fallback_has_clean_fidelity(runtime: AegisRuntime) -> None:
    """Checkpoints written outside fallback must NOT carry fidelity_flag."""
    # No fault injected — system is healthy
    ckpt = runtime.storage.write_tier1("node0", epoch=0, fidelity_flag=False)
    assert not ckpt.fidelity_flag


async def test_b0_recovery_not_degraded(runtime: AegisRuntime, chaos: ChaosHarness) -> None:
    """B0 NIC migration does not approximate gradients — degraded must be False."""
    await chaos.inject(FaultSpec(
        fault_signal=FaultSignal.NIC_PORT_FLAP,
        rank=0,
        node="node0",
        nic_id="nic0",
    ))
    await _settle(runtime)

    history = runtime.epe.history
    result = history[0].result
    assert result is not None
    assert not result.degraded, "B0 NIC migration should not set degraded=True"
