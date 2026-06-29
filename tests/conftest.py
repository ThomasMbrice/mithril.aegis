"""
Shared pytest fixtures for AEGIS tests.

All fixtures are function-scoped (the default with asyncio_mode = "auto")
so each test gets a fresh event loop and clean component state.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from chaos_inject.harness import ChaosHarness


def _configure_runtime(rt: AegisRuntime) -> None:
    """Populate a runtime with a minimal 4-node, 2-rack topology."""
    nodes = ["node0", "node1", "node2", "node3"]
    racks = ["rack0", "rack0", "rack1", "rack1"]

    for node in nodes:
        rt.transport.register_node_nics(node, ["nic0", "nic1"])

    for i, node in enumerate(nodes):
        neighbor = nodes[(i + 1) % len(nodes)]
        rt.compute.register_neighbor(node, neighbor)

    for node in nodes:
        rt.storage.write_tier1(node, epoch=0)
        rt.storage.write_tier2(node, epoch=0)

    rt.storage.write_tier3(epoch=0)

    # Seed URC: all ranks start at epoch 0
    for i in range(len(nodes)):
        rt.consensus.report_epoch(rank=i, epoch=0)


@pytest.fixture
def default_policy() -> OperatorPolicy:
    return OperatorPolicy(
        correlation_window_secs=1.0,
        correlation_node_threshold=3,
    )


@pytest_asyncio.fixture
async def runtime(default_policy: OperatorPolicy) -> AegisRuntime:
    """Full AegisRuntime with a 4-node test topology, started and stopped."""
    rt = AegisRuntime(policy=default_policy)
    _configure_runtime(rt)
    async with rt:
        yield rt


@pytest_asyncio.fixture
async def chaos(runtime: AegisRuntime) -> ChaosHarness:
    """ChaosHarness wired to the shared runtime."""
    return ChaosHarness(runtime.utp, runtime.epoch_service)
