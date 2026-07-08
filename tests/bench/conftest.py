"""
Bench-specific pytest fixtures.

Provides workload configs and system adapter instances for benchmark tests.
All fixtures are function-scoped (the default) to keep tests isolated.
"""

from __future__ import annotations

import pytest

from bench.systems.aegis_adapter import AegisAdapter
from bench.systems.mecefo import MeCeFOAdapter
from bench.systems.r2ccl import R2CCLAdapter
from bench.systems.tiercheck import TierCheckAdapter
from bench.systems.torchft import TorchFTAdapter
from bench.systems.vanilla import VanillaAdapter
from bench.workloads.configs import W1, W2, W3, WorkloadConfig


@pytest.fixture
def workload_w1() -> WorkloadConfig:
    """W1: LLaMA-7B on 8 GPUs — the trust-anchor workload."""
    return W1


@pytest.fixture
def workload_w2() -> WorkloadConfig:
    """W2: LLaMA-13B on 32 GPUs — mid-scale workload."""
    return W2


@pytest.fixture
def workload_w3() -> WorkloadConfig:
    """W3: LLaMA-40B on 64 GPUs — checkpoint-heavy workload."""
    return W3


@pytest.fixture
def aegis_adapter() -> AegisAdapter:
    """AegisAdapter with default correlation policy."""
    return AegisAdapter()


@pytest.fixture
def vanilla_adapter() -> VanillaAdapter:
    """VanillaAdapter: checkpoint-and-restart baseline."""
    return VanillaAdapter()


@pytest.fixture
def r2ccl_adapter() -> R2CCLAdapter:
    """R2CCLAdapter: transport-only baseline."""
    return R2CCLAdapter()


@pytest.fixture
def mecefo_adapter() -> MeCeFOAdapter:
    """MeCeFOAdapter: compute-only baseline."""
    return MeCeFOAdapter()


@pytest.fixture
def tiercheck_adapter() -> TierCheckAdapter:
    """TierCheckAdapter: storage-only baseline."""
    return TierCheckAdapter()


@pytest.fixture
def torchft_adapter() -> TorchFTAdapter:
    """TorchFTAdapter: elastic training baseline."""
    return TorchFTAdapter()


@pytest.fixture
def all_systems(
    aegis_adapter: AegisAdapter,
    vanilla_adapter: VanillaAdapter,
    r2ccl_adapter: R2CCLAdapter,
    mecefo_adapter: MeCeFOAdapter,
    tiercheck_adapter: TierCheckAdapter,
    torchft_adapter: TorchFTAdapter,
):
    """All system adapters for a full comparison run."""
    return [
        aegis_adapter,
        vanilla_adapter,
        r2ccl_adapter,
        mecefo_adapter,
        tiercheck_adapter,
        torchft_adapter,
    ]
