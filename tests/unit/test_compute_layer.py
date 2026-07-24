"""
UT-C — Real ComputeLayer (B1) MeCeFO math.

Not a reproduction of MeCeFO's 4.18%-throughput-drop / LLaMA-7B benchmark
(that needs the real 8×A100 training job, see design.md §8.1) — these
tests validate that the three fallback *mechanisms* (skip-connection,
selective recompute, low-rank approximation) are real tensor math with
measurable, bounded properties, not a canned sleep(0) stub.
"""

from __future__ import annotations

import torch

from aegis.layers.compute import (
    MAX_CONSECUTIVE_FALLBACKS,
    ComputeLayer,
    TorchMeCeFOBackend,
    low_rank_approximate,
    selective_recompute_ffn,
    skip_connection_gradient,
)
from aegis.telemetry.events import BlastRadius, FaultSignal, TelemetryEvent


def _event(node: str, rank: int = 0) -> TelemetryEvent:
    return TelemetryEvent(
        rank=rank, node=node, fault_signal=FaultSignal.NODE_CRASH,
        raw_payload={}, epoch=0,
    )


def test_low_rank_approximate_full_rank_is_near_exact():
    """A rank equal to the matrix dimension should reconstruct almost exactly."""
    grad = torch.randn(32, 32)
    _, rel_error = low_rank_approximate(grad, rank=32)
    assert rel_error < 1e-4


def test_low_rank_approximate_error_grows_as_rank_shrinks():
    """Real SVD math: less rank retained → more reconstruction error."""
    torch.manual_seed(0)
    grad = torch.randn(64, 64)
    _, err_high_rank = low_rank_approximate(grad, rank=48)
    _, err_low_rank = low_rank_approximate(grad, rank=4)
    assert err_low_rank > err_high_rank
    assert err_low_rank > 0.0


def test_skip_connection_is_identity_passthrough():
    """A skip connection passes the incoming gradient through unchanged."""
    grad_output = torch.randn(8, 8)
    result = skip_connection_gradient(grad_output)
    assert torch.equal(result, grad_output)
    assert result is not grad_output  # real clone, not aliasing


def test_selective_recompute_respects_keep_fraction():
    x = torch.randn(100, 16)
    w1 = torch.randn(16, 16)
    w2 = torch.randn(16, 16)
    out, actual_fraction = selective_recompute_ffn(x, w1, w2, keep_fraction=0.3)
    assert out.shape == (100, 16)
    assert 0.25 <= actual_fraction <= 0.35


def test_selective_recompute_full_keep_matches_exact_forward():
    """keep_fraction=1.0 → every row exactly recomputed, matching a plain FFN forward."""
    torch.manual_seed(1)
    x = torch.randn(20, 8)
    w1 = torch.randn(8, 8)
    w2 = torch.randn(8, 8)
    out, fraction = selective_recompute_ffn(x, w1, w2, keep_fraction=1.0)
    expected = torch.relu(x @ w1) @ w2
    assert fraction == 1.0
    assert torch.allclose(out, expected, atol=1e-5)


async def test_torch_backend_absorb_returns_real_measurements():
    backend = TorchMeCeFOBackend(proxy_dim=32)
    result = await backend.absorb("node0", "node1")
    assert result.success
    assert result.elapsed_secs > 0.0
    assert result.low_rank_relative_error > 0.0
    assert result.skip_connection_applied
    assert 0.0 <= result.recomputed_fraction <= 1.0
    assert result.device in ("cpu", "mps", "cuda")


async def test_compute_layer_recover_is_degraded_with_real_error_measurement():
    layer = ComputeLayer(backend=TorchMeCeFOBackend(proxy_dim=16))
    layer.register_neighbor("node0", "node1")

    result = await layer.recover(_event("node0"), BlastRadius.B1, epoch=1)

    assert result.success
    assert result.degraded  # §3.4 no silent approximation
    assert "low-rank rel-error" in result.message


async def test_compute_layer_bounds_consecutive_fallbacks():
    layer = ComputeLayer(backend=TorchMeCeFOBackend(proxy_dim=8))
    layer.register_neighbor("node0", "node1")

    for _ in range(MAX_CONSECUTIVE_FALLBACKS):
        result = await layer.recover(_event("node0"), BlastRadius.B1, epoch=1)
        assert result.success

    assert layer._consecutive_fallbacks["node0"] == MAX_CONSECUTIVE_FALLBACKS

    layer.exit_fallback("node0")
    assert "node0" not in layer._consecutive_fallbacks
