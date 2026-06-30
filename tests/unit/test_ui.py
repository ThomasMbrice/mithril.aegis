"""
Unit tests for the AEGIS public UI API (ui.md §2–§4).

Coverage:
  - aegis.init() — default, observe_only, disable=[], invalid hook, invalid mode
  - aegis.status() — shape and keys
  - aegis.explain() — no-fault message, observe_only flag
  - aegis.disable() — state cleared
  - aegis.policy.set() — known key works, unknown key raises
  - aegis.checkpoint — save/restore before init raises RuntimeError
  - aegis.transport — set_fast_path before/after init
  - aegis.config.load_policy — missing file, valid YAML
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import aegis
import aegis._state as _state
from aegis.config import load_policy
from aegis.policy.dsl import OperatorPolicy


# ---------------------------------------------------------------------------
# Autouse fixture: reset module state before each test so tests are isolated


@pytest.fixture(autouse=True)
def reset_aegis():
    """Ensure a clean AEGIS state before and after every test."""
    aegis._reset()
    yield
    aegis._reset()


# ---------------------------------------------------------------------------
# init() tests


def test_init_default():
    """init() with no args; status() should return expected shape."""
    aegis.init()
    s = aegis.status()
    assert s["initialized"] is True
    assert s["mode"] == "active"
    # All five hooks should be active
    assert set(s["active_hooks"]) == {"transport", "compute", "checkpoint", "telemetry", "policy"}
    assert isinstance(s["current_epoch"], int)
    assert "kpi" in s


def test_init_observe_only():
    """mode='observe_only' is recorded in status."""
    aegis.init(mode="observe_only")
    s = aegis.status()
    assert s["mode"] == "observe_only"
    # EPE should have the flag set
    assert _state.runtime is not None
    assert _state.runtime.epe._observe_only is True


def test_init_disable_hook():
    """Disabling 'checkpoint' removes it from active hooks."""
    aegis.init(disable=["checkpoint"])
    s = aegis.status()
    assert "checkpoint" not in s["active_hooks"]
    # Other hooks still active
    assert "transport" in s["active_hooks"]
    assert "compute" in s["active_hooks"]


def test_init_invalid_hook():
    """Unknown hook name in disable= raises ValueError immediately."""
    with pytest.raises(ValueError, match="Unknown hook"):
        aegis.init(disable=["nonexistent_hook"])


def test_init_invalid_mode():
    """Invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid mode"):
        aegis.init(mode="aggressive")


# ---------------------------------------------------------------------------
# status() tests


def test_status_keys():
    """status() has all expected keys."""
    aegis.init()
    s = aegis.status()
    expected_keys = {"initialized", "mode", "active_hooks", "current_epoch", "kpi"}
    assert expected_keys.issubset(s.keys())


def test_status_before_init_raises():
    """status() before init raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not initialized"):
        aegis.status()


# ---------------------------------------------------------------------------
# explain() tests


def test_explain_no_faults():
    """explain() before any faults returns the 'no faults' message."""
    aegis.init()
    result = aegis.explain()
    assert result["message"] == "No faults processed yet."


def test_explain_observe_only_flag():
    """explain() in observe_only mode sets observe_only=True."""
    aegis.init(mode="observe_only")
    result = aegis.explain()
    assert result["observe_only"] is True


def test_explain_active_mode_observe_only_false():
    """explain() in active mode has observe_only=False."""
    aegis.init(mode="active")
    result = aegis.explain()
    assert result["observe_only"] is False


def test_explain_before_init_raises():
    """explain() before init raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not initialized"):
        aegis.explain()


# ---------------------------------------------------------------------------
# disable() tests


def test_disable_clears_state():
    """After disable(), _state.initialized is False and runtime is None."""
    aegis.init()
    assert _state.initialized is True

    aegis.disable()

    assert _state.initialized is False
    assert _state.runtime is None
    assert _state.hooks is None


def test_disable_before_init_is_noop():
    """disable() before init does not raise."""
    aegis.disable()  # should be silent no-op


def test_disable_then_reinit():
    """After disable(), init() can be called again successfully."""
    aegis.init()
    aegis.disable()
    aegis.init()
    s = aegis.status()
    assert s["initialized"] is True


# ---------------------------------------------------------------------------
# policy.set() tests


def test_policy_set_known_key():
    """policy.set('economics.gpu_hourly_cost_usd', 3.0) updates the live policy."""
    aegis.init()
    aegis.policy.set("economics.gpu_hourly_cost_usd", 3.0)
    policy = _state.runtime.epe._policy  # type: ignore[union-attr]
    assert policy.gpu_hr_cost == pytest.approx(3.0)


def test_policy_set_economics_policy():
    """policy.set('economics.policy', ...) updates economics_policy."""
    aegis.init()
    aegis.policy.set("economics.policy", "correctness_first")
    policy = _state.runtime.epe._policy  # type: ignore[union-attr]
    assert policy.economics_policy == "correctness_first"


def test_policy_set_correlation_window_ms():
    """policy.set('escalation.correlation_window_ms', 3000) converts to secs."""
    aegis.init()
    aegis.policy.set("escalation.correlation_window_ms", 3000)
    policy = _state.runtime.epe._policy  # type: ignore[union-attr]
    assert policy.correlation_window_secs == pytest.approx(3.0)


def test_policy_set_unknown_key():
    """policy.set() with an unrecognised key raises ValueError."""
    aegis.init()
    with pytest.raises(ValueError, match="Unknown policy key"):
        aegis.policy.set("does.not.exist", 42)


def test_policy_set_before_init_raises():
    """policy.set() before init raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not been called"):
        aegis.policy.set("economics.gpu_hourly_cost_usd", 1.0)


# ---------------------------------------------------------------------------
# checkpoint tests


def test_checkpoint_before_init_raises():
    """checkpoint.save() before init raises RuntimeError with clear message."""
    with pytest.raises(RuntimeError, match="aegis.init\\(\\) has not been called"):
        aegis.checkpoint.save(None)


def test_checkpoint_restore_before_init_raises():
    """checkpoint.restore() before init raises RuntimeError."""
    with pytest.raises(RuntimeError, match="aegis.init\\(\\) has not been called"):
        aegis.checkpoint.restore()


def test_checkpoint_save_and_restore():
    """checkpoint.save() then restore() returns metadata dict."""
    aegis.init()
    aegis.checkpoint.save({"weights": [1, 2, 3]}, tier="auto")
    result = aegis.checkpoint.restore()
    assert "tier" in result
    assert "epoch" in result
    assert "fidelity_flag" in result


# ---------------------------------------------------------------------------
# transport tests


def test_transport_before_init_raises():
    """transport.set_fast_path() before init raises RuntimeError."""
    with pytest.raises(RuntimeError, match="aegis.init\\(\\) has not been called"):
        aegis.transport.set_fast_path(False)


def test_transport_set_fast_path():
    """set_fast_path(False) updates the live policy on the EPE."""
    aegis.init()
    # Confirm default is True
    assert aegis.transport.get_fast_path() is True

    aegis.transport.set_fast_path(False)
    assert aegis.transport.get_fast_path() is False

    # And the policy object is actually updated
    policy = _state.runtime.epe._policy  # type: ignore[union-attr]
    assert policy.allow_b0_fast_path is False


def test_transport_get_fast_path_before_init_raises():
    """get_fast_path() before init raises RuntimeError."""
    with pytest.raises(RuntimeError, match="aegis.init\\(\\) has not been called"):
        aegis.transport.get_fast_path()


# ---------------------------------------------------------------------------
# config.load_policy() tests


def test_config_missing_file():
    """load_policy() for a nonexistent file returns default OperatorPolicy."""
    result = load_policy("nonexistent_aegis_policy_file_xyz.yaml")
    assert isinstance(result, OperatorPolicy)
    # Defaults should match the dataclass defaults
    assert result.gpu_hr_cost == pytest.approx(2.35)
    assert result.allow_b0_fast_path is True


def test_config_valid_json():
    """load_policy() correctly parses a JSON policy file."""
    policy_data = {
        "economics": {
            "gpu_hourly_cost_usd": 3.50,
            "policy": "correctness_first",
        },
        "escalation": {
            "correlation_window_ms": 2000,
            "correlation_node_threshold": 5,
        },
        "tiers": {
            "b0_transport": {"fast_path": False},
            "b1_compute": {"max_consecutive_fallbacks": 4},
        },
    }
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    ) as f:
        json.dump(policy_data, f)
        path = f.name

    try:
        result = load_policy(path)
        assert result.gpu_hr_cost == pytest.approx(3.50)
        assert result.economics_policy == "correctness_first"
        assert result.correlation_window_secs == pytest.approx(2.0)
        assert result.correlation_node_threshold == 5
        assert result.allow_b0_fast_path is False
        assert result.max_consecutive_fallbacks == 4
    finally:
        os.unlink(path)


def test_config_valid_yaml():
    """load_policy() correctly parses a YAML policy file (skip if no pyyaml)."""
    yaml = pytest.importorskip("yaml")

    policy_data = {
        "version": "0.1",
        "economics": {
            "gpu_hourly_cost_usd": 2.80,
            "policy": "latency_first",
        },
        "escalation": {
            "correlation_window_ms": 3000,
            "correlation_node_threshold": 4,
        },
        "tiers": {
            "b0_transport": {"fast_path": True},
            "b1_compute": {"max_consecutive_fallbacks": 6},
        },
    }

    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(policy_data, f)
        path = f.name

    try:
        result = load_policy(path)
        assert result.gpu_hr_cost == pytest.approx(2.80)
        assert result.economics_policy == "latency_first"
        assert result.correlation_window_secs == pytest.approx(3.0)
        assert result.correlation_node_threshold == 4
        assert result.allow_b0_fast_path is True
        assert result.max_consecutive_fallbacks == 6
    finally:
        os.unlink(path)


def test_config_policy_passed_directly():
    """init(policy=...) skips file loading and uses the supplied policy."""
    custom = OperatorPolicy(gpu_hr_cost=9.99)
    aegis.init(policy=custom)
    policy = _state.runtime.epe._policy  # type: ignore[union-attr]
    assert policy.gpu_hr_cost == pytest.approx(9.99)
