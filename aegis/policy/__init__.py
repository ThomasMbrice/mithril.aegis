"""Escalation Policy Engine (EPE) and operator policy DSL."""
from __future__ import annotations

from aegis import _state

# Dotted-key → OperatorPolicy attribute mapping
_KEY_MAP: dict[str, str] = {
    "economics.policy": "economics_policy",
    "economics.gpu_hourly_cost_usd": "gpu_hr_cost",
    "escalation.correlation_window_ms": "_correlation_window_ms",  # needs conversion
    "escalation.correlation_node_threshold": "correlation_node_threshold",
    "tiers.b0_transport.fast_path": "allow_b0_fast_path",
    "tiers.b1_compute.max_consecutive_fallbacks": "max_consecutive_fallbacks",
}


def set(key: str, value: object) -> None:
    """
    Update a live policy setting by dotted key.

    Example::

        aegis.policy.set("economics.policy", "correctness_first")
        aegis.policy.set("escalation.correlation_window_ms", 3000)

    Args:
        key:   Dotted policy key (see _KEY_MAP for valid keys).
        value: New value; will be coerced to the type of the existing attribute.

    Raises:
        RuntimeError: if AEGIS is not initialized.
        ValueError:   if key is not recognized.
    """
    if _state.runtime is None:
        raise RuntimeError("aegis.init() has not been called.")
    if key not in _KEY_MAP:
        raise ValueError(
            f"Unknown policy key {key!r}. Known keys: {sorted(_KEY_MAP)}"
        )

    attr = _KEY_MAP[key]
    policy = _state.runtime.epe._policy

    if key == "escalation.correlation_window_ms":
        # Convert milliseconds to seconds
        setattr(policy, "correlation_window_secs", float(value) / 1000.0)
    else:
        # Coerce to the type of the existing attribute
        existing = getattr(policy, attr)
        setattr(policy, attr, type(existing)(value))
