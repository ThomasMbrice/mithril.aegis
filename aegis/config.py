"""
AEGIS policy file loader — T1 integration tier.

Loads operator policy from aegis.yaml (or .json) into an OperatorPolicy
dataclass.  If the file is absent, returns safe conservative defaults.

Field mapping (YAML path → OperatorPolicy attribute):
  economics.gpu_hourly_cost_usd         → gpu_hr_cost
  escalation.correlation_window_ms      → correlation_window_secs (÷1000)
  escalation.correlation_node_threshold → correlation_node_threshold
  tiers.b0_transport.fast_path          → allow_b0_fast_path
  tiers.b1_compute.max_consecutive_fallbacks → max_consecutive_fallbacks
  economics.policy                      → economics_policy
  cluster.gpu_count                     → gpu_count
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from aegis.policy.dsl import OperatorPolicy

logger = logging.getLogger(__name__)


def load_policy(path: str = "aegis.yaml") -> OperatorPolicy:
    """
    Load a policy file into an OperatorPolicy.

    Supports .yaml/.yml (requires pyyaml) and .json (stdlib).
    If the file is not found, returns a default OperatorPolicy and logs a
    debug message — the absence of a policy file is normal at T0.

    Raises:
        RuntimeError: if the file has a .yaml/.yml extension but pyyaml is
            not installed.
    """
    if not os.path.exists(path):
        logger.debug("[AEGIS] Policy file %r not found — using safe defaults", path)
        return OperatorPolicy()

    ext = os.path.splitext(path)[1].lower()

    if ext in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            raise RuntimeError(
                f"aegis.yaml found at {path!r} but pyyaml is not installed. "
                "Run: pip install pyyaml"
            ) from None
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        logger.warning(
            "[AEGIS] Unrecognised policy file extension %r — using safe defaults", ext
        )
        return OperatorPolicy()

    return _parse_policy(raw)


def _get_nested(d: dict, *keys: str, default: Any = None) -> Any:
    """Traverse nested dict keys, returning default if any key is missing."""
    node: Any = d
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is default:
            return default
    return node


def _parse_policy(raw: dict[str, Any]) -> OperatorPolicy:
    """Map the raw YAML/JSON dict to an OperatorPolicy, ignoring unknown keys."""
    policy = OperatorPolicy()

    # economics.gpu_hourly_cost_usd → gpu_hr_cost
    val = _get_nested(raw, "economics", "gpu_hourly_cost_usd")
    if val is not None:
        policy.gpu_hr_cost = float(val)

    # economics.policy → economics_policy
    val = _get_nested(raw, "economics", "policy")
    if val is not None:
        policy.economics_policy = str(val)

    # escalation.correlation_window_ms → correlation_window_secs (÷1000)
    val = _get_nested(raw, "escalation", "correlation_window_ms")
    if val is not None:
        policy.correlation_window_secs = float(val) / 1000.0

    # escalation.correlation_node_threshold → correlation_node_threshold
    val = _get_nested(raw, "escalation", "correlation_node_threshold")
    if val is not None:
        policy.correlation_node_threshold = int(val)

    # tiers.b0_transport.fast_path → allow_b0_fast_path
    val = _get_nested(raw, "tiers", "b0_transport", "fast_path")
    if val is not None:
        policy.allow_b0_fast_path = bool(val)

    # tiers.b1_compute.max_consecutive_fallbacks → max_consecutive_fallbacks
    val = _get_nested(raw, "tiers", "b1_compute", "max_consecutive_fallbacks")
    if val is not None:
        policy.max_consecutive_fallbacks = int(val)

    # cluster.gpu_count → gpu_count (KPI meter's per-fault GPU-hours multiplier)
    val = _get_nested(raw, "cluster", "gpu_count")
    if val is not None:
        policy.gpu_count = int(val)

    logger.debug("[AEGIS] Loaded policy: %r", policy)
    return policy
