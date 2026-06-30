"""
T2 explicit transport API — override B0 fast-path settings at runtime.

Provides direct control over the B0 transport layer's fast-path autonomy,
which normally lets R²CCL begin NIC migration before the Failure Classifier
completes its classification (§3.1).

Disabling the fast path routes B0 faults through the full EPE classification
before any recovery action is taken — useful for debugging whether AEGIS is
handling a B0 fault correctly.

Usage:
    import aegis
    aegis.init()

    # Disable fast-path for debugging (all B0 faults wait for classifier)
    aegis.transport.set_fast_path(False)

    # Re-enable after debugging
    aegis.transport.set_fast_path(True)

    # Query current setting
    enabled = aegis.transport.get_fast_path()
"""

from __future__ import annotations

import logging

from aegis import _state

logger = logging.getLogger(__name__)


def _require_init() -> None:
    if _state.runtime is None:
        raise RuntimeError(
            "aegis.init() has not been called. "
            "Call aegis.init() before using this API."
        )


def set_fast_path(enabled: bool) -> None:
    """
    Override the B0 fast-path setting at runtime.

    When enabled (the default), R²CCL may begin NIC migration immediately
    without waiting for the Failure Classifier to complete classification.
    When disabled, all B0 faults wait for the EPE routing decision.

    Args:
        enabled: True to enable fast-path autonomy; False to disable.

    Raises:
        RuntimeError: if aegis.init() has not been called.
    """
    _require_init()
    rt = _state.runtime
    assert rt is not None  # narrowing

    rt.epe._policy.allow_b0_fast_path = bool(enabled)
    logger.info("[transport] B0 fast-path set to %s", enabled)


def get_fast_path() -> bool:
    """
    Return the current B0 fast-path setting from the live policy.

    Raises:
        RuntimeError: if aegis.init() has not been called.
    """
    _require_init()
    rt = _state.runtime
    assert rt is not None  # narrowing

    return rt.epe._policy.allow_b0_fast_path
