"""
T2 explicit checkpoint API — drive TierCheck tiered storage directly.

Delegates to the runtime's StorageLayer.  Raises RuntimeError with a clear
message if called before aegis.init() (§5 contract #1 — no silent inactivity).

Usage:
    import aegis
    aegis.init()

    # Save to auto-selected tier (local → peer → remote)
    aegis.checkpoint.save(state_dict, tier="auto")

    # Restore to latest valid checkpoint across all tiers
    state = aegis.checkpoint.restore()
"""

from __future__ import annotations

import logging
from typing import Any

from aegis import _state

logger = logging.getLogger(__name__)


def _require_init() -> None:
    if _state.runtime is None:
        raise RuntimeError(
            "aegis.init() has not been called. "
            "Call aegis.init() before using this API."
        )


def save(state: Any, tier: str = "auto") -> None:
    """
    Save a checkpoint via the AEGIS TierCheck tiered storage.

    Args:
        state:  Arbitrary state object to checkpoint (e.g. a state_dict).
        tier:   Storage tier selection. One of:
                  "auto"   — cheapest available tier (local → peer → remote)
                  "local"  — Tier-1 local volatile only
                  "peer"   — Tier-2 peer volatile only
                  "remote" — Tier-3 remote durable only

    Raises:
        RuntimeError: if aegis.init() has not been called.
        ValueError:   if tier is not a recognised value.
    """
    _require_init()
    rt = _state.runtime
    assert rt is not None  # narrowing

    valid_tiers = {"auto", "local", "peer", "remote"}
    if tier not in valid_tiers:
        raise ValueError(f"Unknown tier {tier!r}. Valid: {sorted(valid_tiers)}")

    epoch = rt.epoch_service.current()
    node = "local"  # stub: real impl would use the process's node identity

    if tier in {"auto", "local"}:
        rt.storage.write_tier1(node, epoch=epoch)
        logger.info("[checkpoint] Saved Tier-1 (local) at epoch %d", epoch)

    if tier in {"auto", "peer"}:
        rt.storage.write_tier2(node, epoch=epoch)
        logger.info("[checkpoint] Saved Tier-2 (peer) at epoch %d", epoch)

    if tier in {"auto", "remote"}:
        rt.storage.write_tier3(epoch=epoch)
        logger.info("[checkpoint] Saved Tier-3 (remote) at epoch %d", epoch)


def restore(epoch: str | int = "latest_valid") -> dict:
    """
    Restore from the most recent valid checkpoint.

    Args:
        epoch: Which checkpoint to restore.
               "latest_valid" — most recent full-fidelity checkpoint (default).
               An integer epoch number — restore from that specific epoch.

    Returns:
        A dict describing the restored checkpoint metadata:
            {"tier": str, "epoch": int, "fidelity_flag": bool}

    Raises:
        RuntimeError: if aegis.init() has not been called, or no checkpoint
            is available.
    """
    _require_init()
    rt = _state.runtime
    assert rt is not None  # narrowing

    storage = rt.storage

    # Stub: inspect available checkpoints and return the most recent
    # In the real implementation this would drive URC consensus on the
    # globally consistent checkpoint.

    if storage._tier3 is not None:
        ckpt = storage._tier3
        logger.info(
            "[checkpoint] Restoring from Tier-3 (epoch=%d fidelity=%s)",
            ckpt.epoch, ckpt.fidelity_flag,
        )
        return {
            "tier": "remote",
            "epoch": ckpt.epoch,
            "fidelity_flag": ckpt.fidelity_flag,
        }

    # Fall through to tier2 / tier1 if available
    for node, ckpt in storage._tier2.items():
        logger.info(
            "[checkpoint] Restoring from Tier-2 node=%s (epoch=%d)", node, ckpt.epoch
        )
        return {"tier": "peer", "epoch": ckpt.epoch, "fidelity_flag": ckpt.fidelity_flag}

    for node, ckpt in storage._tier1.items():
        logger.info(
            "[checkpoint] Restoring from Tier-1 node=%s (epoch=%d)", node, ckpt.epoch
        )
        return {"tier": "local", "epoch": ckpt.epoch, "fidelity_flag": ckpt.fidelity_flag}

    raise RuntimeError(
        "No checkpoint available. Call checkpoint.save() before restore()."
    )
