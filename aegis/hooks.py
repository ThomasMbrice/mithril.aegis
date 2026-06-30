"""
Hook registry — tracks which transparent interception hooks are installed.

Each hook name corresponds to one of the five AEGIS integration surfaces:
  transport   — B0 NIC/collective interception (R²CCL shim)
  compute     — B1 neighbor-absorb interception (MeCeFO)
  checkpoint  — B2-B4 tiered checkpoint interception (TierCheck)
  telemetry   — Unified Telemetry Plane sensor registration
  policy      — Escalation Policy Engine live-config hooks
"""

from __future__ import annotations

VALID_HOOKS: frozenset[str] = frozenset(
    {"transport", "compute", "checkpoint", "telemetry", "policy"}
)


class HookRegistry:
    """
    Track which transparent interception hooks are currently active.

    Hooks are installed at aegis.init() and removed at aegis.disable().
    Individual hooks may be excluded at init time via ``disable=[...]``.
    """

    def __init__(self, mode: str = "active") -> None:
        self._mode = mode
        self._active: set[str] = set()

    def install(self, name: str) -> None:
        """
        Mark a hook as installed.

        Raises ValueError for unknown hook names so that typos are caught
        immediately (§5 contract #1 — no silent inactivity).
        """
        if name not in VALID_HOOKS:
            raise ValueError(
                f"Unknown hook {name!r}. Valid hooks: {sorted(VALID_HOOKS)}"
            )
        self._active.add(name)

    def uninstall(self, name: str) -> None:
        """Remove a hook from the active set (no-op if not installed)."""
        self._active.discard(name)

    def is_active(self, name: str) -> bool:
        """Return True iff the named hook is currently installed."""
        return name in self._active

    def disable_all(self) -> None:
        """Remove all active hooks (called by aegis.disable())."""
        self._active.clear()

    @property
    def active_hooks(self) -> list[str]:
        """Sorted list of currently active hook names."""
        return sorted(self._active)

    @property
    def mode(self) -> str:
        """Operating mode: 'active' or 'observe_only'."""
        return self._mode
