"""
AEGIS — Adaptive, Escalating, Graded Infrastructure Survivability

Blast-radius-aware fault-tolerance runtime for LLM training & serving.
Composes R²CCL (transport), MeCeFO (compute), and TierCheck (storage)
under a unified telemetry + escalation policy plane.

Phase 0 MVP: Telemetry plane, Failure Classifier, Fault Epoch Service,
EPE stub, layer stubs, chaos-inject harness.

Public API (§2, §4 of ui.md):
    aegis.init(...)          — T0: install transparent interception hooks
    aegis.status()           — T2: active hooks, epoch, KPI summary
    aegis.explain()          — T2: what AEGIS did for the last fault
    aegis.dashboard(fmt=...) — T2: per-tier fault breakdown + $/GPU-hr KPI report
    aegis.disable()          — T2: kill switch, full passive mode
    aegis.checkpoint         — T2: explicit checkpoint save/restore
    aegis.transport          — T2: explicit transport / fast-path control
    aegis.policy.set(k, v)   — T2: live policy update by dotted key
"""

from __future__ import annotations

__version__ = "0.1.0"

import asyncio
import logging
import threading
from typing import Any

from aegis import _state
from aegis import checkpoint  # noqa: F401 — re-export for `from aegis import checkpoint`
from aegis import transport   # noqa: F401 — re-export for `from aegis import transport`
from aegis import policy      # noqa: F401 — re-export for `aegis.policy.set(...)`
from aegis import dashboard as _dashboard
from aegis.config import load_policy
from aegis.hooks import HookRegistry, VALID_HOOKS
from aegis.policy import dsl as _policy_dsl
from aegis.runtime import AegisRuntime

logger = logging.getLogger(__name__)

_VALID_MODES: frozenset[str] = frozenset({"active", "observe_only"})


class AegisInitError(RuntimeError):
    """Raised when aegis.init() cannot install a required hook."""


# ---------------------------------------------------------------------------
# Public API


def init(
    disable: list[str] | None = None,
    mode: str = "active",
    policy_path: str = "aegis.yaml",
    policy: _policy_dsl.OperatorPolicy | None = None,
) -> None:
    """
    Initialize AEGIS transparent fault-tolerance interception.

    Must be called before ``dist.init_process_group()``.
    Raises ``AegisInitError`` if any required hook cannot be installed.
    **Never silently no-ops** — a fault-tolerance library that silently
    isn't active is the worst possible failure (§5 contract #1).

    Args:
        disable:     Hook names to skip. Valid values:
                     ``transport``, ``compute``, ``checkpoint``,
                     ``telemetry``, ``policy``.
        mode:        ``"active"`` (default) — AEGIS classifies faults and
                     performs recovery.
                     ``"observe_only"`` — AEGIS classifies and logs what it
                     *would* do, but takes no recovery action.  Ideal for
                     shadow-mode evaluation with design partners.
        policy_path: Path to ``aegis.yaml`` policy file.  Ignored if
                     ``policy`` is provided.
        policy:      Pre-built ``OperatorPolicy`` instance.  Takes precedence
                     over ``policy_path``.

    Raises:
        ValueError:      if ``disable`` contains an unknown hook name, or
                         ``mode`` is not a recognised value.
        AegisInitError:  if the runtime cannot be started within the timeout.
    """
    # Validate mode
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Valid modes: {sorted(_VALID_MODES)}"
        )

    # Validate and normalise disable list
    disable_list = list(disable or [])
    for h in disable_list:
        if h not in VALID_HOOKS:
            raise ValueError(
                f"Unknown hook {h!r}. Valid hooks: {sorted(VALID_HOOKS)}"
            )

    # Load policy
    resolved_policy = policy or load_policy(policy_path)

    # Build hook registry
    hooks = HookRegistry(mode=mode)
    for hook_name in VALID_HOOKS:
        if hook_name not in disable_list:
            try:
                hooks.install(hook_name)
            except ValueError as exc:
                # Should not happen — we only install VALID_HOOKS — but be safe.
                raise AegisInitError(
                    f"Failed to install hook {hook_name!r}: {exc}"
                ) from exc

    # Create runtime
    rt = AegisRuntime(policy=resolved_policy)

    # In observe_only mode, set the EPE to not take recovery actions
    if mode == "observe_only":
        rt.epe._observe_only = True

    # Start the runtime in a dedicated background event loop
    loop, thread = _start_event_loop()
    try:
        future = asyncio.run_coroutine_threadsafe(rt.start(), loop)
        future.result(timeout=5.0)
    except Exception as exc:
        # Tear down the loop on failure so we don't leak threads
        loop.call_soon_threadsafe(loop.stop)
        raise AegisInitError(f"AEGIS runtime failed to start: {exc}") from exc

    # Commit state — all-or-nothing
    _state.runtime = rt
    _state.hooks = hooks
    _state.mode = mode
    _state.initialized = True
    _state._loop = loop
    _state._loop_thread = thread

    logger.info("[AEGIS] Initialized: mode=%s, hooks=%s", mode, hooks.active_hooks)


def status() -> dict:
    """
    Return the current AEGIS operational status.

    Returns a dict with keys:
        ``initialized``   — bool
        ``mode``          — "active" or "observe_only"
        ``active_hooks``  — sorted list of installed hook names
        ``current_epoch`` — current fault epoch counter value
        ``kpi``           — KPI summary dict (see KPIMeter.summary())

    Raises:
        RuntimeError: if ``aegis.init()`` has not been called.
    """
    _require_init()
    rt = _state.runtime
    assert rt is not None  # narrowing for type checker
    return {
        "initialized": _state.initialized,
        "mode": _state.mode,
        "active_hooks": _state.hooks.active_hooks if _state.hooks else [],
        "current_epoch": rt.epoch_service.current(),
        "kpi": rt.kpi.summary(),
    }


def explain() -> dict:
    """
    Explain the last fault processed by AEGIS.

    Returns a dict answering "what did AEGIS do and why?" — the first call
    a debugging session should make when AEGIS is suspected of causing a
    problem (§4.2 ui.md).

    Returns:
        Dict with keys: ``epoch``, ``signal``, ``node``, ``original_tier``,
        ``final_tier``, ``escalated``, ``success``, ``degraded``, ``message``,
        ``observe_only``.  If no fault has been processed yet, returns a dict
        with a ``message`` key and ``observe_only``.

    Raises:
        RuntimeError: if ``aegis.init()`` has not been called.
    """
    _require_init()
    history = _state.runtime.epe.history  # type: ignore[union-attr]
    if not history:
        return {
            "message": "No faults processed yet.",
            "observe_only": _state.mode == "observe_only",
        }
    last = history[-1]
    return {
        "epoch": last.epoch,
        "signal": last.event.fault_signal.value,
        "node": last.event.node,
        "original_tier": last.original_tier.name,
        "final_tier": last.final_tier.name,
        "escalated": last.escalated,
        "success": last.result.success if last.result else False,
        "degraded": last.result.degraded if last.result else False,
        "message": last.result.message if last.result else "",
        "observe_only": _state.mode == "observe_only",
    }


def dashboard(fmt: str = "text") -> str | dict:
    """
    Operator dashboard (§4 Layer E4) — per-tier fault breakdown and the
    live $/GPU-hr-saved KPI summary, in one call.

    Args:
        fmt: ``"text"`` (default) — a rendered ASCII report, ready to print.
             ``"json"`` — the same data as a plain, JSON-serialisable dict.

    Raises:
        RuntimeError: if ``aegis.init()`` has not been called.
        ValueError: if ``fmt`` is not ``"text"`` or ``"json"``.
    """
    _require_init()
    rt = _state.runtime
    assert rt is not None  # narrowing

    report = _dashboard.build_report(rt)

    if fmt == "json":
        return report
    if fmt == "text":
        return _dashboard.render_text(report)
    raise ValueError(f"Unknown fmt {fmt!r}. Valid: 'text', 'json'")


def disable() -> None:
    """
    Kill switch — AEGIS goes fully passive.  Training continues on native path.

    After ``disable()``, all hooks are removed and the runtime is stopped.
    Call ``aegis.init()`` again to re-enable.

    Safe to call even if AEGIS was never initialised (no-op).
    """
    if not _state.initialized:
        return

    if _state.hooks:
        _state.hooks.disable_all()

    if _state.runtime and _state._loop:
        future = asyncio.run_coroutine_threadsafe(
            _state.runtime.stop(), _state._loop
        )
        try:
            future.result(timeout=5.0)
        except Exception:
            logger.exception("[AEGIS] Error stopping runtime during disable()")
        finally:
            _state._loop.call_soon_threadsafe(_state._loop.stop)

    _state.runtime = None
    _state.hooks = None
    _state.initialized = False
    _state._loop = None
    _state._loop_thread = None

    logger.info("[AEGIS] Disabled — native PyTorch path active")


# ---------------------------------------------------------------------------
# Internal helpers


def _start_event_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Create and start a dedicated event loop in a daemon background thread."""
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True, name="aegis-event-loop")
    t.start()
    return loop, t


def _require_init() -> None:
    """Raise RuntimeError if aegis.init() has not been called."""
    if not _state.initialized or _state.runtime is None:
        raise RuntimeError(
            "AEGIS is not initialized. Call aegis.init() before using this API."
        )


def _reset() -> None:
    """
    Reset all module-level state.

    For testing only — do **not** call in production.  Calls ``disable()``
    internally to stop the background runtime if it is running.
    """
    disable()  # stop runtime if running
    _state.mode = "active"
    _state.initialized = False
