"""Shared mutable state for the aegis package (avoids circular imports)."""
from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import AegisRuntime
    from .hooks import HookRegistry

runtime: "AegisRuntime | None" = None
hooks: "HookRegistry | None" = None
mode: str = "active"          # "active" | "observe_only"
initialized: bool = False
_loop: "asyncio.AbstractEventLoop | None" = None
_loop_thread: "threading.Thread | None" = None
