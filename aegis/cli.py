"""
aegis CLI — thin wrapper so ``aegis run`` is a torchrun drop-in.

This keeps the CLI surface minimal: AEGIS's real value is the in-process
library, not a daemon.  The CLI exists only to make the launcher swap
(``torchrun`` → ``aegis run``) a single-word change.

Usage:
    aegis run --nproc_per_node=8 train.py [args...]
"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    """Entry point for the ``aegis`` CLI command."""
    args = sys.argv[1:]

    if not args or args[0] != "run":
        print(
            "Usage: aegis run [torchrun-args...] script.py [script-args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    torchrun_args = args[1:]  # strip "run"
    cmd = [sys.executable, "-m", "torch.distributed.run"] + torchrun_args

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(
            "aegis run: torchrun not found. Install PyTorch: pip install torch",
            file=sys.stderr,
        )
        sys.exit(1)
