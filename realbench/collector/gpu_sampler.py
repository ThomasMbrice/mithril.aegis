"""
GPU utilization sampler — the nvidia-smi fallback path from test_suite.md §4.5.4(a).

DCGM (``dcgmi dmon``) is the doc's preferred sampler because it exposes
``DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`` — the signal that distinguishes real
training compute from a rank spin-waiting on a hung collective. DCGM
availability on the target cluster is unconfirmed, so this module
deliberately implements only the ``nvidia-smi`` fallback.

**No ``tensor_active`` field is written, ever.** Every row this sampler
produces has only ``gpu_util_pct`` (SM utilization — can be nonzero during a
spin-wait), ``mem_used_mib``, and ``power_draw_w``. Downstream code
(``realbench/collector/align.py``) must never synthesize a tensor-active
value from these fields — the absence of the field is the signal that it
wasn't measured, not that it was zero. See the honesty requirements in the
real-cluster harness plan.

Runs as an independent process (not inside any training rank), one per node,
matching test_suite.md §4.5.1's three-coexisting-processes architecture.
Stops when a stop-file sentinel appears (written by the SLURM script once
the training job exits) rather than a fixed duration, so a longer-than-
expected run is never cut off mid-collection.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import time
from pathlib import Path

from realbench.training.step_log import append_jsonl

DEFAULT_INTERVAL_SECS = 1.0

NVIDIA_SMI_QUERY_FIELDS = ["index", "utilization.gpu", "memory.used", "power.draw"]


def sample_once(nvidia_smi_bin: str = "nvidia-smi") -> list[dict]:
    """
    Run one nvidia-smi query, return one row per GPU.

    Raises ``subprocess.CalledProcessError`` if nvidia-smi isn't available or
    fails — callers decide whether that's fatal (real cluster: yes, should
    never happen; local dev without a GPU: caller can catch and skip).
    """
    query = ",".join(NVIDIA_SMI_QUERY_FIELDS)
    result = subprocess.run(
        [nvidia_smi_bin, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    wall_clock = time.time()
    rows = []
    reader = csv.reader(io.StringIO(result.stdout))
    for line in reader:
        if not line:
            continue
        index, util, mem_used, power = (v.strip() for v in line)
        rows.append({
            "wall_clock": wall_clock,
            "gpu_index": int(index),
            "gpu_util_pct": float(util),
            "mem_used_mib": float(mem_used),
            "power_draw_w": float(power) if power not in ("", "[N/A]") else None,
        })
    return rows


def run(
    out_path: str | Path,
    stop_file: str | Path,
    interval_secs: float = DEFAULT_INTERVAL_SECS,
    nvidia_smi_bin: str = "nvidia-smi",
) -> None:
    """
    Poll nvidia-smi at ``interval_secs`` resolution, appending rows to
    ``out_path`` until ``stop_file`` appears.
    """
    stop_path = Path(stop_file)
    while not stop_path.exists():
        loop_start = time.monotonic()
        try:
            for row in sample_once(nvidia_smi_bin):
                append_jsonl(out_path, row)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            append_jsonl(out_path, {
                "wall_clock": time.time(),
                "sample_error": str(exc),
            })
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, interval_secs - elapsed))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Path to write gpu_log.jsonl")
    p.add_argument("--stop-file", required=True,
                    help="Sampler exits once this file appears (SLURM script creates it "
                         "after the training job exits).")
    p.add_argument("--interval-secs", type=float, default=DEFAULT_INTERVAL_SECS)
    p.add_argument("--nvidia-smi-bin", default="nvidia-smi")
    args = p.parse_args()

    run(args.out, args.stop_file, args.interval_secs, args.nvidia_smi_bin)


if __name__ == "__main__":
    main()
