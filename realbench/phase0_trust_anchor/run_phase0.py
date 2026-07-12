"""
Phase 0 — trust-anchor orchestration (test_suite.md §7/§8).

Launches the control (bare torchrun, no AEGIS) and AEGIS-on conditions of
``realbench.training.w1_llama7b`` — same script, one flag — back to back,
``--repeat`` times each, with NO fault injection at all. This is pure
steady-state overhead measurement; the gate this phase computes ("did AEGIS's
hooks reproduce a sane overhead, not a broken integration") must pass before
Phase 1's real fault injection is meaningful to run at all.

Intended to run inside a SLURM allocation (see realbench/slurm/sbatch_phase0.sh)
or an interactive allocation. Only launches subprocesses and waits for them —
does not import torch itself, so it stays usable as a lightweight coordinator
even before the real dependencies are on the path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = ("control", "aegis")


def build_cmd(
    nproc: int, steps: int, log_dir: Path, aegis_enabled: bool,
    seed: int, batch_size: int, seq_len: int, tiny: bool, aegis_policy_path: str,
    rdzv_endpoint: str | None = None,
) -> list[str]:
    # --standalone is the documented torchrun choice for single-node
    # multi-GPU jobs (our case — W1 is one 8xA100 node) and is the default
    # here. --rdzv-endpoint is an escape hatch: some cluster network configs
    # (and this project's own local dev sandbox) hang doing reverse-DNS
    # lookups under --standalone's rendezvous; pass e.g. 127.0.0.1:29500 to
    # use a fixed static endpoint instead if that happens on SeaWulf too.
    rdzv_args = (
        ["--rdzv-backend=static", f"--rdzv-endpoint={rdzv_endpoint}", "--node-rank=0"]
        if rdzv_endpoint else ["--standalone"]
    )
    cmd = [
        "aegis", "run", f"--nproc_per_node={nproc}", *rdzv_args,
        "-m", "realbench.training.w1_llama7b",
        "--steps", str(steps), "--log-dir", str(log_dir),
        "--seed", str(seed), "--batch-size", str(batch_size), "--seq-len", str(seq_len),
    ]
    if aegis_enabled:
        cmd += ["--aegis-enabled", "--aegis-policy-path", aegis_policy_path]
    if tiny:
        cmd.append("--tiny")
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Base output directory")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--repeat", type=int, default=3,
                    help="N>=3 runs per condition, test_suite.md §6.3 (mean +/- std).")
    p.add_argument("--nproc-per-node", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--aegis-policy-path", default="aegis.yaml")
    p.add_argument("--tiny", action="store_true",
                    help="DEBUG ONLY — see realbench/training/w1_llama7b.py.")
    p.add_argument("--run-timeout-secs", type=float, default=3600.0)
    p.add_argument("--rdzv-endpoint", default=None,
                    help="Escape hatch for --standalone rendezvous hangs — see build_cmd().")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    run_index = []
    for condition in CONDITIONS:
        for i in range(args.repeat):
            run_dir = out / condition / f"run{i}"
            cmd = build_cmd(
                nproc=args.nproc_per_node, steps=args.steps, log_dir=run_dir,
                aegis_enabled=(condition == "aegis"), seed=args.seed,
                batch_size=args.batch_size, seq_len=args.seq_len,
                tiny=args.tiny, aegis_policy_path=args.aegis_policy_path,
                rdzv_endpoint=args.rdzv_endpoint,
            )
            print(f"[run_phase0] {condition} run{i}: {' '.join(cmd)}", flush=True)
            start = time.time()
            try:
                result = subprocess.run(cmd, timeout=args.run_timeout_secs)
                ok = result.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
            elapsed = time.time() - start
            run_index.append({
                "condition": condition, "run": i, "log_dir": str(run_dir),
                "ok": ok, "elapsed_secs": elapsed,
            })
            print(f"[run_phase0] {condition} run{i}: {'OK' if ok else 'FAILED'} ({elapsed:.1f}s)", flush=True)

    (out / "run_index.json").write_text(json.dumps(run_index, indent=2))

    if not all(r["ok"] for r in run_index):
        print("[run_phase0] one or more runs failed — see run_index.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
