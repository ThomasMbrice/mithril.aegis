"""
Phase 0 report — steady-state overhead of AEGIS's hooks (present, zero
faults) vs. bare torchrun, plus the gate that decides whether Phase 1 is
allowed to run at all (test_suite.md §8: "if we can't reproduce, stop and
fix integration before any comparison").

SCOPE NOTE, read before trusting the gate: this Phase 0 measures "AEGIS
hooks present, zero faults" overhead — a superset of MeCeFO's/R2CCL's/
TierCheck's individually-claimed overheads (4.18% / <1% / <10s), because no
fault fires here at all, so R2CCL's migration cost and TierCheck's restore
cost are never triggered; only idle-path hook overhead is measured. The
gate below is therefore a **sanity ceiling** (catches "AEGIS's hooks are
grossly expensive or something is broken"), not a strict reproduction of
the papers' numbers — those need Phase 1's per-tier isolation with each
layer's real fault fired. Both figures are printed so a reader can judge
for themselves.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from realbench.collector.align import load_step_log
from realbench.honesty import render_honesty_block

DEFAULT_OVERHEAD_SANITY_CEILING_PCT = 25.0

# Informational only — NOT the pass/fail bar for this phase's gate (see
# module docstring for why). Surfaced in the report for context.
PAPER_TARGETS_PCT = {
    "MeCeFO (compute)": 4.18,
    "R2CCL (transport)": 1.0,
}


def steady_state_throughput(step_log_path: Path, warmup_steps: int) -> float | None:
    records = load_step_log(step_log_path)
    post_warmup = records[warmup_steps:]
    if len(post_warmup) < 2:
        return None
    span = post_warmup[-1].wall_clock - post_warmup[0].wall_clock
    if span <= 0:
        return None
    return (len(post_warmup) - 1) / span


def collect_condition_stats(out: Path, condition: str, repeat: int, warmup_steps: int) -> dict:
    throughputs = []
    manifest = None
    for i in range(repeat):
        run_dir = out / condition / f"run{i}"
        step_log_path = run_dir / "step_log.jsonl"
        if not step_log_path.exists():
            continue
        tput = steady_state_throughput(step_log_path, warmup_steps)
        if tput is not None:
            throughputs.append(tput)
        manifest_path = run_dir / "run_manifest.json"
        if manifest is None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
    return {
        "throughputs": throughputs,
        "mean": statistics.mean(throughputs) if throughputs else None,
        "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "manifest": manifest,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=50,
                    help="Discarded before measuring steady-state throughput (test_suite.md §6.5).")
    p.add_argument("--overhead-sanity-ceiling-pct", type=float, default=DEFAULT_OVERHEAD_SANITY_CEILING_PCT)
    args = p.parse_args()

    out = Path(args.out)
    control = collect_condition_stats(out, "control", args.repeat, args.warmup_steps)
    aegis_stats = collect_condition_stats(out, "aegis", args.repeat, args.warmup_steps)

    gate_passed = False
    overhead_pct = None
    if control["mean"] and aegis_stats["mean"]:
        overhead_pct = (control["mean"] - aegis_stats["mean"]) / control["mean"] * 100.0
        gate_passed = overhead_pct <= args.overhead_sanity_ceiling_pct

    manifest = aegis_stats["manifest"] or control["manifest"] or {}

    lines = ["# Phase 0 — Trust-Anchor Report", ""]
    lines.append(render_honesty_block(manifest))
    lines.append("## Steady-State Throughput (post-warmup, no faults)")
    lines.append("")
    lines.append("| Condition | Mean steps/sec | Std | N runs with data |")
    lines.append("|---|---|---|---|")
    for name, stats in (("control (bare torchrun)", control), ("aegis-on", aegis_stats)):
        mean = f"{stats['mean']:.4f}" if stats["mean"] is not None else "N/A"
        std = f"{stats['std']:.4f}" if stats["mean"] is not None else "N/A"
        lines.append(f"| {name} | {mean} | {std} | {len(stats['throughputs'])}/{args.repeat} |")
    lines.append("")

    if overhead_pct is not None:
        lines.append(f"**Measured overhead (AEGIS hooks present, zero faults): {overhead_pct:.2f}%**")
        lines.append("")
        lines.append(f"Sanity ceiling for this phase's gate: {args.overhead_sanity_ceiling_pct:.1f}% "
                      f"(NOT the papers' per-primitive targets — see module docstring).")
        lines.append("")
        lines.append("Informational comparison to the papers' own per-primitive targets "
                      "(not directly reproduced by this no-fault measurement):")
        for name, target in PAPER_TARGETS_PCT.items():
            lines.append(f"- {name}: {target:.2f}% target")
        lines.append("")
        lines.append(f"**GATE: {'PASS' if gate_passed else 'FAIL'}**")
    else:
        lines.append("**GATE: FAIL — insufficient data to compute overhead "
                      "(one or both conditions produced no usable step-log data).**")
    lines.append("")

    (out / "phase0_report.md").write_text("\n".join(lines))

    summary = {
        "gate_passed": gate_passed,
        "overhead_pct": overhead_pct,
        "overhead_sanity_ceiling_pct": args.overhead_sanity_ceiling_pct,
        "control_mean_steps_per_sec": control["mean"],
        "aegis_mean_steps_per_sec": aegis_stats["mean"],
    }
    (out / "phase0_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n".join(lines))
    sys.exit(0 if gate_passed else 1)


if __name__ == "__main__":
    main()
