"""
Phase 1 — real B1 fault injection + recovery-cost report (test_suite.md
§4.5.1/§4.5.6).

Orchestrates, per condition, the three coexisting processes §4.5.1
describes: the training job, the real fault injector
(``chaos_inject.real_injector``), and the metrics collector
(``realbench.collector.gpu_sampler``) — then aligns their logs
(``realbench.collector.align``) into the recovery-cost numbers.

**Training is launched as N independent per-rank processes, NOT via
``aegis run``/torchrun.** This was empirically discovered while validating
this harness, not an arbitrary choice: torchrun's elastic agent
(``torch.distributed.elastic``) supervises all local ranks from one parent
process and proactively SIGTERMs every surviving sibling the moment any one
rank dies — which tears down the whole job before AEGIS's own heartbeat
detection or the process-group collective-timeout ever gets a chance to
run. That's fine for Phase 0 (nothing dies there, so ``aegis run`` is used),
but it defeats the entire premise of B1 testing here. Instead, each rank is
launched as an independent OS process (this module's own
``launch_rank_processes``, one ``subprocess.Popen`` per rank) with RANK/
LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set directly — the same
env-var convention torchrun uses, so ``w1_llama7b.py`` itself is unchanged
and each rank's ``torch.cuda.set_device(local_rank)`` still picks a
distinct GPU on the node. This process (this script) runs as a single
SLURM task on the one 8xA100 node (see
``realbench/slurm/sbatch_phase1_b1.sh``) and directly spawns its 8 rank
children — there is no torchrun/elastic-agent parent in the loop at all,
so no external supervisor tears down survivors when one rank dies.

Two conditions are run, both against the *same* real B1 kill at the *same*
trace-defined step (test_suite.md §6.1 determinism):
  - ``no_ft``  — no AEGIS at all. The "what happens if you do nothing" floor.
  - ``aegis``  — AEGIS active: real heartbeat detection, real classifier/EPE
    routing, real (proxy-tensor) MeCeFO absorb.

**No TorchFT condition and no B-VANILLA checkpoint-restart harness are
built in this pass** — flagged explicitly in the report rather than
fabricated. See the module docstring of ``realbench/training/w1_llama7b.py``
for why neither condition currently *resumes* training after the kill
(this build's recovery does not reconfigure the process group), which is
why ``recovery_gap``/``wasted_gpu_hours`` show as unmeasurable below —
what IS measured and reported is real detection latency and real
(proxy-tensor) absorb compute time for the ``aegis`` condition.

Explicitly gated on a passing Phase 0 report (test_suite.md §8) unless
``--skip-gate`` is passed for iterative debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from realbench.collector.align import (
    align_fault_events, load_chaos_log, load_gpu_log, load_halt_log, load_step_log,
)
from realbench.honesty import render_honesty_block
from realbench.phase1_per_tier.trace_b1 import suggest_target_step, write_trace
from realbench.training.step_log import read_jsonl

CONDITIONS = ("no_ft", "aegis")
DEFAULT_H100_USD_PER_HR = 2.35


def check_phase0_gate(phase0_dir: Path, skip_gate: bool) -> None:
    if skip_gate:
        print("[report_phase1] --skip-gate set — bypassing Phase 0 gate check.", file=sys.stderr)
        return
    summary_path = phase0_dir / "phase0_summary.json"
    if not summary_path.exists():
        sys.exit(
            f"[report_phase1] No phase0_summary.json found at {summary_path}. "
            "Phase 1 is gated on a passing Phase 0 trust-anchor run (test_suite.md §8). "
            "Run realbench.phase0_trust_anchor first, or pass --skip-gate to override "
            "(iterative debugging only — not for a reportable result)."
        )
    summary = json.loads(summary_path.read_text())
    if not summary.get("gate_passed"):
        sys.exit(
            f"[report_phase1] Phase 0 gate FAILED ({summary_path}) — refusing to run Phase 1. "
            "Fix the integration before any fault-injection comparison (test_suite.md §8). "
            "Pass --skip-gate to override (iterative debugging only)."
        )
    print(f"[report_phase1] Phase 0 gate PASSED ({summary_path}) — proceeding.")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch_rank_processes(run_dir: Path, aegis_enabled: bool, args, master_port: int) -> list[subprocess.Popen]:
    """
    One independent OS process per rank — see module docstring for why this
    is NOT ``aegis run``/torchrun for Phase 1 (torchrun's elastic agent
    tears down survivors the instant any rank dies).
    """
    procs = []
    for rank in range(args.nproc_per_node):
        env = os.environ.copy()
        env.update({
            "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(args.nproc_per_node),
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(master_port),
        })
        cmd = [
            sys.executable, "-m", "realbench.training.w1_llama7b",
            "--steps", str(args.steps), "--log-dir", str(run_dir),
            "--seed", str(args.seed), "--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len),
            "--heartbeat-timeout-secs", str(args.heartbeat_timeout_secs),
            "--collective-timeout-secs", str(args.collective_timeout_secs),
            "--setup-timeout-secs", str(args.setup_timeout_secs),
        ]
        if aegis_enabled:
            cmd += ["--aegis-enabled", "--aegis-policy-path", args.aegis_policy_path]
        if args.tiny:
            cmd.append("--tiny")
        log_file = open(run_dir / f"rank{rank}.log", "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT))
    return procs


def _terminate_all(procs: list[subprocess.Popen], *, grace_secs: float = 5.0) -> None:
    """Best-effort teardown of spawned children: SIGTERM, brief grace, then SIGKILL.

    This module deliberately runs its ranks without a torchrun/elastic parent
    (see module docstring), so nothing else reaps them — a leaked rank process
    keeps its GPU pinned and makes the next condition's run OOM on a dirty GPU,
    the exact failure this harness exists to measure. Every exit path from
    run_condition therefore funnels through here.
    """
    alive = [p for p in procs if p is not None and p.poll() is None]
    for p in alive:
        try:
            p.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_secs
    for p in alive:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass  # unreapable — best effort; SLURM cgroup cleanup is the backstop


def run_condition(condition: str, out: Path, trace_path: Path, args) -> Path:
    run_dir = out / condition
    run_dir.mkdir(parents=True, exist_ok=True)

    master_port = _find_free_port()
    print(f"[report_phase1] {condition}: launching {args.nproc_per_node} independent rank "
          f"processes (master_port={master_port})", flush=True)
    rank_procs = launch_rank_processes(run_dir, condition == "aegis", args, master_port)

    stop_file = run_dir / "GPU_SAMPLER_STOP"
    injector_proc: subprocess.Popen | None = None
    sampler_proc: subprocess.Popen | None = None

    try:
        # Wait for the training job to actually start writing step_log.jsonl and
        # pid_rank*.txt before starting the injector, which needs both.
        deadline = time.monotonic() + args.setup_timeout_secs
        step_log_path = run_dir / "step_log.jsonl"
        while time.monotonic() < deadline and not step_log_path.exists():
            time.sleep(1.0)
            if all(p.poll() is not None for p in rank_procs):
                sys.exit(f"[report_phase1] {condition}: all ranks exited before writing "
                          f"step_log.jsonl — check {run_dir}/rank*.log")

        injector_cmd = [
            sys.executable, "-m", "chaos_inject.real_injector",
            "--trace", str(trace_path), "--step-log", str(step_log_path),
            "--pid-dir", str(run_dir), "--out", str(run_dir / "chaos_log.jsonl"),
        ]
        with open(run_dir / "injector.log", "w") as injector_log:
            injector_proc = subprocess.Popen(
                injector_cmd, stdout=injector_log, stderr=subprocess.STDOUT
            )

        sampler_cmd = [
            sys.executable, "-m", "realbench.collector.gpu_sampler",
            "--out", str(run_dir / "gpu_log.jsonl"), "--stop-file", str(stop_file),
            "--interval-secs", "1.0", "--parent-pid", str(os.getpid()),
        ]
        with open(run_dir / "sampler.log", "w") as sampler_log:
            sampler_proc = subprocess.Popen(
                sampler_cmd, stdout=sampler_log, stderr=subprocess.STDOUT
            )

        train_timeout = args.setup_timeout_secs + args.collective_timeout_secs + 300.0
        deadline = time.monotonic() + train_timeout
        for proc in rank_procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[report_phase1] {condition}: a rank did not exit within {train_timeout}s "
                      "— killing it (this should not happen; the collective-timeout halt logic "
                      "should have exited cleanly).", file=sys.stderr)
                proc.kill()

        # Training is over — tell the sampler to stop, then let the injector
        # drain. Both waits are guarded: a hang here must not skip the sampler
        # stop-signal or leak either process (the finally is the backstop).
        stop_file.touch()
        try:
            injector_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print(f"[report_phase1] {condition}: injector did not exit within 60s — "
                  "will be terminated.", file=sys.stderr)
        try:
            sampler_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"[report_phase1] {condition}: gpu sampler did not exit within 30s — "
                  "will be terminated.", file=sys.stderr)
    finally:
        # Guarantee no orphaned GPU-holding children survive this condition,
        # however we got here (normal exit, sys.exit, timeout, or an unexpected
        # exception). Touch the stop-file first so the sampler can exit on its
        # own before we fall back to signalling it.
        try:
            stop_file.touch()
        except OSError:
            pass
        _terminate_all([*rank_procs, injector_proc, sampler_proc])

    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--phase0-dir", required=True, help="Directory containing phase0_summary.json")
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--target-step", type=int, default=None,
                    help="Override the auto-computed (from Phase 0's measured steps/sec) target step.")
    p.add_argument("--warmup-secs", type=float, default=60.0)
    p.add_argument("--target-rank", type=int, default=None,
                    help="Defaults to the last rank (nproc_per_node - 1).")
    p.add_argument("--nproc-per-node", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--aegis-policy-path", default="aegis.yaml")
    p.add_argument("--tiny", action="store_true", help="DEBUG ONLY.")
    p.add_argument("--heartbeat-timeout-secs", type=float, default=5.0)
    p.add_argument("--collective-timeout-secs", type=float, default=30.0)
    p.add_argument("--setup-timeout-secs", type=float, default=600.0)
    p.add_argument("--gpu-hr-cost-usd", type=float, default=DEFAULT_H100_USD_PER_HR,
                    help="test_suite.md §5.1: $2.35/hr (H100 1-yr). Override for A100 spot on SeaWulf.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    phase0_dir = Path(args.phase0_dir)

    check_phase0_gate(phase0_dir, args.skip_gate)

    target_rank = args.target_rank if args.target_rank is not None else args.nproc_per_node - 1

    if args.target_step is not None:
        target_step = args.target_step
    else:
        phase0_summary = json.loads((phase0_dir / "phase0_summary.json").read_text())
        steps_per_sec = phase0_summary.get("aegis_mean_steps_per_sec") or phase0_summary.get("control_mean_steps_per_sec")
        if not steps_per_sec:
            sys.exit("[report_phase1] Phase 0 summary has no usable steps/sec and no --target-step given.")
        target_step = suggest_target_step(steps_per_sec, args.steps, args.warmup_secs)

    trace_path = out / "trace_b1.json"
    write_trace(trace_path, target_step, target_rank)
    print(f"[report_phase1] trace: B1 on rank {target_rank} at step {target_step} -> {trace_path}")

    run_dirs = {}
    for condition in CONDITIONS:
        run_dirs[condition] = run_condition(condition, out, trace_path, args)

    manifest = {}
    fault_signals_seen = ["B1"]
    lines = ["# Phase 1 — Real B1 Fault Injection Report", ""]

    per_condition_results = {}
    for condition, run_dir in run_dirs.items():
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())

        step_log = load_step_log(run_dir / "step_log.jsonl")
        gpu_log = load_gpu_log(run_dir / "gpu_log.jsonl")
        chaos_log = load_chaos_log(run_dir / "chaos_log.jsonl")
        halt_log = load_halt_log(run_dir / "training_halt.jsonl")
        results = align_fault_events(step_log, gpu_log, chaos_log, halt_log)
        per_condition_results[condition] = results

        detection_records = read_jsonl(run_dir / "detection_log.jsonl") if condition == "aegis" else []
        epe_history_path = run_dir / "epe_history.json"
        epe_history = json.loads(epe_history_path.read_text()) if condition == "aegis" and epe_history_path.exists() else []

        lines.append(f"## Condition: {condition}")
        lines.append("")
        for r in results:
            lines.append(f"- fault_signal={r.fault_signal} target_rank={r.target_rank} "
                          f"fault_fired={r.fault_fired} job_halted={r.job_halted}")
            if r.halt_reason:
                lines.append(f"  - halt_reason: `{r.halt_reason}`")
            lines.append(f"  - recovery_gap: {r.recovery_gap} (see caveats)")
            lines.append(f"  - idle_gpus_by_util_proxy: {r.idle_gpus_by_util_proxy}")
            lines.append(f"  - wasted_gpu_hours: {r.wasted_gpu_hours}")
            for c in r.measurement_caveats:
                lines.append(f"  - caveat: {c}")

        if condition == "aegis" and detection_records:
            fire_wall_clock = chaos_log[0].wall_clock if chaos_log else None
            detect_wall_clock = detection_records[0]["wall_clock"]
            detect_latency = (detect_wall_clock - fire_wall_clock) if fire_wall_clock else None
            lines.append(f"- **real detection latency (fault -> TelemetryEvent published): "
                         f"{detect_latency:.3f}s**" if detect_latency is not None else "- detection latency: N/A")

        if condition == "aegis" and epe_history:
            for rec in epe_history:
                lines.append(f"- **real AEGIS recovery record:** tier {rec['original_tier']}->{rec['final_tier']} "
                              f"escalated={rec['escalated']} success={rec['success']} degraded={rec['degraded']} "
                              f"recovery_secs={rec['recovery_secs']} — \"{rec['message']}\"")
        lines.append("")

    lines.append("## Not built in this pass (follow-up work, not fabricated)")
    lines.append("")
    lines.append("- **B-VANILLA checkpoint-and-restart harness.** The `no_ft` condition above is the "
                  "\"no fault tolerance\" floor (real kill, no recovery attempt at all) — not a periodic-"
                  "checkpoint-and-restart baseline. Building that comparison is the recommended next step "
                  "once these AEGIS-side numbers are validated on real hardware.")
    lines.append("- **TorchFT condition.** Not run in this pass — test_suite.md's full comparison table "
                  "includes a TorchFT-wrapped condition; only `no_ft` and `aegis` are compared here.")
    lines.append("- **End-to-end recovery_gap/wasted_gpu_hours/$ saved.** Not computable in this build for "
                  "either condition — see the per-condition caveats above and "
                  "realbench/training/w1_llama7b.py's module docstring for why.")
    lines.append("")

    honesty = render_honesty_block(manifest, fault_signals_seen=fault_signals_seen)
    report = "\n".join(lines[:1]) + "\n\n" + honesty + "\n" + "\n".join(lines[1:])

    (out / "phase1_report.md").write_text(report)
    (out / "phase1_summary.json").write_text(json.dumps({
        "target_step": target_step, "target_rank": target_rank,
        "conditions_run": list(run_dirs.keys()),
        "gpu_hr_cost_usd": args.gpu_hr_cost_usd,
    }, indent=2))

    print(report)


if __name__ == "__main__":
    main()
