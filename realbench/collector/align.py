"""
§4.5.4 alignment — turns the three real logs (step-log, gpu-log, chaos-log)
into the recovery_gap / idle_GPUs / wasted_GPU_hours numbers the doc's
worked example (§4.5.6) computes.

Pure functions only — no I/O side effects beyond reading the three logs —
so this is independently unit-testable without a cluster.

Chaos-log schema (written by ``chaos_inject.real_injector``):
    {"wall_clock": ..., "step": ..., "fault_fired": bool, "fault_signal": "B1"|"B0",
     "target_rank": int, "reason": str | null}
``fault_fired=False`` marks a structurally-skipped fault (currently only B0,
see ``chaos_inject/real_injector.py``'s ``B0_HARDWARE_VALIDATED`` gate) —
``align_fault_events`` still returns a result row for it so the report can
show *why* no B0 event was measured, rather than silently omitting the row.

Training-halt schema (written by ``realbench.training.w1_llama7b`` when a
collective hangs past ``--collective-timeout-secs``):
    {"wall_clock": ..., "rank": ..., "last_completed_step": ..., "reason": ..., "detail": ...}

IMPORTANT — this build's real fault handling does not reconfigure the
``torch.distributed`` process group after a B1 kill (see
``realbench/training/w1_llama7b.py``'s module docstring), so in practice a
real B1 fault today produces a ``training_halt`` record, not a resumed
step-log. ``compute_recovery_gap`` returns ``None`` when no resuming step is
found — callers must not fabricate a number, and must surface
``measurement_caveats``/``job_halted`` instead. See
``realbench/phase1_per_tier/report_phase1.py`` for how this is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from realbench.training.step_log import StepLogReader, StepRecord, read_jsonl

# nvidia-smi fallback (§4.5.4a) has no tensor-active signal — util% can be
# nonzero during a spin-wait on a hung collective. This caveat must be
# attached to every idle-GPU measurement, never silently dropped.
NVIDIA_SMI_FALLBACK_CAVEAT = (
    "idle_gpus_by_util_proxy is measured via nvidia-smi GPU-util%, not DCGM "
    "tensor-core-active. Util% can be nonzero during a spin-wait on a hung "
    "collective, so this is a proxy, not the tensor-active signal test_suite.md "
    "§4.5.4 specifies — treat as a rough indicator, not a precise measurement."
)

NO_RESUME_CAVEAT = (
    "No resuming step-log record was found after this fault. This build's "
    "ComputeLayer.recover() does not reconfigure the process group, so the "
    "training job halted rather than resumed (see training_halt.jsonl) — "
    "recovery_gap/wasted_gpu_hours cannot be computed end-to-end for this event."
)


@dataclass(frozen=True)
class GpuSample:
    wall_clock: float
    gpu_index: int
    gpu_util_pct: float
    mem_used_mib: float
    power_draw_w: float | None


@dataclass(frozen=True)
class ChaosFireRecord:
    wall_clock: float
    step: int
    fault_fired: bool
    fault_signal: str
    target_rank: int | None
    reason: str | None = None


@dataclass(frozen=True)
class HaltRecord:
    wall_clock: float
    rank: int
    last_completed_step: int
    reason: str
    detail: str


@dataclass(frozen=True)
class FaultAlignmentResult:
    fault_signal: str
    target_rank: int | None
    t_fault: float
    fault_fired: bool
    t_resume: float | None
    recovery_gap: float | None
    idle_gpus_by_util_proxy: int | None
    wasted_gpu_hours: float | None
    job_halted: bool
    halt_reason: str | None
    measurement_caveats: list[str] = field(default_factory=list)


def load_step_log(path: str | Path) -> list[StepRecord]:
    return StepLogReader(path).read_all()


def load_gpu_log(path: str | Path) -> list[GpuSample]:
    return [
        GpuSample(
            wall_clock=r["wall_clock"],
            gpu_index=r["gpu_index"],
            gpu_util_pct=r["gpu_util_pct"],
            mem_used_mib=r["mem_used_mib"],
            power_draw_w=r.get("power_draw_w"),
        )
        for r in read_jsonl(path)
        if "gpu_index" in r  # skip sample_error rows
    ]


def load_chaos_log(path: str | Path) -> list[ChaosFireRecord]:
    return [
        ChaosFireRecord(
            wall_clock=r["wall_clock"],
            step=r["step"],
            fault_fired=r["fault_fired"],
            fault_signal=r["fault_signal"],
            target_rank=r.get("target_rank"),
            reason=r.get("reason"),
        )
        for r in read_jsonl(path)
    ]


def load_halt_log(path: str | Path) -> list[HaltRecord]:
    return [
        HaltRecord(
            wall_clock=r["wall_clock"],
            rank=r["rank"],
            last_completed_step=r["last_completed_step"],
            reason=r["reason"],
            detail=r["detail"],
        )
        for r in read_jsonl(path)
    ]


def compute_recovery_gap(step_log: list[StepRecord], t_fault: float) -> tuple[float | None, float | None]:
    """
    Returns ``(recovery_gap, t_resume)``. ``(None, None)`` if no step-log
    record after ``t_fault`` is found (job halted instead of resuming).
    """
    for rec in step_log:
        if rec.wall_clock > t_fault:
            return rec.wall_clock - t_fault, rec.wall_clock
    return None, None


def compute_idle_gpus_by_util_proxy(
    gpu_log: list[GpuSample], t_fault: float, t_resume: float, util_threshold: float = 5.0
) -> int:
    """
    Count GPUs whose util% stayed below ``util_threshold`` for effectively
    the whole ``[t_fault, t_resume]`` window. util-only proxy — see
    ``NVIDIA_SMI_FALLBACK_CAVEAT``.
    """
    by_gpu: dict[int, list[float]] = {}
    for s in gpu_log:
        if t_fault <= s.wall_clock <= t_resume:
            by_gpu.setdefault(s.gpu_index, []).append(s.gpu_util_pct)
    return sum(1 for utils in by_gpu.values() if utils and max(utils) < util_threshold)


def compute_wasted_gpu_hours(recovery_gap: float, idle_gpus: int) -> float:
    """Direct transcription of §4.5.4's formula."""
    return recovery_gap * idle_gpus / 3600.0


def align_fault_events(
    step_log: list[StepRecord],
    gpu_log: list[GpuSample],
    chaos_log: list[ChaosFireRecord],
    halt_log: list[HaltRecord] | None = None,
    util_threshold: float = 5.0,
) -> list[FaultAlignmentResult]:
    halt_log = halt_log or []
    results = []
    for fire in chaos_log:
        caveats: list[str] = []

        if not fire.fault_fired:
            results.append(FaultAlignmentResult(
                fault_signal=fire.fault_signal,
                target_rank=fire.target_rank,
                t_fault=fire.wall_clock,
                fault_fired=False,
                t_resume=None,
                recovery_gap=None,
                idle_gpus_by_util_proxy=None,
                wasted_gpu_hours=None,
                job_halted=False,
                halt_reason=None,
                measurement_caveats=[fire.reason or "fault not fired — see chaos log 'reason' field"],
            ))
            continue

        recovery_gap, t_resume = compute_recovery_gap(step_log, fire.wall_clock)

        job_halted = False
        halt_reason = None
        halted_after_fault = [h for h in halt_log if h.wall_clock >= fire.wall_clock]
        if halted_after_fault:
            job_halted = True
            halt_reason = halted_after_fault[0].detail

        idle_gpus = None
        wasted_hours = None
        if recovery_gap is not None and t_resume is not None:
            idle_gpus = compute_idle_gpus_by_util_proxy(gpu_log, fire.wall_clock, t_resume, util_threshold)
            wasted_hours = compute_wasted_gpu_hours(recovery_gap, idle_gpus)
            caveats.append(NVIDIA_SMI_FALLBACK_CAVEAT)
        else:
            caveats.append(NO_RESUME_CAVEAT)

        results.append(FaultAlignmentResult(
            fault_signal=fire.fault_signal,
            target_rank=fire.target_rank,
            t_fault=fire.wall_clock,
            fault_fired=True,
            t_resume=t_resume,
            recovery_gap=recovery_gap,
            idle_gpus_by_util_proxy=idle_gpus,
            wasted_gpu_hours=wasted_hours,
            job_halted=job_halted,
            halt_reason=halt_reason,
            measurement_caveats=caveats,
        ))

    return results
