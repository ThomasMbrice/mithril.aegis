"""
ST-3 — $/GPU-hour-saved headline measurement (design.md §5.3, test_suite.md §5.1).

Runs the full baseline matrix (AEGIS + 5 baselines) against FT-PRODUCTION —
the realistic mixed-fault trace the $/GPU-hr headline is computed on —
across all three standard workloads (W1/W2/W3), and writes the comparison
report to bench/reports/ft_production_report.md.

This exercises infrastructure that already existed (bench/runner,
bench/sim, bench/report) — the AEGIS adapter drives the real EPE routing
decision via the real AegisRuntime; the other five adapters and AEGIS's
own recovery-time numbers come from bench/sim/cost_model.py's *target*
cost tables, not hardware measurement. See design.md §8.1 and
bench/sim/cost_model.py's docstring for what that does and doesn't prove.

Usage:
    /tmp/aegis-venv/bin/python -m bench.run_st3
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from bench.metrics.compute import compute_savings_vs_baseline
from bench.report.generate import generate_comparison_matrix
from bench.runner.run import run_benchmark
from bench.systems.aegis_adapter import AegisAdapter
from bench.systems.mecefo import MeCeFOAdapter
from bench.systems.r2ccl import R2CCLAdapter
from bench.systems.tiercheck import TierCheckAdapter
from bench.systems.torchft import TorchFTAdapter
from bench.systems.vanilla import VanillaAdapter
from bench.traces.mixed import make_ft_production
from bench.workloads.configs import W1, W2, W3

GPU_HR_COST = 2.35
TRACE_TOTAL_STEPS = 200_000  # matches make_ft_production()'s internal total_steps


def _systems() -> list:
    return [
        AegisAdapter(),
        VanillaAdapter(),
        R2CCLAdapter(),
        MeCeFOAdapter(),
        TierCheckAdapter(),
        TorchFTAdapter(),
    ]


CAVEAT = (
    "**Caveat (see design.md §8.1):** AEGIS's recovery-time numbers here "
    "come from the real EPE routing decision (via `AegisAdapter` driving "
    "the actual `AegisRuntime`) combined with `bench/sim/cost_model.py`'s "
    "*target* per-tier recovery times — not hardware-measured durations. "
    "The other five systems (B-VANILLA, B-R2CCL, B-MECEFO, B-TIERCHECK, "
    "B-TORCHFT) are entirely simulated cost-table baselines, not real "
    "integrations. This is the pre-hardware-validation, small-scale "
    "number described in test_suite.md §8 Phase 0-2; real A100/IB cluster "
    "validation (test_suite.md §4.5, eval_design.md) is the next gate "
    "before this number can be trusted beyond \"the EPE routes correctly "
    "and the composed-tier story is economically plausible.\""
)


async def _run_workload(workload) -> tuple[dict, str]:
    trace = make_ft_production(seed=2024)
    results = await run_benchmark(
        systems=_systems(),
        workload=workload,
        trace=trace,
        gpu_hr_cost=GPU_HR_COST,
        total_steps=TRACE_TOTAL_STEPS,
    )
    matrix = generate_comparison_matrix(results, baseline_name="B-VANILLA")
    return results, matrix


async def main() -> str:
    lines: list[str] = []
    lines.append("# ST-3 — $/GPU-hour-saved report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Trace: FT-PRODUCTION (seed=2024)")
    lines.append(f"GPU cost: ${GPU_HR_COST}/GPU-hr")
    lines.append("")
    lines.append(CAVEAT)
    lines.append("")

    headline_lines: list[str] = []

    for workload in (W1, W2, W3):
        results, matrix = await _run_workload(workload)

        lines.append(f"## Workload {workload.name} — {workload.model_name} "
                     f"({workload.gpu_count} GPUs, {workload.parallelism})")
        lines.append("")
        lines.append("```")
        lines.append(matrix)
        lines.append("```")
        lines.append("")

        aegis_result = results["AEGIS"]
        vanilla_result = results["B-VANILLA"]
        savings = compute_savings_vs_baseline(aegis_result, vanilla_result)
        headline_lines.append(
            f"- **{workload.name}** ({workload.gpu_count} GPUs): "
            f"AEGIS saved **${savings['dollars_saved']:.2f}** vs B-VANILLA "
            f"({savings['gpu_hours_saved']:.2f} GPU-hrs, "
            f"{savings['recovery_time_reduction_pct']:.1f}% faster recovery) "
            f"over {len(aegis_result.fault_outcomes)} faults."
        )

    report = "\n".join(lines)

    headline = "## Headline: AEGIS vs B-VANILLA on FT-PRODUCTION\n\n" + "\n".join(headline_lines) + "\n"
    report = report.replace(
        CAVEAT, CAVEAT + "\n\n" + headline, 1
    )

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ft_production_report.md"
    out_path.write_text(report)

    print(report)
    print(f"\nReport written to {out_path}")
    return report


if __name__ == "__main__":
    asyncio.run(main())
