"""
Report generation for the AEGIS benchmark evaluation suite.

Produces ASCII comparison tables and prints the headline $/GPU-hr saved
metric. This is the auto-generated artifact that goes in the deck.
"""

from __future__ import annotations

from bench.metrics.compute import compare_systems
from bench.sim.engine import SimulationResult


def generate_comparison_matrix(
    results: dict[str, SimulationResult],
    baseline_name: str = "B-VANILLA",
) -> str:
    """
    Generate an ASCII comparison matrix.

    Format:
        System    | $/GPU-hr wasted | Goodput | Recovery (s) | vs B-VANILLA savings ($)
        AEGIS     | ...             | ...     | ...          | ...
        B-VANILLA | ...             | ...     | (baseline)   | —
        ...

    Args:
        results: Mapping of system_name → SimulationResult.
        baseline_name: Name of the baseline to compare against.

    Returns:
        Multi-line ASCII table string.
    """
    if not results:
        return "(no results)"

    rows = compare_systems(results, baseline_name=baseline_name)

    # Column headers
    col_system = "System"
    col_dollars = "$/GPU-hr wasted"
    col_goodput = "Goodput"
    col_recovery = "Recovery (s)"
    col_savings = f"vs {baseline_name} savings ($)"

    # Compute column widths
    w_system = max(len(col_system), max(len(r["system_name"]) for r in rows))
    w_dollars = max(len(col_dollars), 15)
    w_goodput = max(len(col_goodput), 8)
    w_recovery = max(len(col_recovery), 12)
    w_savings = max(len(col_savings), 25)

    def fmt_row(system, dollars, goodput, recovery, savings):
        return (
            f"  {system:<{w_system}} | "
            f"{dollars:>{w_dollars}} | "
            f"{goodput:>{w_goodput}} | "
            f"{recovery:>{w_recovery}} | "
            f"{savings:>{w_savings}}"
        )

    separator = (
        "  "
        + "-" * w_system
        + "-+-"
        + "-" * w_dollars
        + "-+-"
        + "-" * w_goodput
        + "-+-"
        + "-" * w_recovery
        + "-+-"
        + "-" * w_savings
    )

    lines = []
    lines.append("")
    lines.append("  AEGIS Benchmark Comparison Matrix")
    lines.append("  " + "=" * (w_system + w_dollars + w_goodput + w_recovery + w_savings + 16))
    lines.append(
        fmt_row(col_system, col_dollars, col_goodput, col_recovery, col_savings)
    )
    lines.append(separator)

    for row in rows:
        name = row["system_name"]
        dollars = f"${row['total_dollars_wasted']:.4f}"
        goodput = f"{row['goodput']:.3f}"
        recovery = f"{row['total_recovery_secs']:.1f}"

        if name == baseline_name:
            savings_str = "(baseline)"
        else:
            saved = row.get("dollars_saved", 0.0)
            savings_str = f"${saved:.4f}" if saved >= 0 else f"-${abs(saved):.4f}"

        lines.append(fmt_row(name, dollars, goodput, recovery, savings_str))

    lines.append("")
    return "\n".join(lines)


def print_report(
    results: dict[str, SimulationResult],
    baseline_name: str = "B-VANILLA",
) -> None:
    """
    Print the full comparison report to stdout.

    Includes:
    - The comparison matrix
    - Per-system headline numbers
    - AEGIS savings vs. each baseline

    Args:
        results: Mapping of system_name → SimulationResult.
        baseline_name: Name of the baseline to compare against.
    """
    print(generate_comparison_matrix(results, baseline_name=baseline_name))

    baseline = results.get(baseline_name)
    if baseline is None:
        print(f"  Warning: baseline '{baseline_name}' not in results")
        return

    print("  AEGIS Savings Summary")
    print("  " + "-" * 50)

    aegis = results.get("AEGIS")
    if aegis is None:
        print("  AEGIS not in results")
        return

    for name, result in sorted(results.items()):
        if name == "AEGIS":
            continue
        dollars_saved = result.total_dollars_wasted - aegis.total_dollars_wasted
        pct = 0.0
        if result.total_recovery_secs > 0:
            pct = (
                (result.total_recovery_secs - aegis.total_recovery_secs)
                / result.total_recovery_secs
                * 100.0
            )
        print(
            f"  AEGIS vs {name:20s}: "
            f"${dollars_saved:.4f} saved "
            f"({pct:.1f}% faster recovery)"
        )

    print()
