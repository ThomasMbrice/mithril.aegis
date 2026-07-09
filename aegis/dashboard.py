"""
Operator dashboard (§4 Layer E4) — text/JSON report, no daemon.

ui.md §6 explicitly rejects a daemon/sidecar for AEGIS ("premature for
MVP... a separate daemon adds an IPC boundary and a second failure domain
for no MVP benefit"). Consistent with that, this is an in-process
introspection surface only — call it from the same Python process that
called ``aegis.init()``, the same way ``aegis.status()``/``aegis.explain()``
already work. It is not a live HTTP/JSON endpoint for external polling;
if that's needed later it's a deliberate, separate addition (see
design.md §8.1 Phase 2 status for the scope decision).

Usage:
    import aegis
    aegis.init()
    ...
    print(aegis.dashboard())            # rendered ASCII report
    report = aegis.dashboard(fmt="json")  # structured dict, same data
"""

from __future__ import annotations

from typing import Any

from aegis.runtime import AegisRuntime
from aegis.telemetry.events import BlastRadius


def build_report(rt: AegisRuntime) -> dict[str, Any]:
    """
    Assemble a structured (JSON-serialisable) snapshot of runtime state.

    Reuses ``EscalationPolicyEngine.escalation_audit()`` and
    ``KPIMeter.summary()`` — this is a view over existing data, not a new
    measurement source.
    """
    audit = rt.epe.escalation_audit()
    kpi = rt.kpi.summary()

    by_tier: dict[str, dict[str, int]] = {
        tier.name: {"count": 0, "escalated": 0, "degraded": 0, "failed": 0}
        for tier in BlastRadius
    }
    for record in audit:
        row = by_tier[record["final_tier"]]
        row["count"] += 1
        if record["escalated"]:
            row["escalated"] += 1
        if record["degraded"]:
            row["degraded"] += 1
        if not record["success"]:
            row["failed"] += 1

    violations = [r for r in audit if not r["escalation_valid"]]

    return {
        "current_epoch": rt.epoch_service.current(),
        "faults_processed": len(audit),
        "by_tier": by_tier,
        "escalation_invariant_violations": len(violations),
        "kpi": kpi,
        "recent_faults": audit[-10:],
    }


def render_text(report: dict[str, Any]) -> str:
    """Render a build_report() dict as an ASCII report."""
    lines: list[str] = []
    lines.append("")
    lines.append("  AEGIS Operator Dashboard")
    lines.append("  " + "=" * 40)
    lines.append(f"  Current epoch:      {report['current_epoch']}")
    lines.append(f"  Faults processed:   {report['faults_processed']}")
    violations = report["escalation_invariant_violations"]
    flag = "  <-- INVESTIGATE" if violations else ""
    lines.append(f"  Invariant violations: {violations}{flag}")
    lines.append("")

    lines.append("  Per-tier breakdown")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'Tier':<6} {'Count':>7} {'Escalated':>10} {'Degraded':>9} {'Failed':>7}")
    for tier_name, row in report["by_tier"].items():
        lines.append(
            f"  {tier_name:<6} {row['count']:>7} {row['escalated']:>10} "
            f"{row['degraded']:>9} {row['failed']:>7}"
        )
    lines.append("")

    kpi = report["kpi"]
    lines.append("  KPI summary (§4 Layer E3 — real measured recovery_secs;")
    lines.append("  see aegis/kpi.py docstring for the dev-hardware caveat)")
    lines.append("  " + "-" * 40)
    lines.append(f"  Total $ saved:      ${kpi['total_dollars_saved']:.6f}")
    lines.append(f"  Total GPU-hrs saved: {kpi['total_gpu_hrs_saved']:.6f}")
    for tier_name, tier_kpi in kpi["by_tier"].items():
        if tier_kpi["count"] == 0:
            continue
        lines.append(
            f"    {tier_name}: {tier_kpi['count']} faults, "
            f"${tier_kpi['dollars_saved']:.6f} saved"
        )
    lines.append("")

    return "\n".join(lines)
