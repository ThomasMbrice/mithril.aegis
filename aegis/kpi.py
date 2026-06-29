"""
Cost / KPI Meter (§4 Layer E3).

Measures $/GPU-hour saved per fault, per tier, vs. the checkpoint-and-restart
baseline.  This is the product's primary sales artifact (§5.3 ST-3).

Baseline assumption: without AEGIS every fault triggers full
checkpoint-and-restart, costing the tier-specific baseline recovery time
across all GPUs in the job.

The KPI number that "sells the product":
  total_dollars_saved = Σ (baseline_time_secs - recovery_time_secs) / 3600
                        × gpu_count × $/GPU/hr
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .telemetry.events import BlastRadius


# Baseline checkpoint-and-restart time per tier (seconds).
# These are conservative estimates used as the comparison point.
_BASELINE_RECOVERY_SECS: dict[BlastRadius, float] = {
    BlastRadius.B0: 60.0,     # 1 min NIC timeout + restart
    BlastRadius.B1: 900.0,    # 15 min node replacement + re-sync
    BlastRadius.B2: 600.0,    # 10 min software restart
    BlastRadius.B3: 1800.0,   # 30 min node replacement + checkpoint restore
    BlastRadius.B4: 3600.0,   # 1 hr rack/cluster recovery
}


@dataclass
class FaultKPI:
    """Per-fault KPI record."""

    epoch: int
    tier: BlastRadius
    recovery_time_secs: float
    baseline_time_secs: float
    gpu_count: int
    gpu_hr_cost: float
    gpu_hrs_saved: float = field(init=False)
    dollars_saved: float = field(init=False)

    def __post_init__(self) -> None:
        time_saved = max(0.0, self.baseline_time_secs - self.recovery_time_secs)
        self.gpu_hrs_saved = (time_saved / 3600.0) * self.gpu_count
        self.dollars_saved = self.gpu_hrs_saved * self.gpu_hr_cost


class KPIMeter:
    """
    Accumulates per-fault KPI records and summarises total savings.

    Usage:
        kpi = KPIMeter(gpu_hr_cost=2.35)
        kpi.record(epoch=42, tier=BlastRadius.B1,
                   recovery_time_secs=2.5, gpu_count=256)
        print(kpi.summary())
    """

    def __init__(self, gpu_hr_cost: float = 2.35) -> None:
        self._gpu_hr_cost = gpu_hr_cost
        self._records: list[FaultKPI] = []

    def record(
        self,
        epoch: int,
        tier: BlastRadius,
        recovery_time_secs: float,
        gpu_count: int,
    ) -> FaultKPI:
        baseline = _BASELINE_RECOVERY_SECS.get(tier, 600.0)
        kpi = FaultKPI(
            epoch=epoch,
            tier=tier,
            recovery_time_secs=recovery_time_secs,
            baseline_time_secs=baseline,
            gpu_count=gpu_count,
            gpu_hr_cost=self._gpu_hr_cost,
        )
        self._records.append(kpi)
        return kpi

    @property
    def total_dollars_saved(self) -> float:
        return sum(r.dollars_saved for r in self._records)

    @property
    def total_gpu_hrs_saved(self) -> float:
        return sum(r.gpu_hrs_saved for r in self._records)

    def summary(self) -> dict:
        """Return a structured summary suitable for the operator dashboard."""
        by_tier: dict[str, dict] = {}
        for tier in BlastRadius:
            tier_records = [r for r in self._records if r.tier == tier]
            by_tier[tier.name] = {
                "count": len(tier_records),
                "dollars_saved": sum(r.dollars_saved for r in tier_records),
                "gpu_hrs_saved": sum(r.gpu_hrs_saved for r in tier_records),
            }

        return {
            "faults_handled": len(self._records),
            "total_dollars_saved": self.total_dollars_saved,
            "total_gpu_hrs_saved": self.total_gpu_hrs_saved,
            "by_tier": by_tier,
        }
