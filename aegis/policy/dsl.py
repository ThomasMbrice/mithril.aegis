"""
Operator Policy DSL (§4 Layer E1).

Operators declare cost/latency tradeoffs here — how aggressively to pursue
in-tier recovery vs. escalating to a more expensive tier.

All fields have sensible defaults calibrated to H100 clusters at $2.35/GPU/hr.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OperatorPolicy:
    """
    Declare recovery cost/latency thresholds.

    Example: "never spend >$X to avoid a rollback of <Y GPU-hrs"

    The EPE reads this at routing time to decide whether to attempt
    a more expensive tier or escalate immediately.
    """

    # $/GPU/hr benchmark (H100 on-demand pricing, per §5.3 ST-3)
    gpu_hr_cost: float = 2.35

    # ---------------- B0 fast-path ----------------------------------------
    # Allow R²CCL to begin NIC migration before the FC completes classification
    # (§3.1 mitigation: transport may act in ms–s and report concurrently)
    allow_b0_fast_path: bool = True

    # ---------------- Correlation window (§3.3) ---------------------------
    # A B1 burst in the same rack within this window is re-classified to B4
    # *before* committing to neighbor-absorb.
    correlation_window_secs: float = 5.0
    # Minimum correlated node failures in a rack within the window to trigger B4
    correlation_node_threshold: int = 3

    # ---------------- MeCeFO fallback bounds (§6) -------------------------
    # After this many consecutive fallback windows on a node, emit a warning
    # recommending a full-fidelity checkpoint.
    max_consecutive_fallbacks: int = 10

    # ---------------- Economics policy -----------------------------------
    # High-level cost/correctness trade-off mode.
    # Parsed from aegis.yaml economics.policy; also settable via aegis.policy.set().
    economics_policy: str = "minimize_expected_cost"  # | "correctness_first" | "latency_first"

    # ---------------- Per-tier recovery budget (seconds) ------------------
    # Recovery must complete within this time budget.
    # Exceeding the budget is logged but does not currently abort the attempt.
    tier_budget_secs: dict[str, float] = field(default_factory=lambda: {
        "B0": 1.0,
        "B1": 30.0,
        "B2": 10.0,
        "B3": 60.0,
        "B4": 300.0,
    })
