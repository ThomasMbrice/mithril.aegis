"""
Single source of truth for the disclaimer text every real-cluster report
must carry, per the plan's honesty/labeling requirements — so the text
lives in exactly one place and can't drift between phase0_report.md and
phase1_report.md over time.

Each bullet below is enforced at the source (field naming, structural gates,
manifest fields), not just documented — see the docstrings of
``realbench.collector.align``, ``chaos_inject.real_injector``, and
``realbench.training.w1_llama7b`` for where each fact actually comes from.
"""

from __future__ import annotations

from realbench.collector.align import NO_RESUME_CAVEAT, NVIDIA_SMI_FALLBACK_CAVEAT

PROXY_TENSOR_CAVEAT = (
    "MeCeFO neighbor-absorb (aegis/layers/compute.py's TorchMeCeFOBackend) performs real "
    "low-rank/skip-connection/selective-recompute torch math and is genuinely measured, but "
    "operates on synthetic proxy tensors, not the dead rank's actual live gradients/activations "
    "— and performs no real cross-rank state transfer. recovery_secs reflects real compute time "
    "for the proxy math, not a claim that the model's training state was restored with full "
    "fidelity."
)

SINGLE_NODE_CAVEAT = (
    "This allocation is single-node, 8xA100 (or fewer, for local smoke tests). B2/B3/B4 "
    "(node replacement, rack outage) are not exercised — those fault classes require a "
    "multi-node allocation this deliverable does not have."
)

B0_SIM_ONLY_CAVEAT = (
    "B0 (NIC/link failure) is sim-validated only on this allocation — single NIC/plane "
    "topology per test_suite.md §4.5.5, confirmed before this harness was built. See "
    "chaos_inject/real_injector.py's B0_HARDWARE_VALIDATED gate and its skip records."
)

TINY_DEBUG_CAVEAT = (
    "**THIS RUN USED --tiny DEBUG MODEL DIMS, NOT REAL W1 (LLaMA-7B).** It exercised the "
    "harness's plumbing only (aegis wiring, logging, fault delivery) — none of the numbers "
    "in this report are a real W1 measurement. Re-run without --tiny for a reportable result."
)


def render_honesty_block(
    manifest: dict,
    fault_signals_seen: list[str] | None = None,
) -> str:
    """
    Render the shared markdown disclaimer block, tailored to what this
    particular run actually did (read from ``run_manifest.json``, plus
    which fault signals appeared in the chaos log, if any).
    """
    fault_signals_seen = fault_signals_seen or []
    lines = ["## Honesty / Scope Notes", ""]

    if manifest.get("tiny_debug_run"):
        lines.append(f"- {TINY_DEBUG_CAVEAT}")

    parallelism = manifest.get("parallelism", "unknown")
    if parallelism == "DP(FSDP)":
        lines.append(
            "- **DP(FSDP)-only, not DP+PP.** W1's config table (bench/workloads/configs.py) "
            "specifies DP+PP (MeCeFO's own testbed); this run used FSDP-sharded data "
            "parallelism only — real GPUs, real NCCL collectives, real steady-state overhead, "
            "but not a reproduction of the exact DP+PP topology MeCeFO's 4.18% number was "
            "measured on."
        )
    elif "DDP" in parallelism or "smoke-test" in parallelism:
        lines.append(
            f"- **parallelism={parallelism}** — local smoke-test path (CPU/gloo), never used "
            "for a real W1 run; see realbench/training/w1_llama7b.py."
        )

    lines.append(f"- {PROXY_TENSOR_CAVEAT}")
    lines.append(f"- {SINGLE_NODE_CAVEAT}")

    if "B0" in fault_signals_seen or manifest.get("world_size", 0) >= 1:
        # Always state the B0 limitation explicitly — even if no B0 trace
        # entry ran this time, a reader inspecting only this report should
        # still know B0 isn't hardware-validated on this allocation.
        lines.append(f"- {B0_SIM_ONLY_CAVEAT}")

    if any(s == "B1" for s in fault_signals_seen):
        lines.append(f"- **nvidia-smi fallback:** {NVIDIA_SMI_FALLBACK_CAVEAT}")
        lines.append(f"- **No end-to-end recovery measurement (yet):** {NO_RESUME_CAVEAT}")

    lines.append("")
    return "\n".join(lines)
