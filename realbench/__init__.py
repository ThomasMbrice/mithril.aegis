"""
realbench — real A100/IB cluster validation harness (SeaWulf, W1).

Counterpart to ``bench/`` (the simulator).  Nothing here runs through
``bench/sim/engine.py``'s cost model — every number this package produces
is a real wall-clock/GPU measurement from an actual training job on real
hardware, per ``test_suite.md`` §4.5.  Keeping this a separate top-level
package (rather than folding it into ``bench/``) keeps the sim/real line
the rest of the project is careful to maintain (see ``aegis/kpi.py``'s
own docstring on the same distinction).

Layout:
    training/    real W1 (LLaMA-7B) training entrypoint + shared step-log format
    sensors/     real fault sensors (only ``RankHeartbeatSensor`` for B1 today)
    collector/   nvidia-smi sampler + §4.5.4 alignment/cost computation
    phase0_trust_anchor/   steady-state overhead vs. paper targets, no faults
    phase1_per_tier/       real B1 fault injection + recovery-cost report
    slurm/       sbatch scripts (user fills in partition/account and submits)

Phase 0 must pass before Phase 1 is run — see ``test_suite.md`` §7/§8.
"""

from __future__ import annotations
