"""
Recovery cost tables for each system under test.

These tables encode how long each system takes to recover from each
blast-radius tier. AEGIS handles every tier optimally; the single-primitive
baselines handle their own tier and fall back to vanilla for others.

Sources:
  - AEGIS: design targets from the product spec
  - R²CCL: arXiv:2512.25059 — <1% training overhead, B0 in seconds
  - MeCeFO: neighbor-absorb ~30s (paper: 4.18% overhead target)
  - TierCheck: T1 <10s, T2 <60s, T3 <300s
  - TorchFT: elastic training, handles all tiers but slower than composed AEGIS
"""

from __future__ import annotations

from aegis.telemetry.events import BlastRadius


# AEGIS: each tier handled optimally by the composed runtime
AEGIS_RECOVERY_SECS: dict[BlastRadius, float] = {
    BlastRadius.B0: 5.0,     # R²CCL NIC migration (ms–s range)
    BlastRadius.B1: 25.0,    # MeCeFO neighbor-absorb
    BlastRadius.B2: 8.0,     # TierCheck T1 local restore
    BlastRadius.B3: 55.0,    # TierCheck T2 peer restore
    BlastRadius.B4: 270.0,   # TierCheck T3 cluster restore
}

# B-R2CCL: handles B0 only (transport layer), falls back on B1+
# Key only contains B0; adapters check membership and fall back to vanilla otherwise.
R2CCL_RECOVERY_SECS: dict[BlastRadius, float] = {
    BlastRadius.B0: 5.0,   # designed for this — NIC migration in seconds
}

# B-MECEFO: handles B1 only (compute layer), falls back on others
MECEFO_RECOVERY_SECS: dict[BlastRadius, float] = {
    BlastRadius.B1: 30.0,   # neighbor-absorb (MeCeFO paper result)
}

# B-TIERCHECK: handles B2/B3/B4 via tiered checkpoints, falls back on B0/B1
TIERCHECK_RECOVERY_SECS: dict[BlastRadius, float] = {
    BlastRadius.B2: 10.0,    # T1 local restore (<10s claim from paper)
    BlastRadius.B3: 60.0,    # T2 peer restore
    BlastRadius.B4: 300.0,   # T3 cluster restore
}

# B-TORCHFT: elastic training — handles all tiers but slower than composed AEGIS
TORCHFT_RECOVERY_SECS: dict[BlastRadius, float] = {
    BlastRadius.B0: 60.0,
    BlastRadius.B1: 45.0,
    BlastRadius.B2: 90.0,
    BlastRadius.B3: 300.0,
    BlastRadius.B4: 1800.0,
}
