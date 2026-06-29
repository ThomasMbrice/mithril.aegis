aegis/

    telemetry/events.py      — TelemetryEvent schema, FaultSignal enum, BlastRadius enum
    telemetry/bus.py         — UnifiedTelemetryPlane: asyncio pub/sub, non-blocking publish
    telemetry/sensors.py     — SensorBase + SyntheticSensor
    classifier/classifier.py — FailureClassifier: rule-based signal → B0–B4
    epoch/service.py         — FaultEpochService: thread-safe monotonic counter
    layers/base.py           — RecoveryLayer ABC + RecoveryResult
    layers/transport.py      — B0 stub (R²CCL): NIC migration, multi-NIC prerequisite check
    layers/compute.py        — B1 stub (MeCeFO): neighbor-absorb, degraded/fidelity flag
    layers/storage.py        — B2/B3/B4 stub (TierCheck): 3-tier checkpoints + metadata
    consensus/urc.py         — UnifiedRecoveryConsensus: epoch-gated surviving-rank reduction
    policy/dsl.py            — OperatorPolicy: correlation window, cost thresholds
    policy/engine.py         — EscalationPolicyEngine: the integration keystone
    kpi.py                   — KPIMeter: $/GPU-hr-saved vs. checkpoint-restart baseline
    runtime.py               — AegisRuntime: async context manager wiring everything

chaos_inject/  — ChaosHarness, FaultSpec, BurstSpec

tests/unit/        — UT-A1 (UTP), UT-A2 (classifier), UT-A3 (epoch), EPE routing
tests/integration/ — IT-1 NIC flap · IT-2 node death · IT-3 B1→B4 correlation
                     IT-4 fidelity flag · IT-5 concurrent B0+B1 · IT-6 invariant fuzz

Key design properties verified by tests

- One-directional escalation invariant — proven by IT-6 (60-fault fuzz), never violated
- B4 never misclassified as B1 — explicit parametrized safety test in UT-A2
- Correlation window — 3+ correlated B1s in same rack within W → re-classified B4 before
  committing to neighbor-absorb
- Fidelity flag — MeCeFO fallback always sets degraded=True; no silent approximation
- Epoch monotonicity — 20 concurrent threads × 100 increments, zero duplicates

Run with: /tmp/aegis-venv/bin/pytest
