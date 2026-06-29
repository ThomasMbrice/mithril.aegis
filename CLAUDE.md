# AEGIS — Claude Context

## What this is

Blast-radius-aware fault-tolerance runtime for LLM training & serving.
Composes R²CCL (transport), MeCeFO (compute), and TierCheck (storage) under a unified
controller. Full design in `design.md`.

## The one invariant that must never break

**Faults only escalate upward (B0 → B4), never downward.**
Every test in `tests/integration/test_it6_escalation_invariant.py` guards this.
If you touch `aegis/policy/engine.py:_route()`, run IT-6 before anything else.

## Blast-radius tiers

| Tier | Fault class | Owning layer |
|------|-------------|--------------|
| B0 | Transient NIC/link | `layers/transport.py` (R²CCL stub) |
| B1 | Single GPU / node death | `layers/compute.py` (MeCeFO stub) |
| B2 | Software crash, recoverable in place | `layers/storage.py` Tier-1 |
| B3 | Node-level state loss | `layers/storage.py` Tier-2 |
| B4 | Rack / cluster outage | `layers/storage.py` Tier-3 |

## Project layout

```
aegis/          Core library — do not import from chaos_inject or tests
chaos_inject/   Fault injection harness — imports aegis only
tests/          pytest suite — imports both; never imported by source
design.md       Authoritative engineering spec — read this for context
```

## Running tests

```bash
/tmp/aegis-venv/bin/pytest          # full suite (59 tests, ~3 s)
/tmp/aegis-venv/bin/pytest tests/unit/
/tmp/aegis-venv/bin/pytest tests/integration/
```

The venv lives at `/tmp/aegis-venv/` (outside the external drive to avoid UTF-8 issues).
To recreate: `python3 -m venv /tmp/aegis-venv && /tmp/aegis-venv/bin/pip install -e ".[dev]"`

## Key files

| File | Role |
|------|------|
| `aegis/policy/engine.py` | EPE — the integration keystone; routes every fault |
| `aegis/telemetry/bus.py` | UTP — the single event bus; all layers publish here |
| `aegis/epoch/service.py` | Fault epoch counter — consistency backbone for URC |
| `aegis/consensus/urc.py` | Cross-layer recovery consensus (highest research risk) |
| `aegis/kpi.py` | $/GPU-hr-saved meter — the product's primary sales artifact |
| `chaos_inject/harness.py` | ChaosHarness — drives all tests via UTP injection |

## Phase status

- **Phase 0 (complete):** UTP + FC + Epoch + EPE stub + layer stubs + chaos harness + IT-1–IT-6
- **Phase 1 (next):** Wire real R²CCL / MeCeFO / TierCheck primitives under EPE; build URC properly
- **Phase 2:** E1–E4 cost/policy plane; ST-3 $/GPU-hr-saved measurement
- **Phase 3:** Learned classifier, predictive pre-staging, straggler tier (B-1), inference path

## Design decisions to preserve

- **B0 fast-path autonomy:** transport layer may begin NIC migration before the FC classifies.
  The EPE logs it but does not block the fast path (`allow_b0_fast_path` in `OperatorPolicy`).
- **Correlation window:** B1 burst ≥ N nodes in same rack within W seconds → re-classify B4
  *before* committing neighbor-absorb. Tunable via `OperatorPolicy`.
- **Fidelity flag:** any checkpoint written during MeCeFO fallback must carry `fidelity_flag=True`.
  No silent approximation (§3.4 of design.md).
- **Epoch tagging:** every classified fault increments the epoch; state from different epochs
  must never be mixed in a checkpoint.
