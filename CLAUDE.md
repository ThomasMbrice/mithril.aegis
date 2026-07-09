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
| B0 | Transient NIC/link | `layers/transport.py` (R²CCL: real NIC state machine + bandwidth math, hardware-pending RDMA backend) |
| B1 | Single GPU / node death | `layers/compute.py` (MeCeFO: real torch low-rank/skip-connection/recompute math) |
| B2 | Software crash, recoverable in place | `layers/storage.py` Tier-1 (real file I/O + diff/base checkpoints) |
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
/tmp/aegis-venv/bin/pytest          # full suite (165 tests, ~25-30 s)
/tmp/aegis-venv/bin/pytest tests/unit/
/tmp/aegis-venv/bin/pytest tests/integration/
```

Test count and timing drift as the suite grows — verify with `pytest --collect-only -q | tail -1`
rather than trusting this number long-term.

The venv lives at `/tmp/aegis-venv/` (outside the external drive to avoid UTF-8 issues).
To recreate: `python3 -m venv /tmp/aegis-venv && /tmp/aegis-venv/bin/pip install -e ".[dev]"`
(now pulls in `numpy` + `torch` — real MeCeFO math needs them; CUDA is used
automatically when available, falling back to MPS/CPU on this dev machine.)

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

This table drifts — see `design.md` §8.1 for the authoritative, longer-form
version and verify against the repo before trusting either.

- **Phase 0 (complete):** UTP + FC + Epoch + EPE + layer stubs + chaos harness + IT-1–IT-7
- **Phase 1 (software-composition complete, hardware validation pending):**
  Real NIC state machine + R2CC-Balance bandwidth math (`layers/transport.py`);
  real MeCeFO tensor math via torch — low-rank SVD, skip-connection,
  selective recompute (`layers/compute.py`); real file-based differential
  checkpoints with SHA-256 verification (`layers/storage.py`); URC now
  genuinely gates B2-B4 restores via `report_epoch()`/`agree()` wired into
  the EPE (`policy/engine.py::_urc_gate`). Still missing: real NCCL shim,
  real RDMA/IB migration, real S3/Lustre backend, real A100 training-job
  reproduction of the papers' numbers, TorchFT integration — all hardware/
  infra-pending, see `eval_design.md` / `test_suite.md` §4.5 for the plan.
- **Phase 2:** E1–E4 cost/policy plane; ST-3 $/GPU-hr-saved measurement.
  `bench/` scaffolding exists (adapters, traces, cost model, sim engine) but
  most of it simulates rather than measures — see design.md §8.1.
- **Phase 3 (deferred, not started):** Learned classifier, predictive
  pre-staging, straggler tier (B-1), inference path.

## Design decisions to preserve

- **B0 fast-path autonomy:** transport layer may begin NIC migration before the FC classifies.
  The EPE logs it but does not block the fast path (`allow_b0_fast_path` in `OperatorPolicy`).
- **Correlation window:** B1 burst ≥ N nodes in same rack within W seconds → re-classify B4
  *before* committing neighbor-absorb. Tunable via `OperatorPolicy`.
- **Fidelity flag:** any checkpoint written during MeCeFO fallback must carry `fidelity_flag=True`.
  No silent approximation (§3.4 of design.md).
- **Epoch tagging:** every classified fault increments the epoch; state from different epochs
  must never be mixed in a checkpoint.
