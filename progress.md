# AEGIS — Progress Log

## How to run the suite and get the "AEGIS beats baseline" report

### 1. Set up the venv (if not already done)

```bash
python3 -m venv /tmp/aegis-venv
/tmp/aegis-venv/bin/pip install -e ".[dev]"
```

Pulls in `numpy` + `torch` (Phase 1's real MeCeFO math needs them). CUDA is
used automatically if present; falls back to MPS/CPU otherwise.

### 2. Run the test suite (sanity check everything still works)

```bash
/tmp/aegis-venv/bin/pytest -q
```

Expect all tests to pass (175 as of this writing — check with
`pytest --collect-only -q | tail -1` since this number drifts).

### 3. Generate the $/GPU-hr-saved report (the "beats baseline" artifact)

```bash
/tmp/aegis-venv/bin/python -m bench.run_st3
```

This runs the full 6-system baseline matrix (AEGIS, B-VANILLA, B-R2CCL,
B-MECEFO, B-TIERCHECK, B-TORCHFT) against the FT-PRODUCTION fault trace
(realistic mixed failure profile, seed=2024) across all three standard
workloads (W1: 8 GPUs, W2: 32 GPUs, W3: 64 GPUs), and:

- prints the comparison matrix + headline savings to stdout
- writes the same report to `bench/reports/ft_production_report.md`

**Read the headline numbers from the top of that file** — the
`## Headline: AEGIS vs B-VANILLA on FT-PRODUCTION` section gives one line
per workload, e.g.:

```
- W2 (32 GPUs): AEGIS saved $116.40 vs B-VANILLA (49.53 GPU-hrs,
  92.5% faster recovery) over 14 faults.
```

Below that, each workload section has the full comparison matrix (all 6
systems, sorted by cost) so you can see AEGIS beats every baseline, not
just B-VANILLA.

To re-run and refresh the report after code changes:

```bash
/tmp/aegis-venv/bin/python -m bench.run_st3
cat bench/reports/ft_production_report.md
```

### 4. Live KPI from a running job (not the sim report — the real runtime)

Inside a process that has called `aegis.init()`:

```python
import aegis
aegis.init()
# ... faults happen ...
print(aegis.dashboard())            # rendered per-tier + $/GPU-hr text report
report = aegis.dashboard(fmt="json")  # same data as a dict
```

This is a different, smaller number than the `bench/run_st3.py` report —
it reflects real measured recovery time from this specific process (on a
dev machine with no GPU cluster, that's proxy-scale CPU/MPS timing, not
representative of production cost). See the caveat below.

---

## Important caveat on what these numbers mean

Both reports are honest about what they are and aren't — read the caveat
text baked into `bench/reports/ft_production_report.md` and into
`aegis/kpi.py`'s docstring before quoting a number externally:

- **`bench/run_st3.py`** — only the AEGIS adapter drives the real EPE
  *routing decision*; all six systems' recovery-*time* inputs come from
  `bench/sim/cost_model.py`'s target/design numbers (matching the papers'
  claims), not hardware measurements.
- **`aegis.dashboard()` / live `KPIMeter`** — real measured `recovery_secs`
  from the actual runtime, but on this dev machine that's CPU/MPS proxy
  timing (sub-millisecond), not a real GPU recovery duration.

Both are legitimate, useful numbers for validating that the composed
routing logic and economics math are directionally correct — neither is
the hardware-validated number `design.md` §5.3 ST-3 ultimately wants. That
needs the real A100/IB cluster run described in `eval_design.md` /
`test_suite.md` §4.5, which hasn't happened yet.

---

## Phase status (see `design.md` §8.1 for full detail, `CLAUDE.md` for the short version)

- **Phase 0** — done: UTP, FC, epoch service, EPE, chaos-inject harness, IT-1–IT-7.
- **Phase 1** — done at the software-composition level: real NIC
  state-machine + bandwidth math (transport), real torch MeCeFO math
  (compute), real file-based differential checkpoints (storage), URC
  wired into the EPE's routing path. Hardware-pending: real NCCL shim,
  real RDMA/IB migration, real S3/Lustre backend, real A100 paper
  reproductions, TorchFT integration.
- **Phase 2** — done at the software/simulation level: KPI meter wired
  into the EPE (E3), text/JSON operator dashboard added (E4, no daemon),
  ST-3 report generated and saved (`bench/run_st3.py` →
  `bench/reports/ft_production_report.md`).
- **Phase 3** — not started (deferred per user decision): learned
  classifier, predictive pre-staging, straggler tier, inference path.

Next gate for a credible $/GPU-hr number: real A100/IB cluster validation
per `eval_design.md` / `test_suite.md` §4.5 — not more software work.
