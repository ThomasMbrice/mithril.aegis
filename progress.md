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

## Real-cluster harness (`realbench/` + `chaos_inject/real_injector.py`)

Built and smoke-tested locally (CPU/gloo, `--tiny` toy model, world_size up
to 2) — **not yet run on SeaWulf**. Confirmed for this allocation: 8xA100
single node, single NIC/plane (B0 stays sim-only per §4.5.5), no confirmed
DCGM (nvidia-smi fallback only, no tensor-active signal).

- `realbench/training/w1_llama7b.py` — real LLaMA-7B-shaped model
  (`transformers`, random init, DP(FSDP) on CUDA), real `aegis.init()`
  wiring, real step/heartbeat logging. Launched via the existing `aegis
  run` (torchrun) for Phase 0.
- `realbench/sensors/rank_heartbeat.py` — the first real (non-synthetic)
  AEGIS sensor: peer heartbeat-file polling → real `TelemetryEvent` on the
  live UTP.
- `chaos_inject/real_injector.py` — standalone process, real `SIGKILL` of a
  target rank at a fixed step (separate mechanism from the existing
  in-process `ChaosHarness`, which stays synthetic-only).
- `realbench/collector/{gpu_sampler,align}.py` — nvidia-smi sampler +
  §4.5.4 alignment (`recovery_gap`/`idle_gpus_by_util_proxy`/
  `wasted_gpu_hours`), with the DCGM-vs-nvidia-smi caveat enforced in the
  field names, not just prose.
- `realbench/phase0_trust_anchor/` and `realbench/phase1_per_tier/` —
  orchestration + report generation for the two phases test_suite.md §8
  mandates be run in order; Phase 1 refuses to run without a passing
  Phase 0 report (`--skip-gate` overrides for debugging only).
- `realbench/slurm/` — `sbatch_phase0.sh` / `sbatch_phase1_b1.sh`, headers
  styled after a known-working SeaWulf training script (module load python,
  `pip install torch` from the cu121 index + `pip install -e .`, OMP_*/
  TORCHDYNAMO_DISABLE env vars, `srun --cpu-bind=cores`). `--partition`
  and `--cpus-per-task`/`--mem` in the header are placeholders to edit
  before submitting; the A100 cost rate is passed via `sbatch
  --export=A100_SPOT_USD_PER_HR=...` at submission time.

**Real finding from local validation, not a guess:** training for Phase 1
is launched as independent per-rank OS processes, not via `aegis run`/
torchrun — torchrun's elastic agent tears down every surviving rank the
instant one dies, which would defeat B1 testing entirely. Phase 0 (nothing
dies there) still uses `aegis run` normally.

**Known limitation surfaced by real testing, documented in every report,
not hidden:** this build's `ComputeLayer.recover()` does real (proxy-
tensor) MeCeFO absorb math and is genuinely measured, but does not
reconfigure the `torch.distributed` process group — so after a real B1
kill, training halts rather than resumes. `recovery_gap`/`wasted_gpu_hours`
are therefore not yet end-to-end measurable; what IS real and reported is
detection latency and absorb compute time. Wiring real process-group
reconfiguration (and a B-VANILLA checkpoint-restart baseline, and a
TorchFT condition) are the natural next steps once this lands on SeaWulf.

To run for real: edit the `--partition`/`--cpus-per-task`/`--mem` placeholders
in both sbatch headers for SeaWulf's actual node spec, then
`sbatch realbench/slurm/sbatch_phase0.sh`, confirm `phase0_report.md`
PASSes, then `sbatch --export=A100_SPOT_USD_PER_HR=<rate>
realbench/slurm/sbatch_phase1_b1.sh <phase0_out_dir>`.
