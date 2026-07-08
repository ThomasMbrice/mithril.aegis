# AEGIS — Benchmark & Evaluation Suite

**Design Document — v0.1 (working draft)**
Owner: Thomas
Status: Pre-implementation design
Scope: How we measure AEGIS against open-source alternatives on a real A100 cluster + simulator for scale.
Last updated: 2026-06-29

> Hero metric: **$/GPU-hour saved vs. baselines.** Everything else (goodput, per-tier recovery time) is supporting evidence that explains *why* the dollar number is what it is. The suite is designed so the headline number is defensible, reproducible, and traceable down to individual fault events.

---

## 0. Evaluation philosophy

A fault-tolerance system can only be measured *under failure*. So the suite is a **fault-injection benchmark**, not a throughput benchmark. The structure is:

```
fixed workload  ×  fixed failure trace  ×  {AEGIS, baseline_1, ... baseline_n}
        │                  │                          │
   same model,        same faults at            only the FT system
   same data,         same wall-clock           changes between runs
   same parallelism   moments
        └──────────────────┴──────────────────────────┘
                           ▼
           per-run: goodput, recovery time/tier, $/GPU-hr
```

The cardinal rule: **only the fault-tolerance system varies between runs.** Same model, same data, same parallelism config, same *deterministic* failure trace replayed identically. Any difference in outcome is attributable to the FT system alone. This is what makes the $/GPU-hr comparison honest.

---

## 1. Two-tier evaluation: real cluster + simulator

| | Real cluster (SeaWulf A100 partition) | Simulator |
|---|---|---|
| **Purpose** | credibility — "we ran it on real hardware" | scale claims — "it holds at 1000+ GPUs" |
| **Scale** | 1–4 nodes, 8–32 A100s | hundreds to thousands of simulated GPUs |
| **What's real** | actual NCCL, actual NIC failover, actual checkpoint I/O, actual recovery | fault model, recovery-cost model, goodput model |
| **What it proves** | the mechanisms *work* and the per-tier costs are real | the *economics* scale the way the small-cluster data predicts |
| **Risk it controls** | "does this even function on hardware" | "does the $/GPU-hr story survive at the scale labs actually run" |

**The bridge between them is the calibration step (§5):** the simulator's cost model is *fit to the real-cluster measurements*, so simulated $/GPU-hr at scale is an extrapolation of measured small-scale reality, not an invented number. This is the single most important methodological move — without it, the scale claims are hand-waving. R²CCL's own paper uses exactly this real-testbed + simulator pattern, so it's a defensible precedent.

---

## 2. The baseline matrix (what we beat)

Full comparison. Each baseline isolates a different "why not just use X" objection.

| Baseline | Represents | What beating it proves | Setup notes |
|----------|-----------|------------------------|-------------|
| **B-VANILLA** | status-quo checkpoint-and-restart (periodic `torch.save` + full job restart) | the core thesis: per-step/per-tier beats full rollback | PyTorch native; tune checkpoint interval fairly (sweep it — see §6) |
| **B-TORCHFT** | the direct OSS competitor | we beat the thing a lab would reach for first | TorchFT elastic + its checkpoint story |
| **B-R2CCL** | the transport paper's own released tool, alone | composition > any single layer (transport only) | R²CCL standalone, no compute/storage FT |
| **B-MECEFO** | the compute paper's own tool, alone | composition > single layer (compute only) | MeCeFO standalone |
| **B-TIERCHECK** | the storage paper's own tool, alone | composition > single layer (storage only) | TierCheck standalone |
| **AEGIS** | our composed runtime | — | full five-tier blast-radius routing |

**The three single-primitive baselines are the most important comparison in the whole suite.** They directly test the central product claim — that *no single paper is a product, but composed they are*. If AEGIS doesn't clearly beat each primitive run alone, the entire thesis is wrong, and you want to know that on a 32-GPU cluster, not after raising money.

**Fairness discipline (or the comparison is worthless):**
- Each baseline gets its *best* configuration, not a strawman. Sweep B-VANILLA's checkpoint interval and report its best result.
- Same hardware, same NCCL version, same model, same trace.
- Where a baseline can't handle a fault class at all (e.g., B-R2CCL can't recover a dead node), that's recorded as a forced full-restart at that fault, not as a crash — model the realistic fallback the baseline would actually use.

---

## 3. Workloads

Pick workloads that (a) match the papers' own testbeds for reproducibility and (b) are realistic enough that labs believe the result.

| Workload | Model | Parallelism | Why |
|----------|-------|-------------|-----|
| **W1 — reproduction** | LLaMA-7B | DP+PP, 8×A100 | matches MeCeFO's exact testbed; lets us validate our MeCeFO integration reproduces 4.18% before trusting anything else |
| **W2 — mid-scale** | ~13B | DP+PP+TP, 16–32 A100 | exercises all three parallelism dims; TP path stresses MeCeFO's node-locality claim |
| **W3 — checkpoint-heavy** | up to ~40B (sharded) | FSDP/HSDP | matches TierCheck's 40B evaluation; stresses the storage tiers (B2–B4) |
| **W4 — inference** (optional, Phase 2) | serving workload + KV-cache | TP serving | exercises R²CCL's inference path; ties to KV-cache thesis |

W1 is the **trust anchor** — if we can't reproduce each paper's headline number on its own testbed, the composed numbers are meaningless. Run W1 reproductions *first* and gate everything else on them.

---

## 4. Failure traces (the experimental variable)

Faults must be **deterministic and replayable** — same fault, same wall-clock moment, every run. Built on the `chaos-inject` harness from the core design doc.

### 4.1 Per-tier isolation traces (clean attribution)

One trace per blast-radius tier — each injects only that fault class, so we measure each tier's recovery cost in isolation.

| Trace | Fault injected | Tier exercised | Baseline that should struggle |
|-------|----------------|----------------|-------------------------------|
| **FT-B0** | NIC port down / link flap (bring RNIC down N s, back up) | transport | B-VANILLA (full rollback for a transient!), B-MECEFO, B-TIERCHECK |
| **FT-B1** | kill a node/rank mid-iteration | compute | B-VANILLA, B-R2CCL, B-TIERCHECK |
| **FT-B2** | CUDA kernel crash / hung rank (recoverable in place) | storage T1 | everyone except TierCheck-class |
| **FT-B3** | node unrecoverable, force hardware replacement | storage T2 | B-R2CCL, B-MECEFO |
| **FT-B4** | rack-level outage (kill a correlated group) | storage T3 | all single-primitive baselines |

### 4.2 Realistic mixed traces (the money traces)

These drive the $/GPU-hr hero number. Faults arrive as they do in production: mixed classes, correlated bursts, drawn from realistic distributions.

| Trace | Model | Purpose |
|-------|-------|---------|
| **FT-POISSON** | independent failures at swept MTBF (e.g., 10M-step-style sweep) | standard goodput-vs-MTBF curve, comparable to MoE-checkpointing literature |
| **FT-BURST** | correlated rack-event bursts (B1s leading into a B4) | tests §3.3 escalation correlation window; the scenario single-primitive baselines handle worst |
| **FT-PRODUCTION** | replayed from a published/realistic large-cluster failure profile | the trace the $/GPU-hr headline is computed on |

FT-BURST is where AEGIS should win biggest, because it's exactly the correlated-failure case that a single-layer tool mishandles — and it's the case the core doc's correlation-window design was built for. Make sure the suite spotlights it.

---

## 4.5 How a fault is *physically* tested on the GPUs

This section is the concrete mechanism the rest of the doc assumes. Nothing here is simulated on the real cluster — a real distributed training job runs on the A100 partition, real faults are injected into it, and real recovery is stopwatched. The simulator (§1, §5.3) only extrapolates these measured costs to scale; it never invents recovery physics.

### 4.5.1 The running system under test

```
SLURM allocation (8–32 A100s)
        │
   aegis run  (wraps torchrun)  ──►  real training job:
        │                            real NCCL, real gradients,
        │                            real checkpoint I/O to parallel FS
        ├──►  chaos-inject process   (reads fixed trace, fires faults at step N)
        └──►  metrics collector      (DCGM/nvidia-smi sampler + step-log tailer)
```

Three processes coexist in the allocation: the **training job** (the thing being protected), **chaos-inject** (the thing breaking it), and the **metrics collector** (the thing stopwatching it). Only the FT system embedded in the training job changes between runs; chaos-inject and the collector are identical across all systems.

### 4.5.2 How each fault class is actually injected

Every injection is a *real* fault, not a flag telling the code to pretend. The recovery path exercised is the same one that would fire in production.

| Tier | Real injection mechanism on the A100 cluster | What genuinely happens |
|------|----------------------------------------------|------------------------|
| **B1 — dead rank/node** | `chaos-inject` sends `SIGKILL` to the training process bound to a target GPU at a scheduled step (or evicts all ranks on a node) | the rank vanishes from the process group; NCCL collectives actually hang/error; the FT system must actually recover a genuinely missing rank |
| **B0 — NIC/link failure** | administratively down the RDMA interface: `ip link set <rnic> down` for N s then back up, and/or `tc qdisc` to drop/delay packets on the RoCE/IB interface | NCCL transport genuinely times out; R²CCL must migrate to a real backup NIC. **Requires ≥2 physical RNICs** — see §4.5.5 |
| **B2 — kernel crash / hung rank** | inject a CUDA error, or wedge a rank (stop it calling collectives) so peers block on the barrier | a real in-place-recoverable software fault; peers really stall on the collective |
| **B3 — node unrecoverable** | `SIGKILL` the node's ranks *and* mark the node cordoned so it can't rejoin — forces recovery to a replacement rather than in-place | the FT system must genuinely restore node state from peer/remote, not restart the same process |
| **B4 — rack outage** | simultaneously evict all ranks on a scheduled subset of nodes (correlated group) at the same step | approximates a rack power event on a small cluster; forces the most expensive recovery path |

### 4.5.3 Determinism: the same fault at the same step, every run

`chaos-inject` reads a **fixed trace file keyed to training step numbers**, not wall-clock, so replays are identical regardless of per-system throughput differences:

```
# trace file (illustrative)
step=5000   fault=B1   target=rank3         duration=—
step=5000   fault=B0   target=node2:rnic0   duration=30s
step=12000  fault=B4   target=nodes[2,3]    duration=—
```

All RNG seeded. B-VANILLA, TorchFT, R²CCL-alone, and AEGIS all get the *same* rank killed at step 5000 for the *same* duration — so any difference in outcome is attributable to the FT system alone. chaos-inject watches the training job's step counter (via the shared step-log, §4.5.4) and fires when the target step is reached.

### 4.5.4 On-device instrumentation (the part that turns "stalled" into dollars)

The metrics collector is the actual measurement apparatus. Two data streams, time-aligned:

**(a) Per-GPU utilization — DCGM (preferred) or `nvidia-smi` fallback.**
- Sample at **1 s resolution** (finer if the idle windows are short; 1 s is the floor for capturing a 30 s recovery gap cleanly).
- Fields: `DCGM_FI_DEV_GPU_UTIL` (SM utilization %), `DCGM_FI_DEV_FB_USED` (memory, to see checkpoint spikes), `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (tensor-core activity — the truest "is this GPU doing training work" signal), and `DCGM_FI_DEV_POWER_USAGE` (power draw drops during idle — a clean secondary confirmation of the idle window).
- Run `dcgmi dmon` or the DCGM Python bindings as a sidecar; write timestamped rows to a per-GPU log.
- **Why tensor-core-active, not just GPU-util:** GPU-util reports "a kernel is running," which can be nonzero during a spin-wait on a hung collective. Tensor-core-active distinguishes *real training compute* from *a GPU spinning on a stalled AllReduce* — critical for measuring the true idle window, because a rank blocked on a dead peer looks "busy" to naive util.

**(b) Training progress — step-log tailer.**
- The training loop emits `{wall_clock, global_step, loss}` every step to a shared log.
- The collector tails it; the gap between consecutive steps' wall-clock is the per-step time; a *gap that balloons* marks the recovery window (training produced no steps).

**Time alignment.** Both streams carry wall-clock timestamps from the same clock (all processes in one SLURM allocation → one node's clock as reference, or NTP-synced). chaos-inject also logs `{wall_clock, step, fault_fired}`. Aligning the three logs on wall-clock gives, per fault:

```
t_fault      = when chaos-inject fired          (from chaos-inject log)
t_resume     = first step after the gap          (from step-log tailer)
recovery_gap = t_resume − t_fault                 (wall-clock seconds)
idle_GPUs    = GPUs with tensor-active ≈ 0 during the gap  (from DCGM stream)

wasted_GPU_hours(this fault) = recovery_gap × idle_GPUs / 3600
```

That last line is the whole ballgame: `wasted_GPU_hours` is **measured**, not modeled — a real stopwatch reading (`recovery_gap`) times a real device count (`idle_GPUs` from DCGM). The $/GPU-hr headline is arithmetic on these measured quantities × $2.35.

### 4.5.5 What "real" costs on the real cluster, and what falls to sim

- **Real-validated on A100s:** B1, B2, B3, B4 — all injectable and measurable on any A100 partition, because they're process/CUDA-level faults.
- **B0 is conditional:** it needs ≥2 physical RNICs per node with spare bandwidth for R²CCL to migrate onto. **Confirm SeaWulf's NIC topology before committing to real B0 runs.** If nodes are single-NIC/single-plane, B0 is exercised in the **simulator only**, and every results table must label B0 as "sim-validated" rather than "hardware-validated." This is the single biggest determinant of how much of the suite is real vs. extrapolated — check it on day one.
- **Scale (100s–1000s of GPUs):** always sim. The simulator consumes the *measured* per-fault `recovery_gap` and `idle_GPUs` from the real runs (§5.3 calibration) and replays them across a larger topology and a longer trace. The scale $/GPU-hr is real small-cluster measurement extrapolated, with calibration error reported.

### 4.5.6 One concrete end-to-end example (B1, 8×A100, W1)

1. Launch W1 (LLaMA-7B, 8 A100s) under `aegis run`. DCGM sidecar sampling at 1 s; step-log tailer active.
2. At step 5000, chaos-inject `SIGKILL`s the process on GPU 3. Logs `{t=…, step=5000, fault=B1, target=rank3}`.
3. NCCL's next collective hangs; AEGIS's classifier tags B1; MeCeFO neighbor-absorb fires; training resumes.
4. Step-log shows no new steps from `t_fault` until `t_resume` (say 8 s later). DCGM shows GPUs 0–2,4–7 at tensor-active ≈ 0 during those 8 s (they waited); GPU 3 gone.
5. `wasted_GPU_hours = 8 s × 7 idle GPUs / 3600 = 0.0156 GPU-hr` → `× $2.35 = $0.037` for that one fault.
6. Re-run identically with B-VANILLA: it can't absorb a dead rank, so it full-restarts from the last checkpoint — `recovery_gap` might be 400 s across all 8 GPUs = `0.89 GPU-hr = $2.09`.
7. **AEGIS saved $2.05 on that single B1 event.** Sum over the FT-PRODUCTION trace → the headline.

---

## 5. Metrics

### 5.1 Hero metric

**$/GPU-hour saved vs. each baseline**, computed per-trace:

```
cost(system, trace) = wasted_GPU_hours(system, trace) × gpu_hourly_cost
wasted_GPU_hours    = recovery_time_GPU_hours
                    + idle_GPU_hours_during_recovery   ← the "1023 idle while 1 fails" waste
                    + steady_state_FT_overhead_GPU_hours

$/GPU-hr_saved = cost(baseline, trace) − cost(AEGIS, trace)
```

- `gpu_hourly_cost` = $2.35/hr (H100 1-yr; also report A100 spot for the real cluster).
- On the real cluster, `wasted_GPU_hours` is **measured, not modeled** — it is `recovery_gap × idle_GPUs` per fault, both read off the on-device instrumentation in §4.5.4 (DCGM tensor-active + step-log tailer), summed over the trace. The dollar figure is arithmetic on real stopwatch readings.
- Report **per baseline, per trace**, plus an aggregate on FT-PRODUCTION as the single headline figure.
- Report a **breakdown**: how much of the saving came from each tier (B0/B1/B2-4). This is what makes the number credible — a reviewer can see *where* the dollars come from.

### 5.2 Supporting metrics

| Metric | Definition | Why it matters |
|--------|-----------|----------------|
| **Goodput** | productive_training_time / total_wall_clock | the standard FT comparison metric; the curve vs. MTBF |
| **Recovery time per tier** | wall-clock from fault → training resumed, per blast-radius class | proves the per-tier cost-matching claim; should show B0≪B4 |
| **Steady-state overhead** | throughput hit with FT on, *no faults* | proves AEGIS isn't expensive when nothing's failing (R²CCL <1%, MeCeFO 4.18%, TierCheck <10s targets) |
| **Escalation correctness** | % faults routed to the correct (cheapest valid) tier | proves the classifier works; a B4-as-B1 misroute is the dangerous error |
| **Convergence fidelity** | final loss/perplexity vs. no-fault baseline | proves MeCeFO's approximation didn't break the model |
| **Time-to-recover distribution** | p50/p99 recovery, not just mean | labs care about tail; one slow recovery can dominate |

### 5.3 Calibration metric (the real↔sim bridge)

For every fault class, measure recovery cost on the **real cluster**, fit the simulator's per-tier cost model to it, then report **calibration error** (simulated vs. measured recovery cost on held-out real runs). Low calibration error is what licenses the scale extrapolation. **State this error explicitly in any scale claim** — it's the honesty that makes the big-number plausible.

---

## 6. Experimental controls & fairness

These are the things a hostile reviewer (or a sharp VC's technical advisor) will attack. Lock them down:

1. **Deterministic fault replay.** Same trace → identical fault timing across all systems. Seed everything.
2. **Best-config baselines.** Sweep B-VANILLA's checkpoint interval; report its optimum, not a deliberately bad one.
3. **Repeat for variance.** N≥3 runs per (system × trace); report mean ± std. Fault-tolerance results are noisy; a single run proves nothing.
4. **Identical software stack.** Same NCCL, CUDA, PyTorch version across all systems. Pin them.
5. **Warm-up exclusion.** Discard the first K iterations (allocator warm-up, etc.) before measuring steady state.
6. **No cherry-picked traces.** Pre-register the trace set before running. The FT-PRODUCTION headline trace is fixed *before* seeing results.
7. **AEGIS-self-fault control.** Include a run where AEGIS's *own* components are fault-injected (transparency-contract promise #5) — proves AEGIS doesn't make things worse when it breaks.

Control #2 and #6 are the ones that separate a credible benchmark from a marketing chart. Skipping them is how good ideas get dismissed as cooked numbers.

---

## 7. Suite structure & deliverables

```
aegis-bench/
├── workloads/        W1–W4 configs (model, data, parallelism)
├── traces/           FT-B0..B4, FT-POISSON/BURST/PRODUCTION (deterministic)
├── systems/          adapters: aegis, vanilla, torchft, r2ccl, mecefo, tiercheck
├── chaos-inject/     the fault-injection harness (shared with core design)
├── sim/              simulator + cost model + calibration fitter
├── metrics/          $/GPU-hr, goodput, recovery, calibration computation
├── runner/           orchestration: (system × workload × trace × repeat)
└── report/           auto-generated comparison matrix + plots
```

**Auto-generated deliverable:** a single comparison matrix (systems × traces) with $/GPU-hr saved as the headline cell, goodput and recovery-time as drill-downs, plus the goodput-vs-MTBF curve and the per-tier saving breakdown. This *is* the artifact you put in the deck.

---

## 8. Phased execution

- **Phase 0 — trust anchor.** Real cluster, W1, reproduce each paper's headline number individually (MeCeFO 4.18%, R²CCL <1%/<3%, TierCheck <10s). *Gate: if we can't reproduce, stop and fix integration before any comparison.*
- **Phase 1 — per-tier isolation.** Real cluster, FT-B0..B4 × full baseline matrix on W1/W2. Produces the clean per-tier recovery-cost evidence and the "composition beats single primitive" result.
- **Phase 2 — mixed traces + hero number.** Real cluster, FT-POISSON/BURST/PRODUCTION × full matrix on W2/W3. Produces the small-scale $/GPU-hr number.
- **Phase 3 — calibrate & scale.** Fit simulator to Phase 1–2 real data; report calibration error; extrapolate $/GPU-hr to hundreds/thousands of GPUs on FT-PRODUCTION. Produces the scale claim.
- **Phase 4 — (optional) inference.** W4 + R²CCL inference path; ties to KV-cache thesis.

The deck-ready result exists after Phase 2 (real, small-scale, honest). Phase 3 makes it *big*. Don't let Phase 3 extrapolation run ahead of Phase 0's reproductions — a scaled number built on an unvalidated integration is worse than no number.

---

## Appendix — Notes on SeaWulf as the real-cluster target

- A100 partition is the natural fit; you've already run headless A100 jobs there (CARLA PoC) and know the SLURM environment, which removes the infra-debugging tax that usually eats the first weeks of a benchmark effort.
- **Check the NIC topology before committing to FT-B0 traces:** R²CCL/B0 requires ≥2 RNICs per node with spare bandwidth to migrate onto. If SeaWulf nodes are single-NIC or single-plane, the B0 transport tier can't be exercised on real hardware — you'd validate B0 in the simulator only and note that limitation explicitly. Confirm this first; it changes which tiers are "real-validated" vs. "sim-only" in every results table.