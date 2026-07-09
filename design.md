# AEGIS — A Unified Blast-Radius-Aware Fault-Tolerance Runtime for LLM Training & Serving

**Design Document — v0.1 (working draft)**
Owner: Thomas
Status: Phase 0 implemented; Phase 1 not started; Phase 2 partially prototyped out of order (see §8.1)
Last updated: 2026-07-08

> Working codename **AEGIS** = *Adaptive, Escalating, Graded Infrastructure Survivability*. Rename later.

---

## 0. Document purpose

This is an engineering design doc, not a thesis. It specifies **what to build, in what order, with what interfaces, and how to test each piece**. The investment/GTM framing lives in a separate doc — here we assume the product is worth building and focus on making it buildable.

The central engineering claim: **a single failure-classification controller can route each fault to the cheapest layer capable of absorbing it**, instead of defaulting every fault to full checkpoint-and-restart. Three published primitives each own one layer of the blast-radius spectrum. None is a product alone. The product is the *controller that composes them plus the unified telemetry, policy, and recovery-consensus plane that makes them interoperate*.

---

## 1. The blast-radius model (the spine of the whole system)

Every fault tolerance decision in AEGIS is a function of **blast radius** — how much of the job a fault invalidates — and we always recover at the *smallest* radius that restores correctness. This is the organizing principle; everything else hangs off it.

| Tier | Blast radius | Example faults | Owning primitive | Target recovery cost |
|------|-------------|----------------|------------------|---------------------|
| **B0** | Transient link / single NIC | NIC port flap, link fluctuation, RDMA timeout | **R²CCL** (transport) | <1% train / <3% infer overhead, no rollback |
| **B1** | Single GPU / node death | GPU fell off bus, node crash, OOM-killed rank | **MeCeFO** (compute fallback) | ~4% throughput drop, neighbor absorbs, no rollback |
| **B2** | Software crash, recoverable in place | CUDA kernel crash, runtime bug, hung rank | **TierCheck Tier-1** (local volatile) | sub-10s restore from local memory |
| **B3** | Node-level state loss | node unrecoverable, must replace hardware | **TierCheck Tier-2** (peer volatile) | restore from peer replica |
| **B4** | Rack-level / cluster outage | rack power loss, fabric partition, cluster-wide | **TierCheck Tier-3** (remote durable) | restore from remote durable base checkpoint |

**Design invariant:** a fault is only ever escalated to the next tier if the current tier cannot absorb it. Escalation is one-directional and logged. This is the single most important property to preserve through every refactor.

```
        fault detected
              │
              ▼
   ┌──────────────────────┐
   │  Failure Classifier   │◄──── unified telemetry plane
   └──────────┬───────────┘
              │ classify → blast radius
   ┌──────────▼───────────┐
   │  Escalation Policy    │
   └──────────┬───────────┘
   B0 │  B1 │  B2 │ B3/B4
      ▼      ▼     ▼     ▼
   R²CCL  MeCeFO  Tier1 Tier2/3
   (keep  (absorb (in-  (peer/
   socket) on     place  remote
          neighbor)restore)restore)
```

---

## 2. Component inventory & external dependencies

### 2.1 The three composed primitives

| Primitive | Layer | Core mechanism | Key reported numbers | Source |
|-----------|-------|----------------|---------------------|--------|
| **R²CCL** | Network/transport | Multi-NIC connection migration; bandwidth-aware load redistribution; resilient collective algorithms (R2CC-Balance variant) | <1% training / <3% inference overhead; maintains 85–89% throughput on AllGather/ReduceScatter/SendRecv under single-NIC failure; beats AdapCC 12.18×, DéjàVu 47× | arXiv 2512.25059 (Wang, Yu, Xiong, Z. Liu — UMD) |
| **MeCeFO** | Compute/algorithm | Node death → neighbor absorbs both workloads via (i) skip-connection on MHA backprop, (ii) selective activation recomputation in FFN, (iii) low-rank gradient approximation | 4.18% throughput drop under high-frequency failures; matches baseline convergence rate; local to each node so extends to TP/PP | arXiv 2510.16415 |
| **TierCheck** | Storage/checkpoint | 3-tier differential checkpointing aligned to failure severity; decentralized recovery consensus over surviving ranks | end-to-end checkpoint <10s; tested to 40B params; tier matched to failure class | arXiv 2605.17821 |

### 2.2 Upstream dependencies (things we build *on*, do not own)

| Dependency | Why we need it | Risk level | Notes |
|-----------|----------------|-----------|-------|
| **NCCL** (NVIDIA) | R²CCL is a CCL — sits where NCCL sits; we must intercept or replace the collective path | HIGH | NVIDIA owns this; primary moat risk. Our shim must track NCCL ABI. |
| **PyTorch / TorchFT** | Process-group integration, elastic membership, the framework labs actually use | HIGH | Meta owns it; "still evolving." Our integration point is `ProcessGroup` + `torch.distributed.elastic`. |
| **Multi-NIC hardware** (≥2 RNICs/node) | R²CCL's entire failover model assumes spare NIC bandwidth to migrate onto | MEDIUM | Hard requirement. Single-plane single-NIC clusters can't use B0 tier — they fall through to B1+. Document as a deployment prerequisite. |
| **RDMA / InfiniBand or RoCE** | transport substrate for connection migration | MEDIUM | RC transport assumptions (exactly-once). Validate on both IB and RoCEv2. |
| **CUDA / driver** | MeCeFO recompute + low-rank ops, all GPU-side | LOW | Standard. |
| **Remote durable store** (S3-class / parallel FS) | TierCheck Tier-3 | LOW | Pluggable backend; Lustre/GPFS/S3. |

### 2.3 What we build (the actual product surface)

These are the net-new components — the parts no paper ships:

1. **Unified Telemetry Plane (UTP)** — single event bus that all three layers emit to and the classifier consumes.
2. **Failure Classifier (FC)** — maps raw signals → blast-radius tier.
3. **Escalation Policy Engine (EPE)** — decides which layer handles a classified fault; enforces the one-directional escalation invariant.
4. **Unified Recovery Consensus (URC)** — generalizes TierCheck's per-tier consensus to a cross-layer "what's the latest globally-valid state after this fault" decision, so transport/compute/storage recoveries don't disagree.
5. **Control-plane API & policy DSL** — how an operator declares cost/latency tradeoffs ($/GPU-hr thresholds).
6. **Cost accounting / KPI meter** — measures $/GPU-hour saved (the viability KPI), per fault, per tier. This is also the product's primary sales artifact.

---

## 3. The hard integration problems (where the real work is)

Composing three independent research systems is not "import three libraries." These are the genuine design conflicts:

### 3.1 Conflict: who detects the fault first?

R²CCL detects NIC failure at the transport layer (~10s retry timeout today). MeCeFO assumes it's *told* a node died. TierCheck assumes a failure already happened and asks "which tier do I restore from." **Three different detection assumptions.**

**Design decision:** UTP is the single source of truth for fault events. Each primitive's native detector becomes a *sensor* that publishes to UTP rather than acting unilaterally. The FC, not the primitive, decides the response.

- *Risk:* R²CCL's transport-level retry is latency-critical (must act in ms–s). Routing through a central classifier may add latency. **Mitigation:** B0 transport failover runs a *fast path* — R²CCL may begin migration immediately and publish to UTP concurrently; the classifier only intervenes if the fault escalates. Fast-path autonomy for the lowest tier only.

### 3.2 Conflict: state consistency across layers

If R²CCL migrates a connection mid-AllReduce *and simultaneously* MeCeFO's neighbor starts absorbing a dead node, which view of the gradient is authoritative? TierCheck's recovery consensus derives the latest valid checkpoint **from surviving data only** — we must extend that property so that an in-flight transport migration or compute fallback doesn't produce a checkpoint that captures a torn state.

**Design decision:** URC introduces a **fault epoch counter**. Every classified fault increments the epoch. Checkpoints, collective ops, and compute fallbacks are all tagged with the epoch they belong to. Recovery consensus only considers state from a consistent epoch boundary. This is the cross-layer generalization of TierCheck's "global minimum recoverable version" reduction.

### 3.3 Conflict: overlapping/simultaneous faults

Real clusters fail in correlated bursts (a rack power event = many B1s + B4 at once). The escalation model must handle a fault that *starts* as B1 but is actually the leading edge of B4.

**Design decision:** EPE uses a **debounce + correlation window**. A classified B1 that is followed within window W by ≥N correlated node failures in the same rack is re-classified upward to B4 *before* committing to the expensive neighbor-absorb path. Tunable W, N.

### 3.4 Conflict: MeCeFO degrades model fidelity; checkpoints must know that

MeCeFO's skip-connection + low-rank approximation means iterations computed under fallback are *approximate*. If TierCheck checkpoints during a MeCeFO fallback window, the checkpoint encodes approximated state. Acceptable (the paper shows convergence holds), but the checkpoint metadata must record "this base was taken under degraded compute" for audit and for reproducibility guarantees.

**Design decision:** Checkpoint metadata schema includes a `fidelity_flag` and the active fault epoch. No silent approximation.

---

## 4. Architecture: layered build plan

### Layer A — Foundation: Telemetry + Classification plane
*Nothing routes correctly without this. Build first.*

- **A1. Unified Telemetry Plane (UTP)**
  - Event schema: `{timestamp, rank, node, nic_id, fault_signal, raw_payload, epoch}`
  - Transport: low-overhead, must not become a scaling bottleneck itself (TierCheck explicitly designs to avoid new bottlenecks — same discipline here)
  - Sensors: wrap each primitive's native detector as a publisher
- **A2. Failure Classifier (FC)**
  - Input: UTP stream → output: blast-radius tier B0–B4
  - Start rule-based (signal → tier lookup), leave hook for learned classifier later (see §7)
- **A3. Fault epoch service** — monotonic counter, the consistency backbone for URC

### Layer B — Transport survivability (B0)
*Wraps R²CCL.*

- **B1-impl. NCCL interception shim** — sit on the collective path; detect NIC/link failure; trigger migration
- **B2-impl. Connection migration** — move QP to backup NIC (R²CCL primary-backup model)
- **B3-impl. Bandwidth-aware redistribution** — reschedule AllReduce around reduced bandwidth (R2CC-Balance)
- **B4-impl. Fast-path autonomy** — act in ms–s, publish to UTP concurrently

### Layer C — Compute survivability (B1)
*Wraps MeCeFO.*

- **C1. Neighbor-absorb trigger** — on classified node death, hand workload to neighbor
- **C2. The three MeCeFO mechanisms** — skip-connection (MHA backprop), selective recompute (FFN), low-rank gradient approx
- **C3. Parallelism-aware redistribution** — DP+PP primary, extend to TP (MeCeFO is node-local so this is supported by design)
- **C4. Fidelity flagging** — emit degraded-compute marker to UTP/checkpoint metadata

### Layer D — Storage survivability (B2–B4)
*Wraps TierCheck.*

- **D1. Tier-1 local volatile** — differential checkpoints, software-crash recovery in place
- **D2. Tier-2 peer volatile** — async replicate to peer for node-loss recovery
- **D3. Tier-3 remote durable** — async migrate base checkpoints for rack recovery
- **D4. Decoupled persistence + differential/base split** — frequent compressed diffs to volatile, infrequent base to remote
- **D5. Unified Recovery Consensus (URC)** — extend TierCheck's surviving-rank consensus across epochs and across layers B/C

### Layer E — Control plane & economics
*The product-defining layer; what you actually sell.*

- **E1. Policy DSL** — operator declares thresholds (e.g., "never spend >$X to avoid a rollback of <Y GPU-hrs")
- **E2. Escalation Policy Engine (EPE)** — consumes FC output + policy → routes to B/C/D; enforces one-directional escalation + debounce/correlation window
- **E3. Cost/KPI meter** — per-fault $/GPU-hour-saved accounting vs. the checkpoint-and-restart baseline
- **E4. Operator dashboard + API** — observability, the marketing artifact

**Build order:** A → (B ∥ C ∥ D can parallelize once A exists) → E. E2 (EPE) is the integration keystone and should be scaffolded early as a stub even while B/C/D are stubs, so the contract is fixed.

---

## 5. Test plan — per layer, with fault injection

The whole system is a fault-response system, so **fault injection is the test methodology**, not an add-on. Build a fault-injection harness as a first-class tool (call it `chaos-inject`).

### 5.1 Unit / component tests

| ID | Component | Test | Pass criterion |
|----|-----------|------|----------------|
| UT-A1 | UTP | inject 10k synthetic events/s | no event loss, <X ms p99 publish latency, no throughput regression on training |
| UT-A2 | FC | replay labeled fault traces | classification accuracy ≥ target; zero misclassification of B4-as-B1 (the dangerous direction) |
| UT-A3 | Epoch service | concurrent fault bursts | strictly monotonic, no duplicate epochs under race |
| UT-B | R²CCL shim | single-NIC down on 2-node H100 IB (paper's own testbed) | reproduce <1% train / <3% infer overhead; 85–89% collective throughput retained |
| UT-C | MeCeFO | kill one node, LLaMA-7B, 8×A100 (paper testbed) | reproduce ~4.18% throughput drop; no OOM on neighbor; convergence matches baseline |
| UT-D | TierCheck | inject B2/B3/B4 separately | each restores from correct tier; end-to-end checkpoint <10s; no cross-tier restore from wrong tier |
| UT-D5 | URC | fail a rank mid-checkpoint | consensus picks latest globally-valid version from surviving ranks only |

### 5.2 Integration tests (the part no paper has done)

| ID | Scenario | What it validates | Pass criterion |
|----|----------|-------------------|----------------|
| IT-1 | NIC flap during AllReduce | B0 fast-path doesn't trip classifier into B1 | transport migrates, no neighbor-absorb triggered, no checkpoint rollback |
| IT-2 | Node death during active R²CCL migration | §3.2 consistency | epoch boundary clean; no torn gradient; neighbor absorbs from consistent state |
| IT-3 | B1 that is leading edge of B4 (rack failing) | §3.3 correlation window | EPE re-classifies upward *before* committing neighbor-absorb; restores from Tier-3 |
| IT-4 | Checkpoint taken during MeCeFO fallback | §3.4 fidelity | checkpoint carries `fidelity_flag`; reproducible; convergence preserved |
| IT-5 | Simultaneous independent B0 + B1 in different DP groups | concurrency | both handled at correct tier, no interference, no double escalation |
| IT-6 | Escalation invariant fuzz | one-directional escalation never violated | property test: no path ever de-escalates mid-fault |

### 5.3 System / scale tests

- **ST-1. Simulator-based scale-out** — R²CCL paper uses large-scale ML simulators for hundreds of GPUs; reuse that approach for the *whole stack* under diverse failure patterns (Poisson failures, correlated bursts, rack events).
- **ST-2. Goodput under sustained MTBF** — measure productive-time / total-time across MTBF sweeps (10M-step style sweeps like the MoE checkpointing literature uses). Compare against checkpoint-and-restart baseline.
- **ST-3. $/GPU-hour-saved end-to-end** — the KPI test. Run a realistic failure trace, measure dollars saved vs. baseline at $2.35/hr/GPU H100 pricing. **This is the number that sells the product.**

### 5.4 Regression / continuous

- Re-run UT-B/C/D reproductions on every dependency bump (NCCL version, PyTorch version) — these are the high-risk upstreams (§2.2).
- ABI canary: detect NCCL/Torch ABI drift before it breaks the shim.

---

## 6. Risks & open questions (engineering, not market)

| Risk | Layer | Mitigation / open question |
|------|-------|---------------------------|
| NCCL ABI drift breaks R²CCL shim | B | ABI canary in CI; consider upstreaming intercept hooks. Open Q: intercept vs. fork NCCL? |
| TorchFT evolves and overlaps our membership logic | C/E | Track TorchFT closely; design EPE to *delegate* to TorchFT elastic where it exists rather than duplicate. Open Q: is TorchFT a dependency or a competitor surface? |
| Central classifier adds latency to B0 fast path | A/B | Fast-path autonomy (§3.1). Open Q: what's the max tolerable classifier latency before transport recovery degrades? |
| Correlation window mis-tuned → either thrash or slow B4 detection | E | Make W,N policy-tunable; learn from production traces (§7). |
| Multi-NIC hardware not present | B | Document as prerequisite; graceful degradation to B1+ when absent. |
| MeCeFO approximation compounds over many fallbacks | C | Bound consecutive-fallback iterations; force a full-fidelity checkpoint after K fallback windows. Open Q: what's K? |
| URC cross-layer consensus is novel — TierCheck only did it within storage | D | Highest-research-risk net-new component. Prototype URC first among the net-new pieces. |

---

## 7. Where to extend beyond the three papers (build the *best* tool, not just the union)

The three papers are the floor. These additions are where AEGIS becomes defensible beyond "we stapled three repos together":

1. **Learned failure classifier.** FC starts rule-based but production fault traces are training data. A model that predicts blast radius (and even *predicts impending B4 from early B1 signals*) turns reactive escalation into predictive pre-staging. This is a genuine moat the papers don't touch.

2. **Predictive checkpoint pre-staging.** If FC predicts a rack is degrading, pre-warm Tier-3 base checkpoints *before* the B4 hits. Turns a cold remote-restore into a warm one. Directly attacks the most expensive recovery class.

3. **Cost-aware policy as a first-class optimizer.** Don't just route to the cheapest *correct* tier — route to the tier that minimizes expected $ given fault probabilities and current $/GPU-hr. The KPI meter (E3) feeds back into EPE. This makes the economics self-optimizing and is the cleanest sales story.

4. **Inference-path coverage.** R²CCL already covers serving (KV-cache reprocessing avoidance, DéjàVu comparison). MeCeFO/TierCheck are training-centric. Extending blast-radius routing to *inference* fault tolerance (request reprocessing, KV-cache replication) is a whole second market — and it connects directly to your KV-cache optimization thesis. Flag as a Phase 2 product line.

5. **TPU / non-NVIDIA attack surface.** R²CCL/NCCL are GPU-centric. A TPU collective backend is future adoption surface and reduces NVIDIA-dependency risk noted in §2.2.

6. **Straggler detection, not just hard failure.** The fabric-effects literature (e.g., "When Scaling Fails," 2603.04424) shows *soft* degradation — synchronization amplification, topology contention — wastes GPU-hours without any hard fault. A B-(-1) tier for *performance* faults (slow rank, congested link) that R²CCL's bandwidth-awareness could partially address. Open design space; potentially the highest-value extension because it catches waste the three papers ignore entirely.

7. **Observability as product.** VCCL/Mycroft-style μs-level collective monitoring folded into UTP gives operators a reason to deploy AEGIS even before a fault — continuous value, not just insurance. Lowers the "software-layer monetization unproven" GTM risk by making the tool useful every day, not just on failure days.

---

## 8. Suggested build phases

- **Phase 0 (foundation):** A1–A3 + `chaos-inject` harness + EPE stub + reproduce UT-B/C/D individually against each paper's own testbed. *Goal: prove we can reproduce all three primitives in isolation and emit unified telemetry.*
- **Phase 1 (compose):** Wire B/C/D under EPE; build URC (highest research risk — do early); pass IT-1 through IT-6. *Goal: the composition actually works under simultaneous/correlated faults.*
- **Phase 2 (economics):** E1–E4; ST-3 KPI measurement; produce the $/GPU-hour-saved number on a realistic trace. *Goal: the sales artifact exists and is real.*
- **Phase 3 (moat):** §7 extensions — learned classifier, predictive pre-staging, straggler tier, inference path. *Goal: defensibility beyond the union of three papers.*

### 8.1 Actual status vs. this plan (as of 2026-07-08)

The phase order above is the *intended* build order. Reality has diverged from it —
recorded here so this doc doesn't go stale the way `CLAUDE.md`'s phase table did.
Verify against the repo directly (test count, `bench/` contents) before trusting
this table for anything more than orientation; it will drift again.

| Phase | Planned scope | Actual state |
|-------|---------------|--------------|
| **Phase 0** | A1–A3, chaos-inject, EPE stub, UT-B/C/D reproductions | **Done, and grown past its original scope.** `aegis/` has UTP, FC, epoch service, EPE, and stub B/C/D layers exactly as specified. Test suite has grown from the original 59 tests to **138**, and integration coverage now runs **IT-1 through IT-7** (an IT-7 "transparency contract" test exists that predates any mention of it in this doc or `CLAUDE.md`). All recovery layers remain stubs — no real NCCL, GPU, or checkpoint I/O; the only sensor is `SyntheticSensor`. |
| **Phase 1** | Wire real R²CCL/MeCeFO/TierCheck under EPE; build real URC; pass IT-1–IT-6 under simultaneous/correlated faults | **Software-realistic composition done; hardware validation pending.** All three layers now do genuine work instead of `asyncio.sleep(0)`: `layers/transport.py` runs a real NIC migration state machine and real R2CC-Balance bandwidth-redistribution math on a pluggable `TransportBackend` (default `SimulatedTransportBackend`; hardware-pending `LinuxRDMABackend` stubbed for the IB cluster). `layers/compute.py` runs real MeCeFO tensor math via torch — truncated-SVD low-rank approximation, skip-connection pass-through, selective FFN recompute — device-agnostic (CUDA on the A100 cluster, MPS/CPU here) via `TorchMeCeFOBackend`. `layers/storage.py` does real file I/O: base+differential checkpoints (XOR-diff, zlib-compressed), SHA-256 integrity verification, and measured wall-clock timing, on a pluggable `CheckpointBackend` (peer/remote tiers are local directories standing in for a real peer node and S3/Lustre — `RemoteObjectStoreBackend` is a documented `NotImplementedError` extension point, not yet wired). `consensus/urc.py`'s epoch-reduction logic is now genuinely exercised: the EPE calls `report_epoch()` after every successful recovery and gates B2–B4 restores on `agree()`'s `min_valid_epoch` (see `policy/engine.py::_urc_gate`), so URC reconciles real cross-layer state instead of having nothing to reduce over. IT-1–IT-7 and 27 new unit tests (`test_transport_layer.py`, `test_compute_layer.py`, `test_storage_layer.py`, `test_urc_wiring.py`) pass — 165 tests total. **What's still hardware-pending, not yet done:** no real NCCL interception shim, no real RDMA/IB migration (`LinuxRDMABackend` untested — no RNICs on the dev machine), no real 8×A100 training job reproducing MeCeFO's 4.18% or R²CCL's <1%/<3%/85–89% numbers, no real S3/Lustre client, no TorchFT integration. This is real software composition validated on CPU/MPS; the A100-cluster hardware validation described in `eval_design.md`/`test_suite.md` §4.5 is the next gate. |
| **Phase 2** | E1–E4 policy/cost plane; ST-1–ST-3 KPI measurement on a realistic trace | **Partially prototyped, ahead of Phase 1.** A full benchmark framework now exists at `bench/` (not referenced elsewhere in this doc) — a `SimulationEngine`, fault traces (per-tier isolation + Poisson/burst/production mixed traces), a cost model, and system adapters for **AEGIS, TorchFT, R²CCL, MeCeFO, TierCheck, and a vanilla checkpoint-restart baseline**. Only the AEGIS adapter drives the real runtime (`AegisRuntime` + `ChaosHarness` + EPE audit log); the other five adapters are fixed cost-table simulations, not real integrations of those systems. The evaluation methodology this implements is specified in a companion doc, **`test_suite.md`**, which also names the target real-hardware cluster (SeaWulf's A100 partition) that Phase 1's "real cluster" validation is meant to run on — but no cluster driver (Slurm scripts, etc.) exists yet, so that half of `test_suite.md` is still design, not code. `eval_design.md` extends this further into a real-inference (GPT-OSS/GLM) evaluation, still pending your answers to its open items. |
| **Phase 3** | Learned classifier, predictive pre-staging, straggler tier, inference path | **Not started.** |

**Net effect:** the project did not follow its own build order, but Phase 1 has
now landed out of order alongside Phase 2's scaffolding. The `bench/` $/GPU-hr
numbers are still simulated for five of six systems compared, and the AEGIS
adapter's recovery-time numbers now come from software-realistic (not
hardware-measured) transport/compute/storage math rather than an instant
stub — a step closer to credible, but still not the hardware-grounded number
`ST-3` intends. The actual remaining blocker to that number is real A100/IB
cluster validation (`eval_design.md`/`test_suite.md` §4.5), not more software
composition — that part is now built.

---

## Appendix A — Source primitives (verified arXiv IDs)

- **R²CCL** — *Reliable and Resilient Collective Communication Library for LLM Training and Serving*, Wang, Yu, Xiong, Z. Liu (UMD), arXiv:2512.25059, 31 Dec 2025. (This is the "UMaryland network/transport paper" from the source notes — same work.)
- **MeCeFO** — *Enhancing LLM Training Robustness via Fault-Tolerant Optimization*, arXiv:2510.16415, 18 Oct 2025.
- **TierCheck** — *Tiered Checkpointing for Fault Tolerance in Large Language Model Training*, arXiv:2605.17821, 17 May 2026.

## Appendix B — Adjacent papers worth mining for §7 extensions

- *When Scaling Fails: Network and Fabric Effects on Distributed GPU Training Performance* — arXiv:2603.04424 (straggler / soft-degradation tier)
- *VCCL* — arXiv:2510.00991 (primary-backup QP, μs observability)
- *SHIFT* — arXiv:2512.11094 (cross-NIC RDMA fault tolerance, user-space)
- *Mycroft* — arXiv:2509.03018 (collective dependency tracing for root-cause)
- *LowDiff* — arXiv:2509.04084 (low-cost differential checkpointing — complements TierCheck D1/D4)