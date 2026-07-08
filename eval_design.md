# AEGIS Real-Inference Fault-Tolerance Evaluation — Design Doc

**Design Document — v0.1 (draft, awaiting review)**
Owner: Thomas
Status: Pre-implementation — open items in §8 block start
Last updated: 2026-07-07

Relationship to `design.md`: that doc is the architecture spec for the full system
(Phase 0–3). This doc specifies **one evaluation**: run AEGIS against real GPT-OSS
and GLM inference workloads on the university A100 cluster, compare it to TorchFT,
and report the metrics AEGIS is meant to sell (§5 there, `aegis/kpi.py` here). It is
scoped as new work under design.md's Phase 3 item §7.4 ("Inference-path coverage"),
pulled forward because you want to test against something real now rather than wait
for the full training-side build-out.

---

## 0. Reality check before anything else

Per `CLAUDE.md` and the current source, AEGIS today is **Phase 0**: every recovery
layer (`transport.py`, `compute.py`, `storage.py`) is a stub whose `recover()` does
`await asyncio.sleep(0)` and returns a canned `RecoveryResult`. The only sensor that
exists is `SyntheticSensor`, which exists purely to let tests and `chaos_inject`
hand-write `TelemetryEvent`s onto the bus. Nothing in the codebase today watches a
real GPU, process, or NIC. And the whole vocabulary (`node`, `rank`, `nic_id`,
`rack_id`, DP/TP/PP) is training-shaped — there is no concept of "inference request"
anywhere.

You picked, when I flagged this:
- **Build real inference sensors** (not just replay synthetic events) — so detection
  and classification are genuine, even though the recovery *mechanics* stay close to
  stub-level (process restart / replica failover, not real NCCL shims or low-rank
  gradient approximation — those are separate, much larger efforts from design.md
  §4 Layers B/C/D that are out of scope here).
- **Engineer a real, apples-to-apples benchmark against TorchFT**, not a feature table.
- Cluster: **university A100s**, exact GPU/node count TBD (§8).

Everything below is built on those three choices. The main intellectual work this
doc does that `design.md` doesn't is §1: translating the training-oriented
blast-radius model into an inference-serving one, because none of B0–B4 as currently
defined (NIC flap *during AllReduce*, node death *absorbed by gradient math*,
checkpoint tiers *for training state*) literally applies to serving.

---

## 1. Inference blast-radius mapping (new)

Same five tiers, same one-directional-escalation invariant, redefined for a
request-serving workload instead of a training step.

| Tier | Training fault (design.md) | Inference-serving analog | Real signal (sensor) | Recovery action under test |
|------|----------------------------|---------------------------|-----------------------|------------------------------|
| B0 | NIC/link flap during collective | Link flap during tensor-parallel all-reduce or cross-node KV-cache transfer | RDMA/NIC counters, link-state via `ethtool`/netlink | Migrate to backup NIC; in-flight requests must not be dropped |
| B1 | Single GPU/node death | One inference replica (GPU worker) process dies mid-request | Process heartbeat loss, CUDA "device lost", worker crash exception | In-flight + queued requests on that replica fail over to a healthy replica; partial generations are **not** resumable (see below) — requests are resubmitted from scratch |
| B2 | CUDA kernel crash, recoverable in place | Worker hangs or crashes but the GPU/process itself is salvageable | Watchdog timeout / caught exception, no device loss | Restart worker process in place on same GPU, reload weights from local cache, resume serving |
| B3 | Node-level state loss | Node fails entirely; replica must be rescheduled elsewhere | Node heartbeat loss | Cold-start a replacement replica on a spare node; requests queue or redirect meanwhile |
| B4 | Rack/cluster outage | A whole partition of the pool serving this model goes down | Correlated multi-node loss within the existing correlation window (§3.3 of design.md, unchanged) | Failover to a separate replica pool if one exists, else reject-and-signal upstream |

Two deliberate departures from the training semantics, both need your sign-off:

- **No fidelity_flag equivalent.** MeCeFO's `degraded=True` means "this checkpoint
  encodes an approximated gradient." There is no equivalent approximate-compute path
  for autoregressive decoding — you either serve a token correctly or you don't. I'm
  proposing `degraded=True` instead means **"this request was served by a
  capacity-reduced pool"** (remaining replicas absorbing extra load → higher latency,
  same output correctness). Different meaning, same field, documented as a rename in
  the test-scoped fork of `RecoveryResult` usage. Flag if you want a different
  semantic.
- **No partial-generation recovery.** If a replica dies mid-stream, the tokens
  already streamed to the client are unrecoverable-in-place — AEGIS's job is to fail
  the request over fast, not to reconstruct KV-cache state on a new device. This is a
  scope decision (recomputing KV-cache from a checkpoint mid-generation is a real
  research problem, not something to build for this eval). If you want prefix-cache
  resumption in scope, say so — it changes §6 substantially.

---

## 2. What actually gets built (new engineering, test-scoped)

None of this becomes the production R²CCL/MeCeFO/TierCheck implementations from
design.md §4 — it's the minimum real plumbing needed so "AEGIS handled a real fault"
is a true statement instead of a replayed one.

| Component | New code | Notes |
|---|---|---|
| `ProcessHeartbeatSensor` | new, `aegis/telemetry/sensors.py` | Watches inference worker liveness (heartbeat over a control socket or PID poll) → B1/B2 signals |
| `CUDADeviceSensor` | new | Watches for CUDA device-lost / OOM-killed-rank via `pynvml` or caught exceptions in the worker → B1 signal |
| `NodeHeartbeatSensor` | new | Cross-node liveness (ssh/heartbeat) → B3/B4 signals |
| `NICLinkSensor` | new | Link-state polling, or driven by injected `tc netem` faults for B0 |
| Request router / proxy | new | Sits in front of N replicas per model; AEGIS's `ComputeLayer.recover()` marks a replica unhealthy and the router stops sending it traffic — this is what makes B1/B3 failover observable and measurable |
| `TransportLayer.recover()` (test-scoped) | modified | Verify migration/continuity instead of no-op sleep |
| `ComputeLayer.recover()` (test-scoped) | modified | Real replica-eviction + failover call into the router |
| `StorageLayer.recover()` (test-scoped) | modified | B2: in-place worker restart. B3: spin up replacement replica on a spare node. B4: failover to separate pool or reject |
| `kpi.py` baseline table | modified | `_BASELINE_RECOVERY_SECS` is currently tuned for training checkpoint-restart (60s–3600s). Needs inference-appropriate numbers — empirically measured cold model-reload + KV-cache-reinit time per model size, not reused training constants |
| TorchFT harness | new, separate script | Wrap the same N replicas in a TorchFT `Manager` + Lighthouse quorum instead of AEGIS's EPE; kill the same fault, let TorchFT's own membership/reconfig handle it |

The Failure Classifier (`classifier.py`) and EPE (`policy/engine.py`) are reused
as-is — the signal→tier table and escalation logic don't need to change, only what
feeds them and what they call.

---

## 3. Workload

- **Models:** `openai/gpt-oss-20b` and `openai/gpt-oss-120b` (capacity permitting);
  GLM side needs a concrete checkpoint pick — "GLM" alone is ambiguous between
  GLM-4-9B-chat, GLM-4.5-Air, and full GLM-4.5/4.6 (355B-class, likely out of reach
  on a university allocation). **Open item, see §8.**
- **Serving framework:** vLLM. Both model families are natively supported, it
  handles tensor/pipeline parallelism across A100s, and exposes an OpenAI-compatible
  endpoint so one load generator drives every condition identically.
- **Context sizes:** 1K / 8K / 32K / 128K input tokens (or the model's max, if lower),
  fixed 256-token generation to isolate prefill-cost scaling from decode effects,
  plus one long-generation (2K+ output) variant to stress KV-cache growth under
  fault conditions specifically.
- **Load generator:** fixed-concurrency request stream (e.g. `vllm bench serve`
  or a small custom OpenAI-client script) — same script drives baseline, TorchFT,
  and AEGIS conditions so request timing isn't a confound.

---

## 4. Test matrix — two sections as requested

### Section A — Natural execution (no injected faults)

**Goal:** measure how often AEGIS triggers when nothing is actually wrong. Target
is zero; any nonzero trigger during a clean run is a false positive and a bug in the
classifier or a sensor, not a "recovery event."

For each {model} × {context size}: run three conditions back to back —
1. Bare vLLM, no AEGIS, no TorchFT (control)
2. AEGIS sensors + EPE active (`observe_only=False`), no faults injected
3. TorchFT-wrapped, no faults injected

Sustained soak per config (duration sized in §8, not guessed) at fixed
QPS/concurrency, long enough that a rare false trigger would actually surface.

**Metrics recorded:**
- AEGIS trigger count (target: 0) and, if nonzero, which signal/tier fired and why
- Added latency vs control: p50/p99 TTFT, p50/p99 inter-token latency
- Added throughput cost vs control: tokens/sec, req/sec
- Sensor/UTP/EPE process overhead: CPU%, RSS, UTP publish latency p99

### Section B — Fault-injected execution

**Goal:** for each blast-radius tier that's actually inducible on this cluster,
measure detection latency, classification correctness, recovery time, and service
impact — for AEGIS, TorchFT, and no-fault-tolerance, on the same fault.

For each {fault tier} × {model} × {context size}: 3 conditions × N trials
(propose N=10 for variance on recovery-time measurements):
1. **No fault tolerance** — fault fires, requests on the affected replica simply fail; this is the "what happens if you do nothing" floor
2. **TorchFT-wrapped** — same fault, TorchFT's quorum/reconfig handles it
3. **AEGIS-supervised** — same fault, sensor → classifier → EPE → recovery layer handles it

**Fault injection is physical, not synthetic**, wherever the cluster allows it —
this is the point of testing against something real:
- B0: `tc netem`/link-down on one NIC path
- B1/B2: `kill -9` / SIGKILL the worker process, or force a CUDA OOM
- B3: drain/kill an entire node
- B4: correlated kill across multiple nodes in the same rack/partition within the
  EPE's correlation window — **likely not safely inducible on shared university
  infra** (would take down a real rack partition). Recommend simulating B4 only
  (inject the correlated signals directly, don't actually take hardware down) and
  labeling it clearly as simulated in every report table, in contrast to B0–B3 which
  are physically induced. Confirm this is acceptable — see §8.

**Metrics recorded, per fault class, per condition:**

*Detection & classification (AEGIS-only, no TorchFT equivalent):*
- Time-to-detect: fault occurrence → sensor publishes `TelemetryEvent`
- Time-to-classify: event → tier assigned
- Classification accuracy against ground truth (the dangerous-direction check from
  `classifier.py`'s own test comment applies here too: never misclassify a B3/B4
  event as B1 or lower)
- Escalation correctness: never de-escalates (same invariant IT-6 already guards)

*Recovery & availability (comparable across all three conditions):*
- Recovery time: fault → service fully restored to pre-fault capacity
- Requests failed / lost during the fault window
- Requests degraded (served at reduced capacity) — count and duration
- Goodput (successful completions / wall-clock time) during and after the fault
- Tail latency during the fault window: p99 TTFT, p99 inter-token latency
- Time to return to pre-fault steady-state throughput

*Cost (extends `aegis/kpi.py`):*
- `$/GPU-hr saved` using the existing `KPIMeter` formula, but with the
  `_BASELINE_RECOVERY_SECS` table replaced by measured inference cold-restart times
  (see §2) instead of the current training-tuned constants
- Same computation applied to the TorchFT condition and the no-FT condition, so all
  three land in one comparison table, not just AEGIS in isolation

**Final comparison table shape** (one row per fault tier, repeated per model ×
context size):

| Fault tier | Condition | Detect (s) | Classify (s) | Recovery (s) | Reqs failed | Reqs degraded | p99 latency during fault | $/GPU-hr saved |
|---|---|---|---|---|---|---|---|---|
| B1 | No-FT | – | – | – | | | | – |
| B1 | TorchFT | n/a | n/a | | | | | |
| B1 | AEGIS | | | | | | | |

---

## 5. Metrics summary (answering "what metrics do you use")

Four groups, reused/extended from what already exists in the codebase plus what's
net-new for inference:

1. **Detection/classification metrics** — new, AEGIS-only, no analog in `kpi.py` today. Time-to-detect, time-to-classify, classification accuracy, escalation-invariant compliance, false-positive trigger rate (Section A).
2. **Recovery/availability metrics** — new, but comparable across AEGIS/TorchFT/no-FT: recovery time, requests failed/degraded, goodput, tail latency during fault, time-to-steady-state.
3. **Cost/KPI metrics** — reuses `aegis/kpi.py`'s existing `$/GPU-hr saved` formula unchanged, with an inference-appropriate baseline-recovery-time table replacing the training one.
4. **System overhead metrics** — new, Section A primarily: sensor/UTP/EPE CPU, memory, and publish-latency overhead versus bare vLLM.

---

## 6. What this eval deliberately does *not* prove

Worth stating explicitly so results aren't over-claimed: this does not validate
R²CCL's reported <3% inference overhead, MeCeFO's 4.18% throughput drop, or
TierCheck's <10s checkpoint numbers — those require the real primitives from
design.md §4, which are still Phase 1+ work. This eval validates the **detection →
classification → routing plane** (the actual net-new product per design.md §2.3)
against a real workload, using test-scoped recovery mechanics that are simpler than
the eventual production primitives. That's a legitimate and useful result, but the
report should say what it is and isn't.

---

## 7. Timeline shape (no dates until §8 is resolved)

1. Build the four new sensors + router + test-scoped recovery layers (§2)
2. Stand up vLLM serving for both model families at all context sizes on the
   allocated hardware; validate baseline (no AEGIS, no TorchFT) performance numbers
3. Build the TorchFT replica harness for the same topology
4. Run Section A (natural execution) across all model × context configs
5. Run Section B (fault injection) across all tier × model × context configs
6. Produce the comparison tables in §4/§5

---

## 8. Open items — need your answer before implementation starts

1. **Exact GPU/node allocation** on the cluster (count, A100 memory size, whether
   nodes span racks) — determines replica counts, which model sizes are feasible,
   and whether B4 can be even simulated convincingly.
2. **Exact GLM checkpoint(s)** to use — GLM-4-9B-chat and GLM-4.5-Air are both
   plausible on modest A100 allocations; full GLM-4.5/4.6 likely isn't. Pick one.
3. **Job orchestration substrate** on the cluster (Slurm? Kubernetes? bare SSH?) —
   changes how "spin up a replacement replica on a spare node" (B3) is actually
   implemented.
4. **Is a request router/proxy already available**, or does it need to be built
   from scratch for this eval?
5. **B4 handling** — confirm simulate-only is acceptable, since physically inducing
   a correlated rack-level outage on shared research infra is likely not something
   you're able (or willing) to do.
6. **Section A soak duration** — how long/how many requests per config to get a
   statistically meaningful "AEGIS essentially never false-triggers" claim.
