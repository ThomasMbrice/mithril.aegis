# AEGIS — User Interface Design: Transparent Integration with Escape Hatch

**Design Document — v0.1 (working draft)**
Owner: Thomas
Status: Pre-implementation design
Scope: How a user integrates the AEGIS fault-tolerance runtime into an existing training job.
Last updated: 2026-06-29

> Companion to the AEGIS core design doc. That doc covers *what the runtime does* (blast-radius routing across R²CCL/MeCeFO/TierCheck). This doc covers *how a user turns it on* and *how they take back control when they need to*.

---

## 0. Decision

**Primary surface: transparent integration.** The user changes their launch command and adds one init line; existing PyTorch training code runs unchanged. AEGIS intercepts the process group and collective path beneath the user's code.

**Escape hatch: explicit API, always available.** Every transparent behavior has an explicit, documented override. Nothing AEGIS does transparently is a black box the user can't reach into, disable, or replace.

The design tension we're resolving: transparent integration demos well and lowers adoption friction (the thing that matters for landing design partners), but transparent magic is hell to debug when it misbehaves — and a fault-tolerance runtime *will* be blamed for every weird hang. The escape hatch is what makes transparent safe to ship: the user is never trapped inside the magic.

---

## 1. The three integration tiers (user-facing)

A user can adopt AEGIS at exactly three depths. This is the whole interface model.

| Tier | What the user writes | What AEGIS does | Who this is for |
|------|---------------------|-----------------|-----------------|
| **T0 — Zero-touch** | swap `torchrun` → `aegis run`, add `aegis.init()` | intercepts process group, collectives, checkpointing transparently; reads policy from `aegis.yaml` | design partners, first demo, "does it even work on my job" |
| **T1 — Policy-tuned** | T0 + edit `aegis.yaml` | same interception, operator-tuned escalation/cost thresholds | ops engineers tuning $/GPU-hr behavior without touching training code |
| **T2 — Explicit / escape hatch** | call AEGIS APIs directly; disable specific transparent hooks | user drives specific layers manually; AEGIS handles the rest | debugging, custom training loops, advanced users, conflict resolution |

**Key property:** these compose. A user can run T0 transparent for everything *except* checkpointing, which they drive explicitly via T2 — partial escape, not all-or-nothing.

---

## 2. T0 — Zero-touch (the demo path)

### 2.1 What the user does

```bash
# before
torchrun --nproc_per_node=8 train.py

# after
aegis run --nproc_per_node=8 train.py
```

```python
# in train.py — one line, early, before process group init
import aegis
aegis.init()              # reads ./aegis.yaml if present, else safe defaults

# ... existing PyTorch training code, completely unchanged ...
# dist.init_process_group(...), DDP/FSDP, optimizer.step(), torch.save(...)
# all intercepted transparently
```

That's the entire integration. Three changes: launcher swap, one import, one init call.

### 2.2 What `aegis.init()` does under the hood

1. Installs the **NCCL interception shim** (B0 transport, R²CCL) on the collective path.
2. Wraps `ProcessGroup` creation so membership changes route through the Escalation Policy Engine.
3. Registers the **Unified Telemetry Plane** sensors on each primitive's native detector.
4. Hooks `torch.save` / checkpoint calls into the **TierCheck** tiered path (B2–B4).
5. Hooks the autograd/optimizer boundary for **MeCeFO** neighbor-absorb (B1).
6. Loads policy from `aegis.yaml` (or safe defaults).

**Critical constraint:** `aegis.init()` must be called *before* `dist.init_process_group()`. The shim has to be in place before the process group exists. If called after, AEGIS detects it and either (a) re-wraps the existing group if safe, or (b) raises a clear error telling the user to move the call up. **Never silently no-op** — a fault-tolerance library that silently isn't active is the worst possible failure.

### 2.3 Safe defaults (when no `aegis.yaml`)

- All five blast-radius tiers active.
- Conservative escalation (prefer correctness over aggressive cost-saving).
- B0 fast-path autonomy on (transport recovers in ms–s without waiting on classifier).
- Cost/KPI meter on, reporting only (doesn't influence routing until policy enables it).
- Fidelity flagging on (checkpoints record degraded-compute state).

---

## 3. T1 — Policy file (`aegis.yaml`)

The escalation policy and cost thresholds live in declarative YAML, not code, not CLI flags. Operators tune fault behavior without touching the training script.

```yaml
# aegis.yaml — illustrative shape, not final schema
version: 0.1

tiers:
  b0_transport:  { enabled: true,  fast_path: true }
  b1_compute:    { enabled: true,  max_consecutive_fallbacks: 8 }   # the "K" from core doc §6
  b2_b3_b4_storage:
    tier1_local:  { enabled: true }
    tier2_peer:   { enabled: true }
    tier3_remote: { enabled: true, backend: "s3://my-bucket/ckpt" }

escalation:
  correlation_window_ms: 5000     # the "W" from core doc §3.3
  correlation_node_threshold: 4   # the "N"
  direction: one_way_only         # invariant; do not allow de-escalation

economics:
  gpu_hourly_cost_usd: 2.35       # H100 1-yr rental
  policy: minimize_expected_cost  # | correctness_first | latency_first
  max_recovery_spend_ratio: 0.5   # never spend >50% of the GPU-hrs you'd save

telemetry:
  emit: true
  sink: "stdout"                  # | file | otlp endpoint
```

The policy file is also the natural artifact for **per-cluster profiles** — a design partner running on single-plane single-NIC hardware sets `b0_transport.enabled: false` (no spare NIC to migrate onto), and faults fall through to B1+ gracefully.

---

## 4. T2 — The escape hatch (explicit API)

This is the half of the decision that makes transparent safe. **Every transparent behavior is reachable, overridable, and disablable explicitly.**

### 4.1 Three escape mechanisms

**(a) Disable a hook, keep the rest transparent.**
```python
aegis.init(disable=["checkpoint"])     # AEGIS does B0/B1 transparently;
                                        # user drives torch.save themselves
```
Granularity matches the layer model: `transport`, `compute`, `checkpoint`, `telemetry`, `policy`.

**(b) Drive a layer explicitly.**
```python
from aegis import checkpoint, transport, policy

# explicit tiered checkpoint instead of intercepting torch.save
checkpoint.save(state, tier="auto")          # or tier="local"|"peer"|"remote"
checkpoint.restore(epoch="latest_valid")

# force a transport decision
transport.set_fast_path(False)               # route B0 through classifier for debugging

# override policy at runtime
policy.set("economics.policy", "correctness_first")
```

**(c) Full manual mode (escape entirely).**
```python
aegis.init(mode="observe_only")    # telemetry + KPI meter run, but AEGIS
                                    # takes NO recovery action. User sees what
                                    # AEGIS *would* do without it doing it.
```
`observe_only` is the single most valuable debugging and trust-building feature in the whole interface. It lets a skeptical design partner run AEGIS in shadow mode — seeing the blast-radius classification and the $/GPU-hr it *would* have saved — before granting it control of their job. Lead the sales demo with this.

### 4.2 Introspection — never a black box

```python
aegis.status()        # active hooks, current epoch, tier states
aegis.explain()       # for the last fault: what was classified, which tier
                      # handled it, what it cost, why that tier
aegis.disable()       # kill switch — AEGIS goes fully passive, training
                      # continues on native PyTorch path
```

`aegis.explain()` is the answer to "AEGIS hung my job" — it tells the user exactly what AEGIS did and why, turning the worst-case debugging session into a single call.

---

## 5. The transparency contract (what we promise the user)

Because transparent interception is invasive, we make explicit promises and document them as the contract:

1. **No silent inactivity.** If AEGIS can't install a hook, it errors loudly. It never pretends to protect a job it isn't protecting.
2. **No silent semantic change.** AEGIS never alters numerics the user can observe *without* recording it (the `fidelity_flag` on MeCeFO-degraded checkpoints is the canonical example).
3. **Always escapable.** `aegis.disable()` returns the job to the native PyTorch path at any time.
4. **Always inspectable.** `aegis.status()` / `aegis.explain()` expose every decision.
5. **Fail-safe, not fail-active.** If AEGIS itself errors internally, it falls back to native behavior (let PyTorch's own checkpoint-restart take over) rather than crashing the job. A buggy fault-tolerance layer must never be *less* reliable than no layer.

Promise #5 is non-negotiable and worth a dedicated test suite: inject faults *into AEGIS itself* and verify the job survives on the native path.

---

## 6. Why not the alternatives (recorded for posterity)

- **Pure explicit library (`aegis.wrap(model, optimizer)` only):** cleaner and easier to debug, but every design partner has to thread AEGIS objects through their training loop — friction at exactly the moment we're trying to land them. Rejected as *primary*, retained as the T2 escape hatch.
- **Daemon / sidecar server:** premature for MVP. Fault tolerance must live in-process with the training loop (the collective path and process group are in-process). A separate daemon adds an IPC boundary and a second failure domain for no MVP benefit. Revisit only if we later need cross-job coordination.
- **CLI-only:** can't intercept the in-process collective path from outside. CLI stays thin — launcher + chaos-inject + KPI readout — delegating real work to the library.

---

## 7. MVP cut line

**Ship in v1:**
- `aegis run` launcher (torchrun drop-in)
- `aegis.init()` transparent interception, all five tiers
- `aegis.yaml` policy file
- Escape hatch: `disable=[...]`, `mode="observe_only"`, `aegis.disable()`, `aegis.status()`, `aegis.explain()`
- The transparency contract test suite (esp. promise #5)

**Defer past v1:**
- Daemon/cross-job coordination
- Learned classifier / predictive pre-staging (core doc §7)
- Non-PyTorch frameworks (JAX, etc.)
- Fine-grained per-tier explicit drivers beyond checkpoint (add as partners ask)

**One decision still open:** does `aegis.init()` auto-detect the framework (DDP vs FSDP vs raw `dist`) or require the user to declare it? Auto-detect is more transparent and demos better; declared is more robust. Lean auto-detect with a `framework=` override in the escape hatch — consistent with the rest of this design.