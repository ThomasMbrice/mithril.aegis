# AEGIS UI Layer — Implementation Progress

**Date:** 2026-06-30
**Status:** Complete — 102 tests pass (59 pre-existing + 43 new)
**Reference:** ui.md v0.1

---

## What was implemented

### New files

| File | Purpose |
|------|---------|
| `aegis/_state.py` | Shared mutable module state (runtime, hooks, mode, event loop, thread). Exists solely to prevent circular imports between `__init__.py` and the `checkpoint`/`transport` submodules. |
| `aegis/hooks.py` | `HookRegistry` — tracks which of the five interception hooks (`transport`, `compute`, `checkpoint`, `telemetry`, `policy`) are installed. Validates hook names at install time so typos surface at `init()`, not at fault time. |
| `aegis/config.py` | `load_policy(path)` — reads `aegis.yaml` or `aegis.json` into `OperatorPolicy`. Returns safe defaults when the file is absent. Raises a clear error if pyyaml is needed but not installed. Maps the YAML schema (e.g. `escalation.correlation_window_ms` ÷ 1000 → `correlation_window_secs`). |
| `aegis/checkpoint.py` | T2 explicit checkpoint API: `save(state, tier="auto")` and `restore(epoch="latest_valid")`. Delegates to `runtime.storage`. Raises `RuntimeError` with a clear message if called before `aegis.init()`. |
| `aegis/transport.py` | T2 explicit transport API: `set_fast_path(bool)` / `get_fast_path()`. Mutates `runtime.epe._policy.allow_b0_fast_path` on the live policy object. |
| `aegis/cli.py` | `aegis run` — thin CLI shim that translates `aegis run [args]` → `python -m torch.distributed.run [args]`. Registered as the `aegis` entry point in pyproject.toml. |
| `tests/unit/test_ui.py` | 25 unit tests for the public API surface. |
| `tests/integration/test_it7_transparency_contract.py` | 18 integration tests for the §5 transparency contract. |

### Modified files

| File | Change |
|------|--------|
| `aegis/__init__.py` | Complete rewrite. Adds `init()`, `status()`, `explain()`, `disable()`, `_reset()`, `AegisInitError`. Re-exports `checkpoint`, `transport`, `policy`. |
| `aegis/policy/__init__.py` | Added `set(key, value)` — live dotted-key policy updates (e.g. `aegis.policy.set("escalation.correlation_window_ms", 3000)`). |
| `aegis/policy/dsl.py` | Added `economics_policy: str = "minimize_expected_cost"` field to `OperatorPolicy`. |
| `aegis/policy/engine.py` | Added `_observe_only: bool = False` instance attribute and a short-circuit guard in `_on_event` that skips `_route()` when observe_only is True. **`_route()` is untouched** — IT-6 escalation invariant preserved. |
| `pyproject.toml` | Added `pyyaml>=6.0` to `dependencies`. Added `[project.scripts]` entry: `aegis = "aegis.cli:main"`. |

---

## Key design decisions made

**Background event loop.** `aegis.init()` is synchronous (training scripts don't want to `await` their init call). The `AegisRuntime` is async. Resolution: `init()` spawns a daemon thread running `asyncio.run_forever()` and submits coroutines to it via `asyncio.run_coroutine_threadsafe()`. The thread is torn down by `disable()`.

**observe_only via EPE flag, not layer wrapping.** Rather than wrapping all layer `recover()` calls in no-op proxies (fragile, invasive), `_observe_only=True` is set on the EPE and checked in `_on_event` before `_route()` is called. The fault is still classified, the history is still populated (so `explain()` works), but no recovery action is dispatched. This keeps `_route()` untouched.

**`_state.py` for shared state.** `checkpoint.py` and `transport.py` need access to the live `AegisRuntime`. Importing `aegis._runtime` directly would be a circular import (`__init__.py` imports `checkpoint`). The `_state` module breaks the cycle: both `__init__.py` and the submodules import `_state` and read from it at call time (not import time).

**YAML + JSON config both supported.** `load_policy()` uses the file extension to decide the parser. JSON uses stdlib `json`; YAML requires `pyyaml` but gives a clear install instruction if missing.

---

## How to test

### Run everything

```bash
/tmp/aegis-venv/bin/pytest                     # full suite — 102 tests, ~3 s
/tmp/aegis-venv/bin/pytest tests/unit/test_ui.py
/tmp/aegis-venv/bin/pytest tests/integration/test_it7_transparency_contract.py
```

### Manual smoke test — T0 integration path

```python
import aegis

aegis.init()                             # T0: safe defaults, all 5 hooks active
print(aegis.status())                    # {"initialized": True, "mode": "active", ...}
print(aegis.explain())                   # {"message": "No faults processed yet.", ...}
aegis.disable()                          # kill switch — runtime stops
```

### Manual smoke test — T1 policy file

Create `aegis.yaml`:
```yaml
version: 0.1
tiers:
  b0_transport: { enabled: true, fast_path: false }
  b1_compute:   { enabled: true, max_consecutive_fallbacks: 5 }
escalation:
  correlation_window_ms: 3000
  correlation_node_threshold: 4
economics:
  gpu_hourly_cost_usd: 3.50
  policy: correctness_first
```

```python
import aegis
aegis.init()                             # picks up aegis.yaml automatically
print(aegis.status()["kpi"])
```

### Manual smoke test — T2 escape hatches

```python
import aegis
from aegis import checkpoint, transport, policy

# Disable one hook, keep rest transparent
aegis.init(disable=["checkpoint"])

# Drive checkpoint explicitly
checkpoint.save({"step": 1000}, tier="auto")
state = checkpoint.restore()

# Override fast-path
transport.set_fast_path(False)

# Live policy update
policy.set("economics.gpu_hourly_cost_usd", 3.50)
policy.set("escalation.correlation_window_ms", 2000)

# Shadow mode
aegis.disable()
aegis.init(mode="observe_only")         # AEGIS classifies faults but takes no action
print(aegis.explain()["observe_only"])  # True
```

### Manual smoke test — CLI

```bash
# torchrun drop-in (requires torch installed)
aegis run --nproc_per_node=8 train.py

# Without torch — prints a clear install instruction
aegis run --nproc_per_node=8 train.py
# aegis run: torchrun not found. Install PyTorch: pip install torch
```

---

## What the new tests cover

### `tests/unit/test_ui.py` (25 tests)

| Group | Tests |
|-------|-------|
| `init()` | default init; observe_only mode; disable=[hook]; invalid hook name → ValueError; invalid mode → ValueError |
| `status()` | all expected keys present; raises before init |
| `explain()` | no-fault message; observe_only flag in result; active mode sets flag False; raises before init |
| `disable()` | clears `_state.initialized`; no-op before init; re-init after disable works |
| `policy.set()` | known key updates live policy; economics_policy string key; correlation_window_ms conversion (ms→s); unknown key → ValueError; raises before init |
| `checkpoint` | raises before init (save and restore); save+restore round-trip after init |
| `transport` | raises before init; set_fast_path updates live policy; get_fast_path raises before init |
| `config` | missing file → OperatorPolicy defaults; valid JSON; valid YAML; policy= kwarg bypasses file |

### `tests/integration/test_it7_transparency_contract.py` (18 tests)

Maps directly to the five §5 promises:

| Promise | Tests |
|---------|-------|
| #1 No silent inactivity | All hooks active after `init()`; disabled hook is absent from status; invalid hook raises immediately |
| #3 Always escapable | `disable()` stops runtime and clears state; `disable()` is idempotent |
| #4 Always inspectable | `status()` has correct shape; `explain()` handles no-fault gracefully |
| observe_only contract | Mode recorded in state; EPE history populated but `_route()` not called; `explain()["observe_only"]` is True; active vs observe_only comparison; full round-trip |
| Promise #5 Fail-safe (via existing UTP test) | Subscriber exception does not stop dispatch (already in `test_utp.py::test_subscriber_exception_does_not_stop_dispatch`; IT-7 references it and adds the EPE layer above it) |

---

## What is deferred (per ui.md §7)

- **Real PyTorch hook installation** — `aegis.init()` registers the hook names in the registry but does not yet patch `dist.init_process_group`, wrap `torch.save`, or intercept NCCL. These are Phase 1 wiring tasks that require PyTorch as a dependency.
- **`aegis run` topology flags** — `--config`, `--profile` flags on the CLI are deferred past v1.
- **Framework auto-detection** (DDP vs FSDP vs raw `dist`) — the open question in ui.md §7 is still open; the `init()` signature has a `policy=` escape hatch but no `framework=` override yet.
- **Daemon/cross-job coordination** — deferred past v1 per design.
- **`observe_only` cost projection** — `explain()` in observe_only mode does not yet estimate $/GPU-hr saved (the KPI meter is wired but not called from the observe_only code path).
