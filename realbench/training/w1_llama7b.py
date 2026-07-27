"""
Real W1 training entrypoint — the trust-anchor / fault-injection target for
the real-cluster harness (test_suite.md §4.5, W1 = "LLaMA-7B, DP+PP, 8xA100,
MeCeFO's exact testbed").

Launch via ``aegis run`` (aegis/cli.py, unmodified — a thin torchrun wrapper):

    aegis run --nproc_per_node=8 -m realbench.training.w1_llama7b -- \\
        --steps 6000 --log-dir /path/to/run --aegis-enabled

Same script, one flag (``--aegis-enabled``), serves both the AEGIS-on run and
the bare-torchrun control — so Phase 0's "identical software stack" control
(test_suite.md §6.4) is trivially satisfied by construction; there is no
second script that could drift from this one.

KNOWN, DELIBERATE SCOPE LIMITS (see also realbench/honesty.py):

  * **DP(FSDP)-only, not DP+PP.** Real pipeline parallelism (stage
    partitioning, micro-batch scheduling) is a substantial separate effort
    from what's under test here (AEGIS's fault detection/recovery
    integration). FSDP shards optimizer/gradient/parameter state across the
    8 GPUs, which is "DP" in the sense that matters for this deliverable,
    but is NOT a reproduction of MeCeFO's own DP+PP testbed. Report tables
    must read this field from ``run_manifest.json``'s ``"parallelism"`` key
    (literally ``"DP(FSDP)"``) rather than assuming W1's config-table string.
  * **No real cross-rank recovery.** When ``--aegis-enabled`` and a peer
    dies, ``aegis.layers.compute.ComputeLayer.recover()`` genuinely fires
    (real low-rank/skip-connection torch math) and is measured/logged, but
    it does NOT reconfigure the ``torch.distributed`` process group or
    replace the dead rank — training on the surviving ranks cannot actually
    continue past the next collective. This script therefore does not wait
    forever: ``--collective-timeout-secs`` (default 30s) bounds how long a
    hung collective is tolerated before the process gives up and writes a
    ``training_halt.jsonl`` record explaining why. This applies identically
    whether or not ``--aegis-enabled`` is set, so the "no fault tolerance"
    and AEGIS conditions are handled by the same code path — see
    ``realbench/phase1_per_tier/report_phase1.py`` for how the report reads
    this record instead of assuming the job resumed.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from realbench.training.step_log import HeartbeatWriter, StepLogWriter, append_jsonl

# Real W1 dims — LLaMA-7B (standard MHA, no GQA; matches the original 7B config).
REAL_MODEL_DIMS = dict(
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    intermediate_size=11008,
    vocab_size=32000,
)

# Debug-only dims for local smoke-testing this script's plumbing (aegis wiring,
# step/heartbeat logging, FSDP wrap, collective-timeout handling) on a laptop
# with 1 CPU/MPS "rank" — NEVER used for a real W1 run. Real runs must not
# pass --tiny; run_manifest.json records which dims were actually used so a
# report can never confuse a smoke-test run for a real one.
TINY_MODEL_DIMS = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128,
    vocab_size=100,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                    help="Reserved for future TierCheck integration; not used by this script.")
    p.add_argument("--aegis-enabled", action="store_true")
    p.add_argument("--aegis-policy-path", type=str, default="aegis.yaml")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--heartbeat-timeout-secs", type=float, default=5.0)
    p.add_argument("--collective-timeout-secs", type=float, default=30.0,
                    help="How long a hung collective is tolerated mid-training (after a real "
                         "peer kill) before the process halts. Applied AFTER setup completes — "
                         "see --setup-timeout-secs for rendezvous/model-build/FSDP-wrap, which "
                         "need a much longer budget than a mid-training fault does.")
    p.add_argument("--setup-timeout-secs", type=float, default=600.0,
                    help="Timeout for process-group rendezvous + model construction + FSDP wrap "
                         "(these collectives happen once, at startup, and a real 7B model on a "
                         "real cluster can take much longer than a mid-training hang should be "
                         "tolerated). Shrunk to --collective-timeout-secs once setup completes.")
    p.add_argument("--tiny", action="store_true",
                    help="DEBUG ONLY — tiny model dims for local smoke-testing, not a real W1 run.")
    return p.parse_args()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def main() -> None:
    args = parse_args()

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # AEGIS must be initialised BEFORE dist.init_process_group() per
    # aegis/__init__.py's documented contract.
    sensor = None
    aegis_loop = None
    if args.aegis_enabled:
        import aegis
        from aegis import _state
        from realbench.sensors.rank_heartbeat import RankHeartbeatSensor

        aegis.init(policy_path=args.aegis_policy_path, mode="active")
        rt = _state.runtime
        aegis_loop = _state._loop

        peer_heartbeat_paths = {
            r: str(log_dir / f"heartbeat_rank{r}.json") for r in range(world_size)
        }
        sensor = RankHeartbeatSensor(
            utp=rt.utp,
            epoch_service=rt.epoch_service,
            own_rank=rank,
            peer_heartbeat_paths=peer_heartbeat_paths,
            detection_log_path=str(log_dir / "detection_log.jsonl"),
            heartbeat_timeout_secs=args.heartbeat_timeout_secs,
        )
        # Sensor's poll loop must run on AEGIS's own background event loop
        # (the training loop below is plain synchronous torch code, no
        # asyncio of its own) — see aegis/__init__.py's _start_event_loop().
        import asyncio
        asyncio.run_coroutine_threadsafe(sensor.start(), aegis_loop).result(timeout=5.0)

        # Each rank has its own independent AEGIS runtime (no cross-process
        # UTP), so each rank registers the full ring topology itself.
        for i in range(world_size):
            rt.compute.register_neighbor(f"rank{i}", f"rank{(i + 1) % world_size}")

    # ------------------------------------------------------------------
    # Real torch.distributed bootstrap. Real cluster: CUDA + NCCL. This
    # script also runs (--tiny, world_size=1) on CPU/MPS for local
    # smoke-testing, where NCCL isn't available — use gloo in that case.
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Required for a hung NCCL collective to raise a catchable exception
        # in this process rather than abort asynchronously — see the
        # docstring of torch.distributed.distributed_c10d._set_pg_timeout.
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    else:
        device = torch.device("cpu")

    dims = TINY_MODEL_DIMS if args.tiny else REAL_MODEL_DIMS

    # Fail fast with an actionable message if the GPU is already dirty (e.g.
    # orphaned processes from a previous crashed launch still pinning memory —
    # see logs.out) rather than OOM-looping at model.to(device). Runs before
    # rendezvous, so pair with `srun --kill-on-bad-exit=1` to bring the other
    # ranks down instead of leaving them hung in init_process_group.
    _preflight_gpu_check(device, dims, rank)

    # Generous timeout for rendezvous + (for a real 7B model) model
    # construction + FSDP wrap — these are one-time collectives that can
    # legitimately take much longer than a mid-training hang should be
    # tolerated. Shrunk to --collective-timeout-secs right after setup.
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.setup_timeout_secs),
    )

    # Everything after rendezvous is wrapped so that ANY failure (notably an
    # OOM in _build_model, which happens before _train_loop's own try/except)
    # still tears the process group and CUDA context down. A crash that skips
    # teardown is exactly what leaves memory-pinning orphans for the next
    # retry to trip over (see logs.out).
    try:
        # Own PID, for chaos_inject.real_injector to target this rank for SIGKILL.
        (log_dir / f"pid_rank{rank}.txt").write_text(str(os.getpid()))

        model = _build_model(dims, args.seq_len, device, world_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        from torch.distributed.distributed_c10d import _set_pg_timeout
        _set_pg_timeout(timedelta(seconds=args.collective_timeout_secs))

        if world_size > 1 and device.type == "cuda":
            parallelism = "DP(FSDP)"
        elif world_size > 1:
            parallelism = "DP(DDP)-local-smoke-test-only"
        else:
            parallelism = "single-process (no wrapping)"

        if rank == 0:
            manifest = {
                "aegis_enabled": args.aegis_enabled,
                "parallelism": parallelism,
                "world_size": world_size,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "seed": args.seed,
                "collective_timeout_secs": args.collective_timeout_secs,
                "heartbeat_timeout_secs": args.heartbeat_timeout_secs,
                "tiny_debug_run": args.tiny,
                "model_dims": dims,
            }
            (log_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

        step_log = StepLogWriter(log_dir / "step_log.jsonl") if rank == 0 else None
        heartbeat = HeartbeatWriter(log_dir / f"heartbeat_rank{rank}.json")

        _train_loop(
            model=model,
            optimizer=optimizer,
            args=args,
            device=device,
            vocab_size=dims["vocab_size"],
            rank=rank,
            log_dir=log_dir,
            step_log=step_log,
            heartbeat=heartbeat,
        )
    finally:
        if args.aegis_enabled and sensor is not None:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(sensor.stop(), aegis_loop).result(timeout=5.0)
            except Exception:
                pass  # never let sensor teardown mask the original failure
        try:
            dist.destroy_process_group()
        except Exception:
            pass  # process group may already be broken by a fault — best-effort teardown
        if torch.cuda.is_available():
            # Release this rank's CUDA context so a crashed launch doesn't
            # leave the GPU pinned for the next retry.
            torch.cuda.empty_cache()


def _estimate_param_bytes(dims: dict, bytes_per_param: int = 4) -> int:
    """Rough footprint of the LLaMA model these dims describe. Used only by the
    preflight GPU check to turn a bare OOM into an actionable message — an
    over- or under-estimate of a few percent doesn't change the verdict when a
    dirty GPU has tens of GiB missing."""
    h = dims["hidden_size"]
    layers = dims["num_hidden_layers"]
    inter = dims["intermediate_size"]
    vocab = dims["vocab_size"]
    per_layer = 4 * h * h + 3 * h * inter  # attn (q,k,v,o) + MLP (gate,up,down)
    embed_and_head = 2 * vocab * h          # input embedding + (untied) lm_head
    return (embed_and_head + layers * per_layer) * bytes_per_param


def _preflight_gpu_check(device: torch.device, dims: dict, rank: int) -> None:
    """Abort early, with a message that names the likely cause, when the target
    GPU isn't clean enough to build the model.

    Guards the exact failure captured in logs.out: orphaned processes from a
    previous crashed launch stayed resident and pinned ~62 GiB per 80 GiB A100,
    so every retry OOM'd identically at ``model.to(device)`` — with a bare
    ``torch.OutOfMemoryError`` that never hinted the card was already full. This
    check makes the real problem (and the fix) legible instead."""
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info(device)
    needed = int(_estimate_param_bytes(dims) * 1.1)  # +10% headroom for the H2D copy
    if free < needed:
        gib = 1024 ** 3
        raise RuntimeError(
            f"[rank{rank}] GPU {device.index} preflight failed: {free / gib:.2f} GiB "
            f"free of {total / gib:.2f} GiB, but building this model needs "
            f"~{needed / gib:.2f} GiB. Something else already holds this GPU — most "
            f"likely orphaned processes from a previous crashed launch (see logs.out). "
            f"Clear the node before retrying rather than OOM-looping: `scancel` the job "
            f"so Slurm's cgroup/epilog cleanup reaps the orphans, then resubmit; or run "
            f"an overlapping cleanup step on the node "
            f"(`srun --jobid=$SLURM_JOB_ID --overlap -w <node> nvidia-smi`)."
        )


def _build_model(dims: dict, seq_len: int, device: torch.device, world_size: int):
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        intermediate_size=dims["intermediate_size"],
        vocab_size=dims["vocab_size"],
        max_position_embeddings=max(seq_len, 512),
    )
    model = LlamaForCausalLM(config)  # random init — no from_pretrained, no dataset download
    model.to(device)

    if world_size > 1 and device.type == "cuda":
        # Real cluster path (test_suite.md W1: DP(FSDP), see module docstring).
        # FSDP shards optimizer/gradient/parameter state — needed to fit a
        # real 7B model's AdamW state on a single A100.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
        )
        model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=device)
    elif world_size > 1:
        # Local multi-process smoke-testing (CPU/gloo — this platform's FSDP
        # build doesn't support the "mps"/CPU accelerator path). Plain DDP
        # still performs a real allreduce collective every backward() call,
        # which is what's needed to genuinely validate that a killed peer
        # hangs a survivor's next collective — the actual thing under test.
        # NEVER used for a real W1 run (those are always CUDA); see
        # run_manifest.json's "parallelism" field for what actually ran.
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model)
    # world_size == 1: skip wrapping entirely — nothing to shard/sync.

    return model


def _train_loop(*, model, optimizer, args, device, vocab_size, rank, log_dir, step_log, heartbeat) -> None:
    for step in range(args.steps):
        try:
            gen = torch.Generator()
            gen.manual_seed(args.seed + step)  # same batch across control/aegis-on runs (§6.1)
            input_ids = torch.randint(
                0, vocab_size, (args.batch_size, args.seq_len), generator=gen
            ).to(device)

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            heartbeat.touch(step)
            if step_log is not None:
                step_log.write(step, loss.item())

        except Exception as exc:
            # A real peer death (SIGKILL in Phase 1) leaves the process group
            # broken; the next collective (FSDP all-gather in fwd/bwd) times
            # out after --collective-timeout-secs rather than hanging
            # forever. This build's AEGIS recovery does not reconfigure the
            # process group (see module docstring) so the job genuinely
            # cannot continue — halt cleanly and record why, rather than
            # hang until SLURM's wall-clock limit kills the allocation.
            append_jsonl(
                log_dir / "training_halt.jsonl",
                {
                    "wall_clock": __import__("time").time(),
                    "rank": rank,
                    "last_completed_step": step - 1,
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if args.aegis_enabled:
                _dump_epe_history(log_dir / "epe_history.json")
            break


def _dump_epe_history(path: Path) -> None:
    """Best-effort dump of AEGIS's own fault-handling record before exit."""
    try:
        from aegis import _state

        rt = _state.runtime
        if rt is None:
            return
        records = []
        for rec in rt.epe.history:
            records.append({
                "epoch": rec.epoch,
                "fault_signal": rec.event.fault_signal.value,
                "node": rec.event.node,
                "original_tier": rec.original_tier.name,
                "final_tier": rec.final_tier.name,
                "escalated": rec.escalated,
                "success": rec.result.success if rec.result else False,
                "degraded": rec.result.degraded if rec.result else False,
                "recovery_secs": rec.result.recovery_secs if rec.result else None,
                "message": rec.result.message if rec.result else "",
            })
        path.write_text(json.dumps(records, indent=2))
    except Exception:
        pass  # never let the dump itself mask the original halt


if __name__ == "__main__":
    main()
