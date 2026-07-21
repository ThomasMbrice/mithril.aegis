#!/bin/bash

#SBATCH --job-name=aegis-phase0-smoke
#SBATCH --partition=a100-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/phase0_smoke_%j.out
#SBATCH --error=logs/phase0_smoke_%j.err
#
# SMOKE-TEST ONLY — not the real Phase 0 trust-anchor run, and its
# phase0_report.md CANNOT be used to satisfy the gate that unlocks
# sbatch_phase1_b1.sh (that requires the real 8xA100/6000-step run per
# test_suite.md §7/§8). This script exists purely to validate the
# plumbing (module load, pip install, aegis PATH, torchrun rendezvous,
# FSDP/DDP wrap, step/heartbeat logging) on a single small, short
# allocation that clears the queue fast — see sbatch_phase0.sh for the
# real run and its resource footprint.
#
# 1 GPU / 8 cores / 32G / 20 min vs. the real job's 4 GPUs / 64 cores /
# 256G / 4h — smaller footprint queues faster on a shared a100-large
# partition. --tiny (run_phase0.py -> w1_llama7b.py) swaps in a toy
# LlamaConfig (hidden_size=64, 2 layers) so even CPU-speed steps finish
# in seconds; real W1 dims are never used here (run_manifest.json's
# "tiny_debug_run" records this so a report can't be confused for real).
#
# --nproc-per-node 1 matches --gres=gpu:1: world_size=1 skips FSDP/DDP
# wrapping entirely (w1_llama7b.py's `world_size == 1: skip wrapping`
# branch) — still real torch.distributed init + real aegis.init() hooks,
# just no multi-rank collective to exercise.
#
# --warmup-steps must stay well under --steps (report_phase0.py discards
# the first N steps before measuring throughput) — 5/50 leaves 45 usable
# post-warmup steps, unlike the real run's 50/6000.
#
# Usage (from the repo root): sbatch realbench/slurm/sbatch_phase0_smoke.sh [OUT_DIR]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export TORCHDYNAMO_DISABLE=1
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs

module load python

pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet -e ".[dev]"

OUT_DIR="${1:-$SLURM_SUBMIT_DIR/realbench_runs/phase0_smoke_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

echo "[sbatch_phase0_smoke] out=$OUT_DIR node=$(hostname) job=$SLURM_JOB_ID"

srun --cpu-bind=cores python -m realbench.phase0_trust_anchor.run_phase0 \
    --out "$OUT_DIR" \
    --steps 50 \
    --repeat 1 \
    --nproc-per-node 1 \
    --tiny

set +e
srun --cpu-bind=cores python -m realbench.phase0_trust_anchor.report_phase0 \
    --out "$OUT_DIR" \
    --repeat 1 \
    --warmup-steps 5
REPORT_EXIT=$?
set -e

echo "[sbatch_phase0_smoke] report exit=$REPORT_EXIT — see $OUT_DIR/phase0_report.md"
echo "[sbatch_phase0_smoke] REMINDER: this is a smoke test — it does NOT satisfy the real Phase 0 gate for sbatch_phase1_b1.sh"
exit $REPORT_EXIT
