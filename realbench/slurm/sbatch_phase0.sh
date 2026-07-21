#!/bin/bash

#SBATCH --job-name=aegis-phase0-trust-anchor
#SBATCH --partition=a100-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --output=logs/phase0_%j.out
#SBATCH --error=logs/phase0_%j.err
#
# Sized to the a100-large queue's per-job cap: 64 cores / 256G mem / 4 GPUs /
# 1 node, max runtime 8h, max 1 job at a time for this account — do not raise
# --gres/--cpus-per-task/--mem/--nodes above those, and do not submit this
# alongside another a100-large job (Max Jobs=1 will reject/queue it).
#
# --output/--error are relative to wherever `sbatch` is invoked from
# (SLURM opens them at job START, before this script's own `mkdir -p logs`
# runs — that line is just a defensive fallback, not what makes this work).
# The repo already checks in logs/.gitkeep so this resolves as long as you
# submit from the repo root, per Usage below.
#
# Phase 0 — trust-anchor (test_suite.md §7/§8). W1 (LLaMA-7B), single node,
# 4xA100, NO fault injection: control vs AEGIS-on steady-state overhead.
# MUST pass before submitting sbatch_phase1_b1.sh.
#
# Usage (from the repo root): sbatch realbench/slurm/sbatch_phase0.sh [OUT_DIR]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export TORCHDYNAMO_DISABLE=1
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs

module load python

# CUDA-enabled torch first (pyproject.toml's plain "torch>=2.2" pin can
# otherwise resolve to a CPU-only wheel) — bump cu121 if SeaWulf's driver
# needs a different CUDA build.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
pip install --quiet -e ".[dev]"

OUT_DIR="${1:-$SLURM_SUBMIT_DIR/realbench_runs/phase0_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

echo "[sbatch_phase0] out=$OUT_DIR node=$(hostname) job=$SLURM_JOB_ID"

srun --cpu-bind=cores python -m realbench.phase0_trust_anchor.run_phase0 \
    --out "$OUT_DIR" \
    --steps 6000 \
    --repeat 3 \
    --nproc-per-node 4

set +e
srun --cpu-bind=cores python -m realbench.phase0_trust_anchor.report_phase0 \
    --out "$OUT_DIR" \
    --repeat 3 \
    --warmup-steps 50
REPORT_EXIT=$?
set -e

echo "[sbatch_phase0] report exit=$REPORT_EXIT — see $OUT_DIR/phase0_report.md"
if [ "$REPORT_EXIT" -ne 0 ]; then
    echo "[sbatch_phase0] GATE FAILED — do not submit sbatch_phase1_b1.sh until this is fixed" >&2
fi
exit $REPORT_EXIT
