#!/bin/bash

#SBATCH --job-name=aegis-phase1-b1
#SBATCH --partition=a100-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=logs/phase1_b1_%j.out
#SBATCH --error=logs/phase1_b1_%j.err
#
# Sized to the a100-large queue's per-job cap — see sbatch_phase0.sh's
# header comment (64 cores / 256G mem / 4 GPUs / 1 node, max runtime 8h,
# max 1 concurrent job for this account; --output/--error need logs/ to
# already exist — submit from the repo root, which checks in logs/.gitkeep).
#
# Phase 1 — real B1 fault injection (test_suite.md §4.5). Requires a
# PASSING Phase 0 report — this script refuses to run otherwise (enforced
# inside realbench.phase1_per_tier.report_phase1, not just here).
#
# This single SLURM task runs realbench.phase1_per_tier.report_phase1,
# which itself spawns the 4 rank processes, the real fault injector, and
# the GPU sampler as independent local subprocesses on this one node — see
# that module's docstring for why this is NOT `aegis run`/torchrun here
# (torchrun's elastic agent would tear down survivors the instant the
# targeted rank is killed, defeating the point of this test).
#
# Usage (from the repo root): sbatch realbench/slurm/sbatch_phase1_b1.sh PHASE0_OUT_DIR [OUT_DIR]
# Override the A100 cost rate: sbatch --export=A100_SPOT_USD_PER_HR=1.50 ...

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

if [ $# -lt 1 ]; then
    echo "Usage: sbatch sbatch_phase1_b1.sh PHASE0_OUT_DIR [OUT_DIR]" >&2
    exit 1
fi
PHASE0_DIR="$1"
OUT_DIR="${2:-$SLURM_SUBMIT_DIR/realbench_runs/phase1_b1_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

# test_suite.md §5.1 headlines $2.35/hr (H100 1-yr); this is SeaWulf's real
# A100 allocation cost instead — fill in via --export at submission time
# (see Usage above), not hardcoded, since it's allocation/accounting-specific.
A100_SPOT_USD_PER_HR="${A100_SPOT_USD_PER_HR:-<FILL_IN>}"

echo "[sbatch_phase1_b1] out=$OUT_DIR phase0_dir=$PHASE0_DIR node=$(hostname) job=$SLURM_JOB_ID"

srun --cpu-bind=cores python -m realbench.phase1_per_tier.report_phase1 \
    --out "$OUT_DIR" \
    --phase0-dir "$PHASE0_DIR" \
    --steps 20000 \
    --warmup-secs 120 \
    --nproc-per-node 4 \
    --gpu-hr-cost-usd "$A100_SPOT_USD_PER_HR"

echo "[sbatch_phase1_b1] done — see $OUT_DIR/phase1_report.md"
