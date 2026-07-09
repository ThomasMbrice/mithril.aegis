# ST-3 — $/GPU-hour-saved report

Generated: 2026-07-09T00:41:48.906115+00:00
Trace: FT-PRODUCTION (seed=2024)
GPU cost: $2.35/GPU-hr

**Caveat (see design.md §8.1):** AEGIS's recovery-time numbers here come from the real EPE routing decision (via `AegisAdapter` driving the actual `AegisRuntime`) combined with `bench/sim/cost_model.py`'s *target* per-tier recovery times — not hardware-measured durations. The other five systems (B-VANILLA, B-R2CCL, B-MECEFO, B-TIERCHECK, B-TORCHFT) are entirely simulated cost-table baselines, not real integrations. This is the pre-hardware-validation, small-scale number described in test_suite.md §8 Phase 0-2; real A100/IB cluster validation (test_suite.md §4.5, eval_design.md) is the next gate before this number can be trusted beyond "the EPE routes correctly and the composed-tier story is economically plausible."

## Headline: AEGIS vs B-VANILLA on FT-PRODUCTION

- **W1** (8 GPUs): AEGIS saved **$19.35** vs B-VANILLA (8.23 GPU-hrs, 89.1% faster recovery) over 14 faults.
- **W2** (32 GPUs): AEGIS saved **$116.40** vs B-VANILLA (49.53 GPU-hrs, 92.5% faster recovery) over 14 faults.
- **W3** (64 GPUs): AEGIS saved **$478.25** vs B-VANILLA (203.51 GPU-hrs, 96.2% faster recovery) over 14 faults.


## Workload W1 — LLaMA-7B (8 GPUs, DP+PP)

```

  AEGIS Benchmark Comparison Matrix
  =======================================================================================
  System      | $/GPU-hr wasted |  Goodput | Recovery (s) |  vs B-VANILLA savings ($)
  ------------+-----------------+----------+--------------+--------------------------
  AEGIS       |         $2.3761 |    0.994 |        455.0 |                  $19.3504
  B-R2CCL     |        $10.3619 |    0.975 |       1984.2 |                  $11.3646
  B-TORCHFT   |        $13.0033 |    0.969 |       2490.0 |                   $8.7232
  B-MECEFO    |        $13.3919 |    0.968 |       2564.4 |                   $8.3347
  B-VANILLA   |        $21.7265 |    0.948 |       4160.4 |                (baseline)
  B-TIERCHECK |        $22.3887 |    0.946 |       4287.2 |                  -$0.6622

```

## Workload W2 — LLaMA-13B (32 GPUs, DP+PP+TP)

```

  AEGIS Benchmark Comparison Matrix
  =======================================================================================
  System      | $/GPU-hr wasted |  Goodput | Recovery (s) |  vs B-VANILLA savings ($)
  ------------+-----------------+----------+--------------+--------------------------
  AEGIS       |         $9.5044 |    0.997 |        455.0 |                 $116.4033
  B-TORCHFT   |        $52.0133 |    0.985 |       2490.0 |                  $73.8944
  B-R2CCL     |        $57.9841 |    0.983 |       2775.8 |                  $67.9237
  B-MECEFO    |        $79.7433 |    0.977 |       3817.5 |                  $46.1644
  B-TIERCHECK |       $124.8459 |    0.964 |       5976.7 |                   $1.0619
  B-VANILLA   |       $125.9078 |    0.964 |       6027.5 |                (baseline)

```

## Workload W3 — LLaMA-40B (64 GPUs, FSDP/HSDP)

```

  AEGIS Benchmark Comparison Matrix
  =======================================================================================
  System      | $/GPU-hr wasted |  Goodput | Recovery (s) |  vs B-VANILLA savings ($)
  ------------+-----------------+----------+--------------+--------------------------
  AEGIS       |        $19.0089 |    0.999 |        455.0 |                 $478.2511
  B-TORCHFT   |       $104.0267 |    0.995 |       2490.0 |                 $393.2333
  B-R2CCL     |       $236.7756 |    0.989 |       5667.5 |                 $260.4844
  B-MECEFO    |       $308.4244 |    0.985 |       7382.5 |                 $188.8356
  B-TIERCHECK |       $470.8356 |    0.977 |      11270.0 |                  $26.4244
  B-VANILLA   |       $497.2600 |    0.976 |      11902.5 |                (baseline)

```
