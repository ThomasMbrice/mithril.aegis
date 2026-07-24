"""
WorkloadConfig dataclass + W1, W2, W3 instances.

Each workload represents a real LLM training job with parameters
matching the papers' own testbeds for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkloadConfig:
    """
    Configuration for a training workload.

    Used by the benchmark to compute recovery costs in the context of
    a real training job (checkpoint intervals, restore times, etc.).
    """

    name: str
    model_name: str
    params_billions: float
    gpu_count: int
    steps_per_second: float
    checkpoint_interval_steps: int
    checkpoint_write_secs: float
    checkpoint_restore_secs: float
    parallelism: str = ""

    # Computed field: duration of a single training step in seconds
    step_duration_secs: float = field(init=False)

    def __post_init__(self) -> None:
        # frozen=True means we must use object.__setattr__ to set computed fields
        object.__setattr__(
            self, "step_duration_secs", 1.0 / self.steps_per_second
        )

    @property
    def checkpoint_interval_secs(self) -> float:
        """Wall-clock time between checkpoints."""
        return self.checkpoint_interval_steps * self.step_duration_secs

    def vanilla_rollback_secs(
        self, fault_step: int, last_checkpoint_step: int
    ) -> float:
        """
        Compute recovery time for checkpoint-and-restart baseline.

        This is the time to re-run lost steps plus checkpoint restore time:
          lost_steps × step_duration + checkpoint_restore_secs
        """
        lost_steps = max(0, fault_step - last_checkpoint_step)
        return lost_steps * self.step_duration_secs + self.checkpoint_restore_secs


# ---------------------------------------------------------------------------
# Standard workload instances

# W1: LLaMA-7B on 8 GPUs — matches MeCeFO's exact testbed
W1 = WorkloadConfig(
    name="W1",
    model_name="LLaMA-7B",
    params_billions=7.0,
    gpu_count=8,
    steps_per_second=2.5,
    checkpoint_interval_steps=1000,
    checkpoint_write_secs=30.0,
    checkpoint_restore_secs=120.0,
    parallelism="DP+PP",
)

# W2: ~13B on 32 GPUs — mid-scale, exercises all three parallelism dims
W2 = WorkloadConfig(
    name="W2",
    model_name="LLaMA-13B",
    params_billions=13.0,
    gpu_count=32,
    steps_per_second=1.2,
    checkpoint_interval_steps=500,
    checkpoint_write_secs=60.0,
    checkpoint_restore_secs=240.0,
    parallelism="DP+PP+TP",
)

# W3: ~40B on 64 GPUs — checkpoint-heavy, stresses storage tiers B2-B4
W3 = WorkloadConfig(
    name="W3",
    model_name="LLaMA-40B",
    params_billions=40.0,
    gpu_count=64,
    steps_per_second=0.4,
    checkpoint_interval_steps=200,
    checkpoint_write_secs=120.0,
    checkpoint_restore_secs=600.0,
    parallelism="FSDP/HSDP",
)
