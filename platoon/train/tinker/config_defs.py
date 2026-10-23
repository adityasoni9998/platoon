from dataclasses import dataclass, field
from typing import Literal

from platoon.config_defs import RolloutConfig
from platoon.train.components import EnvironmentConfig, normalize_environment_configs
from platoon.utils.stats_logger import StatsLoggerConfig, WandBConfig


@dataclass
class AdamParams:
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    # Set to a large value (e.g., 1e12) to enable grad_norm logging without clipping.
    # When > 0, tinker returns grad_norm in OptimStepResponse.metrics.
    grad_clip_norm: float = 0.0


@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""

    group_size: int = 8
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    leave_one_out_baseline: bool = False  # Use leave-one-out baseline for advantage centering
    depth_level_weighting: bool = False  # Weight trajectories inversely by depth-level frequency
    filter_zero_variance_groups: bool = True  # Preserve old behavior by rejecting zero-variance groups
    algorithm: Literal["grpo", "ppo"] = "grpo"  # Set from TrainConfig for training workflows.
    ppo_reward_mode: Literal["terminal", "turn"] = "terminal"  # Set from TrainConfig.ppo.


@dataclass
class CriticConfig:
    value_clip: float = 0.2
    # Split prepared datums into up to K nonempty minibatches per critic epoch.
    num_minibatches: int = 1
    num_microbatches: int = 1
    optimizer: AdamParams = field(default_factory=lambda: AdamParams(learning_rate=1e-5, grad_clip_norm=1.0))


@dataclass
class PPOConfig:
    """Actor-critic training; gamma/lambda count generated tokens or agent turns."""

    value_granularity: Literal["token", "turn"] = "token"
    reward_mode: Literal["terminal", "turn"] = "terminal"  # Outcome reward, or increments from step["reward"].
    gamma: float = 1.0
    gae_lambda: float = 1.0
    advantage_normalization: Literal["batch", "none"] = "batch"
    advantage_normalization_granularity: Literal["token", "turn"] | None = None  # None follows value_granularity.
    actor_epochs: int = 1
    critic_epochs: int = 1
    critic_warmup_batches: int = 1
    critic: CriticConfig = field(default_factory=CriticConfig)


# TODO:
# - timeout
@dataclass
class TrainConfig:
    model_name: str  # HuggingFace model identifier
    renderer_name: str  # Renderer type for prompt formatting
    renderer_kwargs: dict = field(default_factory=dict)  # Optional renderer attribute overrides
    context_window_length: int | None = None
    batch_size: int = 32
    # Training duration: specify num_epochs, max_training_steps, or both (takes max)
    num_epochs: int | None = None
    max_training_steps: int | None = None
    # Number of minibatches per batch. Batch must be divisible by num_minibatches.
    # Results in num_minibatches weight updates per batch.
    num_minibatches: int = 1
    # Number of microbatches per minibatch. Minibatch must be divisible by num_microbatches.
    # Used for gradient accumulation --> A single weight update per microbatch.
    # With tinker, gradient accumulation is less important, but this may still be useful for streaming
    # to overlap rollout sampling with forward backward computation within the same batch.
    num_microbatches: int = 1
    max_staleness: int | None = None  # Max staleness for off-policy rollouts
    optimizer: AdamParams = field(default_factory=AdamParams)
    loss_fn: str = "cispo"
    loss_fn_config: dict | None = None
    lora_rank: int = 32
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    num_concurrent_rollout_workflow_workers: int | None = None
    algorithm: Literal["grpo", "ppo"] = "grpo"
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        if self.loss_fn_config is None:
            self.loss_fn_config = (
                {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
                if self.algorithm == "ppo" and self.loss_fn == "ppo"
                else {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
            )
        if self.num_concurrent_rollout_workflow_workers is None:
            self.num_concurrent_rollout_workflow_workers = self.batch_size
        if self.algorithm not in ("grpo", "ppo"):
            raise ValueError("train.algorithm must be grpo or ppo")
        self.workflow_config.algorithm = self.algorithm
        self.workflow_config.ppo_reward_mode = self.ppo.reward_mode
        if self.algorithm == "ppo":
            self._validate_ppo()

    def _validate_ppo(self):
        if self.lora_rank != 0 or self.loss_fn not in ("ppo", "cispo"):
            raise ValueError("Actor-critic training requires lora_rank=0 and loss_fn=ppo or cispo")
        if not (0 <= self.ppo.gamma <= 1 and 0 <= self.ppo.gae_lambda <= 1):
            raise ValueError("ppo.gamma and ppo.gae_lambda must be between 0 and 1")
        if self.ppo.value_granularity not in ("token", "turn") or self.ppo.reward_mode not in ("terminal", "turn"):
            raise ValueError("PPO requires value_granularity=token or turn and reward_mode=terminal or turn")
        if not (self.num_minibatches == self.num_microbatches == self.ppo.critic.num_microbatches == 1):
            raise ValueError("PPO requires one actor minibatch and one microbatch for both actor and critic")
        if self.ppo.critic.num_minibatches < 1:
            raise ValueError("ppo.critic.num_minibatches must be positive")
        if self.ppo.advantage_normalization not in ("batch", "none"):
            raise ValueError("ppo.advantage_normalization must be batch or none")
        if self.ppo.advantage_normalization_granularity not in (None, "token", "turn"):
            raise ValueError("ppo.advantage_normalization_granularity must be token, turn, or None")
        if self.ppo.value_granularity == "token" and self.ppo.advantage_normalization_granularity == "turn":
            raise ValueError("Turn-level advantage normalization requires value_granularity=turn")
        if self.workflow_config.leave_one_out_baseline or self.workflow_config.depth_level_weighting:
            raise ValueError("PPO does not support GRPO leave-one-out baselines or depth weighting")


@dataclass
class TrainEventTriggerConfig:
    strategy: Literal["epoch", "step", "none"] = "epoch"
    every: int = 1


@dataclass
class CheckpointConfig(TrainEventTriggerConfig):
    load_checkpoint_path: str | None = None  # For PPO: path to the paired ppo_checkpoints.jsonl history.


@dataclass
class EvalConfig(TrainEventTriggerConfig):
    num_concurrent_rollout_workflow_workers: int = 256
    workflow_config: WorkflowConfig = field(default_factory=lambda: WorkflowConfig(group_size=1))


@dataclass
class StatsConfig:
    """Configuration for stats tracking and logging."""

    experiment_name: str = "platoon_tinker"
    trial_name: str = "run"
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def to_stats_logger_config(self, log_dir: str) -> StatsLoggerConfig:
        """Create a StatsLoggerConfig from this stats config."""
        return StatsLoggerConfig(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            log_dir=log_dir,
            wandb=self.wandb,
        )


@dataclass
class WatchdogConfig:
    """Configuration for the watchdog that monitors for hangs."""

    enabled: bool = True
    timeout_seconds: float = 600  # 10 minutes default
    exit_code: int = 2  # Exit code when watchdog kills process


@dataclass
class PlatoonTinkerRLTrainerConfig:
    train: TrainConfig
    eval: EvalConfig
    log_path: str
    tinker_base_url: str | None = None  # Tinker service URL
    environments: list[EnvironmentConfig] = field(default_factory=lambda: [EnvironmentConfig()])
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)

    def __post_init__(self):
        self.environments = normalize_environment_configs(self.environments)
        if len(self.environments) > 1:
            raise NotImplementedError("Multiple environments are not yet supported; provide exactly one entry")
