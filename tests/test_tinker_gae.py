"""GAE follows rollout decisions, independently of prefix packing and token lengths."""

from types import SimpleNamespace

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.train.tinker.config_defs import PPOConfig, TrainConfig
from platoon.train.tinker.ppo import compute_gae, loss_datums, prepare_batch
from platoon.train.tinker.proxy import TinkerLLMInteraction
from platoon.utils.tinker_data_processing import trajectory_to_data


def convert(prompts, actions, rewards, *, mode="turn", errors=(), filter_errors=False, trajectory_reward=None):
    interactions = {
        str(i): TinkerLLMInteraction(
            tinker.ModelInput.from_ints(prompt), SimpleNamespace(tokens=action, logprobs=[-0.1] * len(action))
        )
        for i, (prompt, action) in enumerate(zip(prompts, actions, strict=True))
    }
    steps = [
        {"misc": {"action_misc": {"completion_id": str(i)}}, "reward": reward, "error": i in errors}
        for i, reward in enumerate(rewards)
    ]
    return trajectory_to_data(
        trajectory={"steps": steps},
        interactions=interactions,
        task_id="same-task",
        trajectory_id="same-id",
        trajectory_advantage=sum(rewards),
        trajectory_reward=sum(rewards) if trajectory_reward is None else trajectory_reward,
        checkpoint_version=0,
        ppo_reward_mode=mode,
        filter_errors=filter_errors,
    ).datums


def predictions(datums, per_turn_values):
    """Deliberately poison observation values: GAE must never use them."""
    outputs = []
    for datum in datums:
        fields = datum.loss_fn_inputs
        mask = fields["mask"].to_torch().bool()
        first, last = fields["_ppo_turn_range"].to_torch().tolist()
        values = torch.full(mask.shape, 9999.0)
        values[mask] = torch.tensor(
            [v for turn in per_turn_values[first : last + 1] for v in turn], dtype=torch.float32
        )
        outputs.append({"values": TensorData.from_torch(values)})
    return outputs


def action_values(datums, field):
    ordered = sorted(datums, key=lambda d: d.loss_fn_inputs["_ppo_turn_range"].data[0])
    return torch.cat([d.loss_fn_inputs[field].to_torch()[d.loss_fn_inputs["mask"].to_torch().bool()] for d in ordered])


@pytest.mark.parametrize("granularity", ["turn", "token"])
def test_discounted_gae_is_identical_for_merged_and_multiple_shorter_prefix_splits(granularity):
    actions = [[12, 13], [21], [31, 32]]
    merged = convert([[10, 11], [10, 11, 12, 13, 20], [10, 11, 12, 13, 20, 21, 30]], actions, [1, 2, 3])
    split = convert([[10, 11], [40], [50, 51]], actions, [1, 2, 3])
    assert len(merged) == 1 and len(split) == 3
    assert [d.loss_fn_inputs["_ppo_turn_range"].data for d in split] == [[0, 0], [1, 1], [2, 2]]
    per_turn = [[0.2, 0.3], [0.4], [0.5, 0.6]]
    # Arrival order and sequence lengths must not determine GAE order.
    split = [split[2], split[0], split[1]]
    kwargs = dict(gamma=0.9, gae_lambda=0.8, value_granularity=granularity)
    a, stats_a = prepare_batch(merged, predictions(merged, per_turn), "none", **kwargs)
    b, stats_b = prepare_batch(split, predictions(split, per_turn), "none", **kwargs)
    # Hand-computed TD-residual expansion, with a zero terminal bootstrap.
    expected = (
        [3.932, 3.932, 3.85, 2.5, 2.5] if granularity == "turn" else [2.555822464, 3.4525312, 3.32296, 1.768, 2.4]
    )
    torch.testing.assert_close(action_values(a, "advantages"), torch.tensor(expected))
    for field in ("advantages", "returns", "old_values", "critic_mask"):
        torch.testing.assert_close(action_values(a, field), action_values(b, field))
    assert stats_a == pytest.approx(stats_b)
    for role in ("policy", "critic"):
        fields = loss_datums(b, role)
        assert all(not any(k.startswith("_ppo_") for k in d.loss_fn_inputs) for d in fields)
        if role == "critic":
            weights = torch.cat([d.loss_fn_inputs["weights"].to_torch() for d in fields])
            assert weights.sum().item() == pytest.approx(1)
            assert weights.count_nonzero().item() == (3 if granularity == "turn" else 5)


@pytest.mark.parametrize(
    "gamma,lam,targets",
    [
        (1.0, 1.0, [6, 5, 3]),
        (0.9, 1.0, [5.23, 4.7, 3]),
        (0.9, 0.0, [1.36, 2.54, 3]),
        (0.0, 0.8, [1, 2, 3]),
        (0.9, 0.8, [4.14496, 4.268, 3]),
    ],
)
def test_gae_matches_monte_carlo_td_and_intermediate_lambda(gamma, lam, targets):
    values = torch.tensor([0.2, 0.4, 0.6])
    advantages, returns = compute_gae(values, torch.tensor([1.0, 2.0, 3.0]), gamma, lam)
    torch.testing.assert_close(returns, torch.tensor(targets, dtype=torch.float32))
    torch.testing.assert_close(advantages, returns - values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("length", [0, 1, 4096])
def test_gae_terminal_returns_match_discount_powers_and_preserve_dtype(length, dtype):
    values = torch.linspace(-1, 1, length, dtype=dtype)
    rewards = torch.zeros_like(values)
    if length:
        rewards[-1] = 1.0
    gamma = 0.999
    advantages, returns = compute_gae(values, rewards, gamma, 1.0)
    # With lambda=1 and one terminal reward, critic values drop out entirely.
    steps_to_terminal = torch.arange(length - 1, -1, -1, dtype=torch.float64)
    expected = (gamma**steps_to_terminal).to(dtype)
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(advantages, expected - values)


def test_terminal_reward_mode_does_not_sum_step_rewards_and_keeps_rollouts_separate():
    # Same external task/trajectory labels in separate samples must not collide.
    a = convert([[10], [20]], [[11], [21]], [1, 2], mode="terminal", trajectory_reward=9)
    b = convert([[10]], [[11]], [8], mode="terminal")
    assert a[0].loss_fn_inputs["_ppo_rewards"].data == [0]
    assert a[1].loss_fn_inputs["_ppo_rewards"].data == [9]
    assert a[0].loss_fn_inputs["_ppo_trajectory_id"].data != b[0].loss_fn_inputs["_ppo_trajectory_id"].data
    rows = [a[1], b[0], a[0]]
    outputs = [{"values": TensorData.from_torch(torch.zeros(d.model_input.length))} for d in rows]
    prepared, _ = prepare_batch(rows, outputs, "none")
    assert [d.loss_fn_inputs["returns"].data[-1] for d in prepared] == [9, 8, 9]


def test_adjacent_actions_are_split_to_preserve_mask_boundaries():
    rows = convert([[10], [10, 11]], [[11], [12]], [1, 2])
    assert len(rows) == 2
    assert [d.loss_fn_inputs["mask"].data for d in rows] == [[1], [0, 1]]
    prepared, _ = prepare_batch(rows, predictions(rows, [[0.2], [0.3]]), "none", value_granularity="turn")
    torch.testing.assert_close(action_values(prepared, "returns"), torch.tensor([3.0, 2.0]))


@pytest.mark.parametrize("granularity", ["turn", "token"])
@pytest.mark.parametrize("merged", [False, True])
def test_error_filter_keeps_transition_and_reward_but_excludes_training(granularity, merged):
    prompts = [[10], [10, 11, 20], [10, 11, 20, 21, 30]] if merged else [[10], [20], [30]]
    args = (prompts, [[11], [21], [31]], [1, 2, 3])
    rows = convert(*args, errors=(1,), filter_errors=True)
    baseline = convert(*args)
    kwargs = dict(gamma=0.9, gae_lambda=0.8, value_granularity=granularity)
    values = [[0.2], [0.4], [0.6]]
    expected, _ = prepare_batch(baseline, predictions(baseline, values), "none", **kwargs)
    actual, _ = prepare_batch(rows, predictions(rows, values), "none", **kwargs)
    assert len(actual) == (1 if merged else 2)
    torch.testing.assert_close(action_values(actual, "advantages"), action_values(expected, "advantages")[[0, 2]])


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize(
    "granularity,expected,mean,std",
    [
        (None, [1, -1, -1, -1, -1], 1.5, 0.5),
        ("turn", [1, -1, -1, -1, -1], 1.5, 0.5),
        ("token", [2, -0.5, -0.5, -0.5, -0.5], 1.2, 0.4),
    ],
)
def test_turn_critic_whitening_granularity_preserves_critic_targets(merged, granularity, expected, mean, std):
    prompts = [[10], [10, 11, 20]] if merged else [[10], [20]]
    rows = convert(prompts, [[11], [21, 22, 23, 24]], [0, 2])
    outputs = predictions(rows, [[0], [1, 77, 88, 99]])
    raw, raw_stats = prepare_batch(rows, outputs, "none", value_granularity="turn")
    prepared, stats = prepare_batch(
        rows, outputs, "batch", value_granularity="turn", advantage_normalization_granularity=granularity
    )
    torch.testing.assert_close(action_values(prepared, "advantages"), torch.tensor(expected, dtype=torch.float32))
    assert stats["critic/raw_advantages_mean"] == pytest.approx(mean)
    assert stats["critic/raw_advantages_std"] == pytest.approx(std)
    assert stats["critic/normalized_advantages_mean"] == pytest.approx(0, abs=1e-6)
    assert stats["critic/normalized_advantages_std"] == pytest.approx(1)
    # The critic still has two equally weighted turn targets, regardless of whitening or packing.
    for actual, baseline in zip(loss_datums(prepared, "critic"), loss_datums(raw, "critic"), strict=True):
        for key in ("values", "returns", "weights"):
            torch.testing.assert_close(actual.loss_fn_inputs[key].to_torch(), baseline.loss_fn_inputs[key].to_torch())
    weights = torch.cat([d.loss_fn_inputs["weights"].to_torch() for d in loss_datums(prepared, "critic")])
    torch.testing.assert_close(weights[weights > 0], torch.tensor([0.5, 0.5]))
    for key in ("critic/old_values_mean", "critic/old_values_std", "critic/returns_mean", "critic/returns_std"):
        assert stats[key] == raw_stats[key]
    # Actor loss remains a mean over five tokens.
    torch.testing.assert_close(
        torch.cat([d.loss_fn_inputs["advantages"].to_torch() for d in loss_datums(prepared, "policy")]),
        torch.cat([d.loss_fn_inputs["advantages"].to_torch() for d in prepared]) / 5,
    )


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("normalization", ["batch", "none"])
def test_token_whitening_excludes_observations_and_filtered_turns(merged, normalization):
    prompts = [[10, 11], [10, 11, 12, 20], [10, 11, 12, 20, 21, 22, 30]] if merged else [[10, 11], [20], [30]]
    rows = convert(prompts, [[12], [21, 22], [31, 32, 33, 34]], [0, 0, 2], errors=(1,), filter_errors=True)
    prepared, stats = prepare_batch(
        rows,
        predictions(rows, [[0], [1000, 2000], [1, 77, 88, 99]]),
        normalization,
        value_granularity="turn",
        advantage_normalization_granularity="token",
    )
    expected = [2, -0.5, -0.5, -0.5, -0.5] if normalization == "batch" else [2, 1, 1, 1, 1]
    torch.testing.assert_close(action_values(prepared, "advantages"), torch.tensor(expected, dtype=torch.float32))
    assert stats["critic/raw_advantages_mean"] == pytest.approx(1.2)
    for datum in prepared:
        inputs = datum.loss_fn_inputs
        assert (inputs["advantages"].to_torch()[~inputs["mask"].to_torch().bool()] == 0).all()


@pytest.mark.parametrize("value_granularity,normalization_granularity", [("token", "turn"), ("turn", "invalid")])
def test_rejects_unsupported_advantage_normalization_granularity(value_granularity, normalization_granularity):
    kwargs = dict(value_granularity=value_granularity, advantage_normalization_granularity=normalization_granularity)
    with pytest.raises(ValueError, match="advantage_normalization_granularity|requires value_granularity=turn"):
        TrainConfig(model_name="test", renderer_name="test", algorithm="ppo", lora_rank=0, ppo=PPOConfig(**kwargs))
    with pytest.raises(ValueError, match="advantage_normalization_granularity|requires value_granularity=turn"):
        prepare_batch([], [], "batch", **kwargs)


@pytest.mark.parametrize(
    "value_granularity,normalization_granularity",
    [("token", None), ("token", "token"), ("turn", None), ("turn", "turn"), ("turn", "token")],
)
def test_advantage_normalization_granularity_config_loads(value_granularity, normalization_granularity):
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import load_config

    args = [
        "--train.model_name",
        "test",
        "--train.renderer_name",
        "test",
        "--train.algorithm",
        "ppo",
        "--train.lora_rank",
        "0",
        "--train.ppo.value_granularity",
        value_granularity,
        "--eval.strategy",
        "none",
        "--log_path",
        "unused",
    ]
    if normalization_granularity is not None:
        args.extend(["--train.ppo.advantage_normalization_granularity", normalization_granularity])
    cfg, _ = load_config(args=args, config_class=PlatoonTinkerRLTrainerConfig)
    assert cfg.train.ppo.value_granularity == value_granularity
    assert cfg.train.ppo.advantage_normalization_granularity == normalization_granularity


def test_missing_turn_is_not_treated_as_a_terminal_boundary():
    rows = convert([[10], [20], [30]], [[11], [21], [31]], [1, 2, 3])
    rows.pop(1)
    with pytest.raises(ValueError, match="missing or overlapping"):
        prepare_batch(rows, predictions(rows, [[0], [0], [0]]), "none")


def test_config_propagates_turn_rewards_and_accepts_both_granularities():
    for granularity in ("token", "turn"):
        cfg = TrainConfig(
            model_name="test",
            renderer_name="test",
            algorithm="ppo",
            lora_rank=0,
            ppo=PPOConfig(value_granularity=granularity, reward_mode="turn", gamma=0.99, gae_lambda=0.95),
        )
        assert cfg.workflow_config.ppo_reward_mode == "turn"


@pytest.mark.parametrize("option,value", [("gamma", -0.1), ("gae_lambda", 1.1)])
def test_config_rejects_invalid_discount(option, value):
    with pytest.raises(ValueError, match="between 0 and 1"):
        TrainConfig(
            model_name="test", renderer_name="test", algorithm="ppo", lora_rank=0, ppo=PPOConfig(**{option: value})
        )
