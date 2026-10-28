"""PPO contracts, frozen targets, role ordering, and paired checkpoint history."""

import asyncio
import json
from copy import deepcopy
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tinker
import torch
from datasets import Dataset
from tinker import TensorData

from platoon.train.tinker.config_defs import (
    CriticConfig,
    EvalConfig,
    PlatoonTinkerRLTrainerConfig,
    PPOConfig,
    TrainConfig,
    WorkflowConfig,
)
from platoon.train.tinker.ppo import (
    MANIFEST_NAME,
    _actor_metric_datums,
    load_resume_manifest,
    loss_datums,
    prepare_batch,
    restore_pair,
    save_pair,
    update_batch,
)
from platoon.train.tinker.workflows.group_rollout_workflow import GroupRolloutWorkflow
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def tensor(values):
    return TensorData.from_torch(torch.tensor(values))


_trajectory_ids = count()


def datum(reward=1.0):
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([10, 11, 12]),
        loss_fn_inputs={
            "target_tokens": tensor([11, 12, 13]),
            "mask": tensor([0.0, 1.0, 1.0]),
            "logprobs": tensor([0.0, -0.2, -0.3]),
            "advantages": tensor([0.0, reward, reward]),
            "returns": tensor([0.0, reward, reward]),
            "_ppo_trajectory_id": tensor([next(_trajectory_ids)]),
            "_ppo_turn_range": tensor([0, 0]),
            "_ppo_rewards": tensor([reward]),
            "_ppo_train_turns": tensor([1]),
        },
    )


def config(**kwargs):
    train = TrainConfig(
        model_name="test",
        renderer_name="test",
        algorithm="ppo",
        lora_rank=0,
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        batch_size=2,
        max_training_steps=2,
        **kwargs,
    )
    return PlatoonTinkerRLTrainerConfig(
        train=train, eval=EvalConfig(strategy="none", every=0), log_path="unused", tinker_base_url="http://test"
    )


def test_terminal_returns_and_first_action_alignment_across_datums():
    original = [datum(1), datum(1), datum(0)]  # first two may be prefix breaks in the same trajectory
    predictions = [{"values": tensor([999.0, 0.2, 0.6])} for _ in original]
    prepared, _ = prepare_batch(original, predictions, "none")
    np.testing.assert_allclose(prepared[0].advantages, [0.0, 0.8, 0.4])
    np.testing.assert_array_equal(prepared[1].returns, [0.0, 1.0, 1.0])
    np.testing.assert_allclose(prepared[2].advantages, [0.0, -0.2, -0.6])
    # The prediction at the final prompt INPUT position is kept: target there is an action.
    assert prepared[0].old_values.tolist() == pytest.approx([0.0, 0.2, 0.6])
    assert original[0].loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize("granularity", [None, "token"])
def test_whitening_is_global_and_does_not_modify_critic_returns(granularity):
    prepared, _ = prepare_batch(
        [datum(1), datum(0)],
        [
            {"values": tensor([9.0, 0.1, 0.4])},
            {"values": tensor([9.0, 0.2, 0.7])},
        ],
        "batch",
        advantage_normalization_granularity=granularity,
    )
    flat = np.concatenate([d.advantages[1:] for d in prepared])
    assert flat.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert flat.std().item() == pytest.approx(1.0, abs=1e-6)
    assert prepared[0].returns.tolist() == [0.0, 1.0, 1.0]
    assert prepared[1].returns.tolist() == [0.0, 0.0, 0.0]
    assert all(d.advantages[0] == 0 for d in prepared)


def test_critic_and_actor_token_normalization_and_frozen_data():
    prepared, _ = prepare_batch([datum()], [{"values": tensor([0.0, 0.2, 0.6])}], "none")
    before = deepcopy(prepared)
    actor = loss_datums(prepared, "policy")[0].loss_fn_inputs
    critic = loss_datums(prepared, "critic")[0].loss_fn_inputs
    torch.testing.assert_close(actor["advantages"].to_torch(), torch.tensor([0.0, 0.4, 0.2]))
    assert critic["weights"].to_torch().tolist() == [0.0, 0.5, 0.5]
    assert critic["returns"].to_torch().tolist() == [0.0, 1.0, 1.0]
    assert "mask" not in actor and "loss_mask" not in actor
    assert "advantages" not in critic
    for key in ("mask", "critic_mask", "old_values", "returns", "advantages"):
        np.testing.assert_array_equal(getattr(prepared[0], key), getattr(before[0], key))
    assert prepared[0].datum == before[0].datum


def test_preparation_keeps_arrays_until_sdk_request_construction(monkeypatch):
    rows = [datum()]
    outputs = [{"values": tensor([0.0, 0.2, 0.6])}]
    before = deepcopy((rows, outputs))
    threads = torch.get_num_threads(), torch.get_num_interop_threads()

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("PPO preparation must not serialize working arrays or convert them to Torch")

    with monkeypatch.context() as patch:
        for name in ("from_numpy", "from_torch", "to_torch"):
            patch.setattr(TensorData, name, unexpected_conversion)
        prepared, _ = prepare_batch(rows, outputs, "batch")

    assert threads == (torch.get_num_threads(), torch.get_num_interop_threads())
    assert (rows, outputs) == before
    assert prepared[0].datum is rows[0]
    for name in ("mask", "critic_mask"):
        assert getattr(prepared[0], name).dtype == np.bool_
    for name in ("advantages", "returns", "old_values"):
        assert getattr(prepared[0], name).dtype == np.float32
    actor = loss_datums(prepared, "policy")[0]
    critic = loss_datums(prepared, "critic")[0]
    assert set(actor.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    assert set(critic.loss_fn_inputs) == {"target_tokens", "values", "returns", "weights"}
    assert actor.loss_fn_inputs["logprobs"] is rows[0].loss_fn_inputs["logprobs"]
    for request in (actor, critic):
        for field, data in request.loss_fn_inputs.items():
            assert data.dtype == ("int64" if field == "target_tokens" else "float32")


def test_actor_metrics_exclude_filtered_turns_with_prepared_mask():
    from platoon.train.tinker.rl import compute_training_metrics

    row = datum()
    row.loss_fn_inputs.update(
        mask=tensor([1.0, 0.0, 1.0]),
        logprobs=tensor([-100.0, -100.0, -0.5]),
        _ppo_turn_range=tensor([0, 1]),
        _ppo_rewards=tensor([0.0, 1.0]),
        _ppo_train_turns=tensor([0, 1]),
    )
    prepared, _ = prepare_batch([row], [{"values": tensor([0.0, 0.0, 0.0])}], "none")
    metrics_input = _actor_metric_datums(prepared)
    assert metrics_input[0].loss_fn_inputs["mask"].data == [0.0, 0.0, 1.0]
    result = tinker.ForwardBackwardOutput(
        loss_fn_output_type="ArrayRecord",
        loss_fn_outputs=[{"logprobs": tensor([100.0, 100.0, -0.4])}],
        metrics={"loss:sum": 0.0},
    )
    metrics = compute_training_metrics(
        metrics_input, result, "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    )
    assert metrics["optim/kl_sample_train_v1"] == pytest.approx(-0.1)
    assert metrics["optim/entropy"] == pytest.approx(0.5)
    assert metrics["optim/importance_weight_mean"] == pytest.approx(np.exp(0.1))
    assert metrics["optim/clip_frac_total"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"algorithm": "ppo", "lora_rank": 0, "loss_fn": "ppo", "num_microbatches": 2},
        {"algorithm": "unknown"},
    ],
)
def test_config_rejects_unsupported_modes(change):
    with pytest.raises(ValueError):
        TrainConfig(model_name="test", renderer_name="test", **change)


def test_grpo_defaults_unchanged():
    train = TrainConfig(model_name="test", renderer_name="test")
    assert train.algorithm == train.workflow_config.algorithm == "grpo"
    assert train.loss_fn == "cispo"
    assert train.workflow_config.filter_zero_variance_groups


@pytest.mark.parametrize("count", [0, -1])
def test_critic_minibatch_count_must_be_positive(count):
    with pytest.raises(ValueError, match="num_minibatches must be positive"):
        config(ppo=PPOConfig(critic=CriticConfig(num_minibatches=count)))


def test_critic_minibatches_load_from_yaml(tmp_path):
    from platoon.utils.config import load_config

    path = tmp_path / "ppo.yaml"
    path.write_text("""train:
  model_name: test
  renderer_name: test
  algorithm: ppo
  lora_rank: 0
  loss_fn: ppo
  ppo:
    critic:
      num_minibatches: 4
eval: {}
log_path: unused
""")
    cfg, _ = load_config(["--config", str(path)], PlatoonTinkerRLTrainerConfig)
    assert cfg.train.ppo.critic.num_minibatches == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm,group_size", [("ppo", 1), ("ppo", 2), ("grpo", 2)])
async def test_ppo_retains_equal_reward_groups_and_grpo_still_filters(algorithm, group_size):
    workflow = GroupRolloutWorkflow.__new__(GroupRolloutWorkflow)
    workflow.config = WorkflowConfig(algorithm=algorithm, group_size=group_size)
    workflow.tracker = StatsTracker("test")

    async def episode(data, index):
        return TrajectoryCollectionResult(
            datums=[datum(1)],
            task_reward=1.0,
            root_rewards_dict={},
            trajectory_stats=[TrajectoryStats(str(index), 1.0, 1, 2, 2, 1, is_root=True)],
        )

    workflow.arun_episode_single = episode
    result = await workflow.arun_episode({"task_id": "task"})
    if algorithm == "grpo":
        assert result is None
    else:
        assert len(result) == group_size
        assert all(d.loss_fn_inputs["_ppo_rewards"].to_torch().tolist() == [1.0] for d in result)


def test_real_prefix_conversion_keeps_observations_masked_and_terminal_targets():
    from platoon.train.tinker.proxy import TinkerLLMInteraction
    from platoon.utils.tinker_data_processing import trajectory_to_data

    def interaction(prompt, tokens):
        return TinkerLLMInteraction(
            tinker.ModelInput.from_ints(prompt), SimpleNamespace(tokens=tokens, logprobs=[-0.1] * len(tokens))
        )

    interactions = {
        "a": interaction([10, 11], [12, 13]),
        "b": interaction([10, 11, 12, 13, 20], [21]),
        "c": interaction([40, 41], [42, 43]),
    }
    result = trajectory_to_data(
        trajectory={"steps": [{"misc": {"action_misc": {"completion_id": key}}} for key in interactions]},
        interactions=interactions,
        task_id="task",
        trajectory_id="trajectory",
        trajectory_advantage=1.0,
        checkpoint_version=0,
        trajectory_reward=1.0,
        ppo_reward_mode="terminal",
    )
    assert len(result.datums) == 2
    assert result.datums[0].loss_fn_inputs["mask"].to_torch().tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert result.datums[1].loss_fn_inputs["mask"].to_torch().tolist() == [0.0, 1.0, 1.0]
    prepared, _ = prepare_batch(
        result.datums,
        [
            {"values": tensor([99.0, 0.2, 0.3, 99.0, 0.4])},
            {"values": tensor([99.0, 0.5, 0.6])},
        ],
        "none",
    )
    np.testing.assert_allclose(prepared[0].advantages, [0.0, 0.8, 0.7, 0.0, 0.6])
    np.testing.assert_allclose(prepared[1].advantages, [0.0, 0.5, 0.4])


def test_degenerate_whitening_is_finite():
    row = datum(1)
    row.loss_fn_inputs["mask"] = tensor([0.0, 0.0, 1.0])
    rows, _ = prepare_batch([row], [{"values": tensor([0.0, 0.0, 0.0])}], "batch")
    assert rows[0].advantages.tolist() == [0.0, 0.0, 0.0]
    assert rows[0].returns.tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("normalization", ["batch", "none"])
def test_zero_mask_datum_does_not_affect_targets_or_normalization(normalization):
    empty = datum(100)
    empty.loss_fn_inputs["mask"] = tensor([0.0, 0.0, 0.0])
    valid = [datum(1), datum(0)]
    outputs = [{"values": tensor([0.0, 0.2, 0.6])}, {"values": tensor([0.0, 0.4, 0.8])}]
    expected, expected_stats = prepare_batch(valid, outputs, normalization)
    actual, stats = prepare_batch(
        [valid[0], empty, valid[1]],
        [outputs[0], {"values": tensor([99.0, 99.0, 99.0])}, outputs[1]],
        normalization,
    )
    assert len(actual) == 2
    assert stats == expected_stats
    for row, expected_row in zip(actual, expected, strict=True):
        for key in ("returns", "old_values", "advantages"):
            np.testing.assert_array_equal(getattr(row, key), getattr(expected_row, key))


def test_yaml_config_loads_ppo_and_legacy_grpo():
    from pathlib import Path

    from platoon.utils.config import load_config

    root = Path(__file__).resolve().parents[1] / "plugins/number-search/platoon/number_search"
    for name, algorithm in (("number_search_ppo_tinker.yaml", "ppo"), ("number_search_tinker.yaml", "grpo")):
        cfg, _ = load_config(args=["--config", str(root / name)], config_class=PlatoonTinkerRLTrainerConfig)
        assert cfg.train.algorithm == cfg.train.workflow_config.algorithm == algorithm


class Future:
    def __init__(self, value, on_result=None):
        self.value = value
        self.on_result = on_result

    async def result_async(self):
        if self.on_result is not None:
            self.on_result()
        return self.value


class ToyClient:
    """CPU model/Adam behind the SDK interface, with shared-residency checks."""

    def __init__(self, role, bus):
        self.role = role
        self.bus = bus
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.optimizer = torch.optim.Adam([self.weight], lr=1e-3)
        self.steps = 0
        self.frozen = []
        self.fail_backward = False

    def predictions(self, datum):
        if self.role == "critic":
            # Value at each input position depends only on the prefix through it.
            return torch.tensor(datum.model_input.to_ints()).float().cumsum(0) * self.weight / 100
        return -self.weight.expand(datum.model_input.length)

    def check_residency(self):
        assert self.bus.get("pending") in (None, self.role)

    async def forward_async(self, data, loss_fn):
        self.check_residency()
        assert self.role == "critic" and loss_fn == "ppo"
        return Future(
            tinker.ForwardBackwardOutput(
                loss_fn_output_type="ArrayRecord",
                loss_fn_outputs=[{"values": tensor(self.predictions(d).detach().tolist())} for d in data],
                metrics={"loss:sum": 0.0},
            )
        )

    async def forward_backward_async(self, data, loss_fn, loss_fn_config):
        self.check_residency()
        if self.fail_backward:
            raise RuntimeError("injected backward failure")
        if "training_started" in self.bus:
            self.bus["training_started"].set()
            await self.bus["overlap"].wait()
        self.bus["pending"] = self.role
        self.bus["events"].append(("backward", self.role, loss_fn))
        self.frozen.append(deepcopy(data))
        losses, outputs = [], []
        for datum in data:
            inputs = {k: v.to_torch() for k, v in datum.loss_fn_inputs.items()}
            prediction = self.predictions(datum)
            if self.role == "critic":
                assert set(inputs) == {"target_tokens", "values", "returns", "weights"}
                clip = loss_fn_config["value_clip"]
                clipped = inputs["values"] + (prediction - inputs["values"]).clamp(-clip, clip)
                loss = (
                    torch.maximum((prediction - inputs["returns"]) ** 2, (clipped - inputs["returns"]) ** 2)
                    * inputs["weights"]
                ).sum()
                key = "values"
            else:
                assert set(inputs) == {"target_tokens", "logprobs", "advantages"}
                ratio = (prediction - inputs["logprobs"]).exp()
                clipped = ratio.clamp(loss_fn_config["clip_low_threshold"], loss_fn_config["clip_high_threshold"])
                if loss_fn == "ppo":
                    loss = -torch.minimum(ratio * inputs["advantages"], clipped * inputs["advantages"]).sum()
                else:
                    assert loss_fn == "cispo"
                    loss = -(clipped.detach() * prediction * inputs["advantages"]).sum()
                key = "logprobs"
            losses.append(loss)
            outputs.append({key: tensor(prediction.detach().tolist())})
        total = torch.stack(losses).sum()
        total.backward()
        return Future(
            tinker.ForwardBackwardOutput(
                loss_fn_output_type="ArrayRecord", loss_fn_outputs=outputs, metrics={"loss:sum": total.item()}
            ),
            on_result=lambda: self.bus["events"].append(("backward_result", self.role)),
        )

    async def optim_step_async(self, params):
        self.check_residency()
        for group in self.optimizer.param_groups:
            group["lr"] = params.learning_rate
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.steps += 1
        self.bus["events"].append(("step", self.role))

        def finish():
            self.bus["pending"] = None
            self.bus["events"].append(("step_result", self.role))

        return Future(SimpleNamespace(metrics={"skipped": 0.0, "grad_norm": 0.1}), on_result=finish)

    async def save_state_async(self, name):
        self.check_residency()
        assert self.bus.get("pending") is None
        path = f"tinker://{self.role}/weights/{name}"
        self.bus["saved"][path] = (self.weight.detach().clone(), deepcopy(self.optimizer.state_dict()), self.steps)
        return Future(SimpleNamespace(path=path))

    async def load_state_with_optimizer_async(self, path):
        self.check_residency()
        weight, optimizer, self.steps = self.bus["saved"][path]
        with torch.no_grad():
            self.weight.copy_(weight)
        self.optimizer.load_state_dict(deepcopy(optimizer))
        return Future(None)

    async def save_weights_and_get_sampling_client_async(self):
        assert self.role == "policy" and self.bus.get("pending") is None
        self.bus["events"].append(("publish", self.role))
        return SimpleNamespace()

    async def save_weights_for_sampler_async(self, name):
        assert self.role == "policy" and self.bus.get("pending") is None
        path = f"tinker://policy/sampler_weights/{name}"
        self.bus["events"].append(("save_sampler", path))
        return Future(SimpleNamespace(path=path))

    def create_sampling_client(self, path):
        assert ("save_sampler", path) in self.bus["events"]
        self.bus["events"].append(("create_sampler", path))
        return SimpleNamespace()


def clients():
    bus = {"events": [], "saved": {}}
    return ToyClient("policy", bus), ToyClient("critic", bus), bus


@pytest.mark.asyncio
async def test_update_batch_uses_token_whitening_with_turn_critic(tmp_path):
    cfg = config(
        ppo=PPOConfig(value_granularity="turn", advantage_normalization_granularity="token", critic_warmup_batches=0)
    )
    trainer = make_trainer(tmp_path, cfg)
    actor, critic, _ = clients()
    shared = SimpleNamespace(
        training_client=actor, critic_client=critic, train_tracker=StatsTracker("test"), train_step=0, watchdog=None
    )
    rows = [datum(1), datum(0)]
    rows[1].loss_fn_inputs["mask"] = tensor([0.0, 0.0, 1.0])
    await update_batch(trainer, shared, rows)
    # Two tokens share the positive turn advantage, and one has the negative advantage.
    expected = torch.tensor([0, 1, 1, 0, 0, -2], dtype=torch.float32) / (3 * 2**0.5)
    actual = torch.cat([d.loss_fn_inputs["advantages"].to_torch() for d in actor.frozen[0]])
    torch.testing.assert_close(actual, expected)
    assert [sum(d.loss_fn_inputs["weights"].data) for d in critic.frozen[0]] == [0.5, 0.5]


@pytest.mark.asyncio
@pytest.mark.parametrize("count,expected_sizes", [(1, [11]), (2, [5, 6]), (4, [2, 2, 2, 5]), (16, [1] * 11)])
@pytest.mark.parametrize("warmup", [False, True])
async def test_critic_minibatches_preserve_targets_and_actor_advantages(
    tmp_path, monkeypatch, count, expected_sizes, warmup
):
    cfg = config(
        ppo=PPOConfig(
            value_granularity="turn",
            critic_epochs=2,
            critic_warmup_batches=int(warmup),
            critic=CriticConfig(num_minibatches=count),
        )
    )
    actor, critic, bus = clients()
    rows = [datum(float(i)) for i in range(11)]
    outputs = [{"values": tensor(critic.predictions(row).detach().tolist())} for row in rows]
    prepared, _ = prepare_batch(rows, outputs, "batch", value_granularity="turn")
    forward_calls = 0
    original_forward = critic.forward_async

    async def forward_once(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return await original_forward(*args, **kwargs)

    monkeypatch.setattr(critic, "forward_async", forward_once)
    shared = SimpleNamespace(
        training_client=actor, critic_client=critic, train_tracker=StatsTracker("test"), train_step=0, watchdog=None
    )
    await update_batch(make_trainer(tmp_path, cfg), shared, rows)
    actual_count = min(count, len(rows))
    assert forward_calls == 1
    assert critic.steps == 2 * actual_count
    assert actor.steps == int(not warmup)
    assert [e[1] for e in bus["events"] if e[0] == "step"] == (
        ["critic"] * (2 * actual_count) + (["policy"] if not warmup else [])
    )
    sizes = [len(batch) for batch in critic.frozen[:actual_count]]
    assert sizes == expected_sizes
    for batch in critic.frozen:
        assert sum(sum(d.loss_fn_inputs["weights"].data) for d in batch) == pytest.approx(1)
    expected = loss_datums(prepared, "critic")
    for epoch in range(2):
        flat = [d for batch in critic.frozen[epoch * actual_count : (epoch + 1) * actual_count] for d in batch]
        for actual, original in zip(flat, expected, strict=True):
            for key in ("values", "returns"):
                assert actual.loss_fn_inputs[key].data == original.loss_fn_inputs[key].data
    assert critic.weight.item() != pytest.approx(0.2)
    if not warmup:
        assert actor.frozen[0] == loss_datums(prepared, "policy")


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_method", ["save_state_async", "save_weights_for_sampler_async"])
async def test_pair_failure_keeps_manifest_and_restores_adam_for_next_update(tmp_path, failed_method):
    actor, critic, bus = clients()
    prepared, _ = prepare_batch([datum()], [{"values": tensor([0.0, 0.1, 0.2])}], "none")
    inputs = loss_datums(prepared, "critic")

    async def step(client):
        await (await client.forward_backward_async(inputs, "ppo", {"value_clip": 0.2})).result_async()
        await (await client.optim_step_async(tinker.AdamParams(learning_rate=0.01))).result_async()

    await step(critic)
    state = await save_pair(
        actor,
        critic,
        tmp_path,
        batch=1,
        wandb_run_id="run",
    )
    before = (tmp_path / MANIFEST_NAME).read_bytes()

    async def fail_save(name):
        raise RuntimeError("disk failure")

    setattr(actor, failed_method, fail_save)
    with pytest.raises(RuntimeError, match="disk failure"):
        await save_pair(actor, critic, tmp_path, batch=2, wandb_run_id="run")
    assert (tmp_path / MANIFEST_NAME).read_bytes() == before
    await step(critic)
    expected_weight = critic.weight.detach().clone()
    expected_adam = deepcopy(critic.optimizer.state_dict())
    restored_actor, restored_critic = ToyClient("policy", bus), ToyClient("critic", bus)
    await restore_pair(restored_actor, restored_critic, state)
    await step(restored_critic)
    torch.testing.assert_close(restored_critic.weight, expected_weight)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            restored_critic.optimizer.state_dict()["state"][0][key], expected_adam["state"][0][key]
        )


def make_trainer(tmp_path, cfg):
    from platoon.train.tinker.rl import PlatoonTinkerRLTrainer

    trainer = PlatoonTinkerRLTrainer.__new__(PlatoonTinkerRLTrainer)
    cfg.log_path = str(tmp_path)
    cfg.watchdog.enabled = False
    trainer.config = cfg
    trainer.train_dataset = Dataset.from_list([{"task_id": str(i)} for i in range(4)])
    trainer.eval_dataset = None
    trainer.batch_transforms = []
    trainer._ppo_resume = None
    trainer._train_tracker = StatsTracker("train")
    trainer.records = []
    trainer._stats_logger = SimpleNamespace(wandb_run_id="run", log=lambda **kwargs: trainer.records.append(kwargs))
    trainer._setup_signal_handlers = lambda event: None
    llm = SimpleNamespace(version=0)
    llm.set_version = lambda version: setattr(llm, "version", version)
    llm.update_sampling_client = lambda client, increment_version=True: setattr(
        llm, "version", llm.version + int(increment_version)
    )
    trainer._model_info = SimpleNamespace(llm=llm)
    return trainer


@pytest.mark.asyncio
async def test_checkpoint_history_resumes_previous_pair_after_removing_last_line(tmp_path):
    actor, critic, bus = clients()
    first = await save_pair(actor, critic, tmp_path, batch=10, wandb_run_id="run")
    expected_actor, expected_critic = actor.weight.detach().clone(), critic.weight.detach().clone()
    with torch.no_grad():
        actor.weight.add_(1)
        critic.weight.add_(2)
    second = await save_pair(actor, critic, tmp_path, batch=20, wandb_run_id="run")
    path = tmp_path / MANIFEST_NAME
    lines = path.read_text().splitlines(keepends=True)
    assert [json.loads(line) for line in lines] == [first, second]
    cfg = config()
    assert load_resume_manifest(cfg, str(tmp_path)) == second

    path.write_text("".join(lines[:-1]))
    state = load_resume_manifest(cfg, str(tmp_path))
    assert state == first
    restored_actor, restored_critic = ToyClient("policy", bus), ToyClient("critic", bus)
    await restore_pair(restored_actor, restored_critic, state)
    torch.testing.assert_close(restored_actor.weight, expected_actor)
    torch.testing.assert_close(restored_critic.weight, expected_critic)


def install_clients(monkeypatch, actor, critic):
    class Service:
        async def create_lora_training_client_async(self, name, rank):
            assert rank == 0
            return actor

    async def create(service, name):
        return critic

    monkeypatch.setattr(tinker, "ServiceClient", lambda **kwargs: Service())
    monkeypatch.setattr("platoon.train.tinker.critic.create_critic", create)


@pytest.mark.asyncio
async def test_checkpoint_sampler_paths_and_between_checkpoint_publication(tmp_path, monkeypatch):
    cfg = config()
    cfg.train.max_training_steps = 3
    cfg.checkpoint.strategy = "step"
    cfg.checkpoint.every = 2
    actor, critic, bus = clients()
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)

    class Workflow:
        async def arun_episode(self, task):
            return [datum(float(int(task["task_id"]) % 2))]

    await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    records = [json.loads(line) for line in (Path(trainer.run_log_path) / MANIFEST_NAME).read_text().splitlines()]
    assert [record["batch"] for record in records] == [0, 2, 3]
    for record in records:
        name = f"{record['batch']:06d}"
        assert record["sampler_path"] == f"tinker://policy/sampler_weights/{name}"
        assert record["paths"] == {
            "policy": f"tinker://policy/weights/{name}",
            "critic": f"tinker://critic/weights/{name}",
        }
    assert [event[1] for event in bus["events"] if event[0] == "create_sampler"] == [
        records[0]["sampler_path"],
        records[1]["sampler_path"],
    ]
    assert sum(event[0] == "publish" for event in bus["events"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("loss_fn", ["ppo", "cispo"])
@pytest.mark.parametrize("explicit_resume", [False, True])
async def test_existing_async_loop_overlaps_rollouts_and_resumes_pair(tmp_path, monkeypatch, loss_fn, explicit_resume):
    cfg = config()
    cfg.train.loss_fn = loss_fn
    cfg.train.ppo.actor_epochs = cfg.train.ppo.critic_epochs = 2
    cfg.checkpoint.strategy = "none"
    actor, critic, bus = clients()
    bus.update(training_started=asyncio.Event(), overlap=asyncio.Event())
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)

    class Workflow:
        count = 0

        async def arun_episode(self, task):
            self.count += 1
            if self.count > cfg.train.batch_size:
                await bus["training_started"].wait()
                bus["overlap"].set()
            return [datum(float(int(task["task_id"]) % 2))]

    await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    assert bus["overlap"].is_set()
    assert (actor.steps, critic.steps) == (2, 4)
    steps = [event[1] for event in bus["events"] if event[0] == "step"]
    assert steps == ["critic"] * 4 + ["policy"] * 2
    update_events = {"backward", "step", "backward_result", "step_result"}
    assert [event[:2] for event in bus["events"] if event[0] in update_events] == [
        (event, role) for role in steps for event in ("backward", "step", "backward_result", "step_result")
    ]
    for key in ("values", "returns", "weights"):
        torch.testing.assert_close(
            critic.frozen[0][0].loss_fn_inputs[key].to_torch(), critic.frozen[1][0].loss_fn_inputs[key].to_torch()
        )
    assert all(event[2] == loss_fn for event in bus["events"] if event[:2] == ("backward", "policy"))
    assert "train/critic/value_loss" in trainer.records[-1]["stats"]
    for record in trainer.records:
        stats = record["stats"]
        assert stats["train/timing/critic_forward_total"] > 0
        assert stats["train/timing/prepare_batch"] > 0
        for operation in ("forward_backward", "optim_step"):
            critic_time = stats[f"train/timing/critic_{operation}_total"]
            policy_time = stats.get(f"train/timing/policy_{operation}_total", 0.0)
            assert critic_time > 0
            assert stats[f"timing/{operation}_total"] == pytest.approx(critic_time + policy_time)
        measured = sum(
            stats[key]
            for key in (
                "timing/rollout_queue_wait",
                "timing/forward_backward_total",
                "timing/optim_step_total",
                "timing/update_weights",
                "train/timing/critic_forward_total",
                "train/timing/prepare_batch",
                "timing/batch_other",
            )
        )
        assert measured == pytest.approx(stats["timing/batch_total"])
    assert "train/timing/policy_optim_step_total" not in trainer.records[0]["stats"]
    assert trainer.records[-1]["stats"]["train/timing/policy_optim_step_total"] > 0
    manifest = Path(trainer.run_log_path) / MANIFEST_NAME
    if explicit_resume:
        cfg.checkpoint.load_checkpoint_path = str(manifest)
        cfg.log_path = str(tmp_path / "resumed")
    trainer._ppo_resume = load_resume_manifest(cfg, trainer.run_log_path)
    cfg.train.max_training_steps = 3
    resumed_actor, resumed_critic = ToyClient("policy", bus), ToyClient("critic", bus)
    install_clients(monkeypatch, resumed_actor, resumed_critic)
    await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    assert (resumed_actor.steps, resumed_critic.steps) == (4, 6)
    assert json.loads((Path(trainer.run_log_path) / MANIFEST_NAME).read_text().splitlines()[-1])["batch"] == 3


@pytest.mark.parametrize("local_state", ["missing", "empty", "populated"])
@pytest.mark.parametrize("external_exists", [False, True])
def test_resume_prefers_current_run_then_external_manifest(tmp_path, local_state, external_exists):
    cfg = config()
    external = tmp_path / "source.jsonl"
    cfg.checkpoint.load_checkpoint_path = str(external)
    source = {"algorithm": "ppo", "batch": 100, "paths": {"policy": "actor-100", "critic": "critic-100"}}
    local = {"algorithm": "ppo", "batch": 120, "paths": {"policy": "actor-120", "critic": "critic-120"}}
    if external_exists:
        external.write_text(json.dumps(source) + "\n")
    if local_state != "missing":
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(local) + "\n" if local_state == "populated" else "\n")
    if local_state == "populated":
        assert load_resume_manifest(cfg, str(tmp_path)) == local
    elif external_exists:
        assert load_resume_manifest(cfg, str(tmp_path)) == source
    else:
        with pytest.raises(FileNotFoundError):
            load_resume_manifest(cfg, str(tmp_path))


def test_explicit_ppo_checkpoint_does_not_fall_back_to_fresh_training(tmp_path):
    cfg = config()
    manifest = tmp_path / MANIFEST_NAME
    cfg.checkpoint.load_checkpoint_path = str(manifest)
    with pytest.raises(FileNotFoundError):
        load_resume_manifest(cfg, str(tmp_path))
    manifest.write_text(json.dumps({"algorithm": "ppo", "paths": {"policy": "actor-only"}}))
    with pytest.raises(ValueError, match="complete actor/critic checkpoint pair"):
        load_resume_manifest(cfg, str(tmp_path))


@pytest.mark.asyncio
async def test_all_zero_mask_batch_does_not_update_either_model(tmp_path, monkeypatch):
    cfg = config()
    cfg.train.ppo.critic_warmup_batches = 0
    actor, critic, _ = clients()
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)

    class Workflow:
        async def arun_episode(self, task):
            row = datum()
            row.loss_fn_inputs["mask"] = tensor([0.0, 0.0, 0.0])
            return [row]

    await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    assert actor.steps == critic.steps == 0
    assert not actor.frozen and not critic.frozen


@pytest.mark.asyncio
async def test_actor_failure_does_not_checkpoint_partial_pair(tmp_path, monkeypatch):
    cfg = config()
    cfg.train.ppo.critic_warmup_batches = 0
    actor, critic, _ = clients()
    actor.fail_backward = True
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)

    class Workflow:
        async def arun_episode(self, task):
            return [datum()]

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    manifest = Path(trainer.run_log_path) / MANIFEST_NAME
    state = json.loads(manifest.read_text().splitlines()[-1])
    assert state["batch"] == 0
    assert critic.steps == 1 and actor.steps == 0


@pytest.mark.parametrize("loss_fn,low,high", [("ppo", 0.8, 1.2), ("cispo", 0.0, 5.0)])
def test_actor_loss_defaults(loss_fn, low, high):
    cfg = TrainConfig(model_name="test", renderer_name="test", algorithm="ppo", lora_rank=0, loss_fn=loss_fn)
    assert cfg.loss_fn_config == {"clip_low_threshold": low, "clip_high_threshold": high}
    assert cfg.workflow_config.algorithm == "ppo"


def test_negative_extra_rollout_groups_rejected():
    with pytest.raises(ValueError, match="extra_rollout_groups"):
        config(extra_rollout_groups=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size,extra_groups,group_size", [(4, 1, 3), (256, 8, 1)])
async def test_extra_groups_bypass_straggler_without_increasing_workers(
    tmp_path, monkeypatch, batch_size, extra_groups, group_size
):
    cfg = config(extra_rollout_groups=extra_groups, max_staleness=1)
    cfg.train.batch_size = cfg.train.num_concurrent_rollout_workflow_workers = batch_size
    cfg.train.workflow_config.group_size = group_size
    cfg.train.max_training_steps = 1
    actor, critic, _ = clients()
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)
    trainer.train_dataset = Dataset.from_list([{"task_id": str(i)} for i in range(batch_size + extra_groups)])
    updates = []

    async def update(trainer, shared, rows):
        updates.append(rows)

    monkeypatch.setattr("platoon.train.tinker.ppo.update_batch", update)

    class Workflow:
        started = active = peak = 0
        straggler_cancelled = False

        async def arun_episode(self, task):
            self.started += 1
            index = self.started
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if index == 1:
                    try:
                        await asyncio.Event().wait()
                    finally:
                        self.straggler_cancelled = True
                await asyncio.sleep(0)
                return [datum() for _ in range(group_size)]
            finally:
                self.active -= 1

    workflow = Workflow()
    await asyncio.wait_for(trainer.train(workflow), timeout=10)
    assert workflow.peak == batch_size
    assert batch_size < workflow.started <= batch_size + extra_groups
    assert workflow.active == 0 and workflow.straggler_cancelled
    assert len(updates) == 1 and len(updates[0]) == batch_size * group_size
    assert trainer.records[0]["stats"]["train/accepted_rollout_groups"] == batch_size


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_result", ["failed", "empty", "stale"])
@pytest.mark.parametrize("extra_groups", [0, 1])
async def test_extra_groups_replace_ineligible_results(tmp_path, monkeypatch, bad_result, extra_groups):
    cfg = config(extra_rollout_groups=extra_groups, max_staleness=1)
    cfg.train.max_training_steps = 1
    actor, critic, _ = clients()
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)

    class Workflow:
        started = 0

        async def arun_episode(self, task):
            self.started += 1
            if self.started == 1:
                if bad_result == "failed":
                    raise RuntimeError("test rollout failure")
                if bad_result == "empty":
                    return []
            row = datum()
            # At step 0, version -1 is eligible; version -2 must be discarded.
            row.loss_fn_inputs["checkpoint_version"] = tensor([-2 if self.started == 1 else -1])
            return [row]

    await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    assert len(critic.frozen[0]) == 1 + extra_groups
    assert trainer.records[0]["stats"]["train/total_datums_per_batch"] == 1 + extra_groups


@pytest.mark.asyncio
async def test_extra_group_budget_exhaustion_does_not_hang_or_sample_more(tmp_path, monkeypatch, caplog):
    cfg = config(extra_rollout_groups=1, max_staleness=1)
    cfg.train.max_training_steps = 1
    actor, critic, _ = clients()
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)

    class Workflow:
        started = 0

        async def arun_episode(self, task):
            self.started += 1
            return None if self.started <= 2 else [datum()]

    workflow = Workflow()
    await asyncio.wait_for(trainer.train(workflow), timeout=10)
    assert workflow.started == 3  # B + K, even when too few eligible results remain.
    assert len(critic.frozen[0]) == 1
    assert "Extra rollout allowance exhausted: using 1/2 eligible groups" in caplog.text


@pytest.mark.asyncio
async def test_extra_group_results_carry_into_next_batch_when_fresh(tmp_path, monkeypatch):
    cfg = config(extra_rollout_groups=1, max_staleness=1)
    actor, critic, _ = clients()
    install_clients(monkeypatch, actor, critic)
    trainer = make_trainer(tmp_path, cfg)
    started = 0
    all_sampled = asyncio.Event()
    updates = []

    class Workflow:
        async def arun_episode(self, task):
            nonlocal started
            started += 1
            row = datum()
            row.loss_fn_inputs["checkpoint_version"] = tensor([trainer.model_info.llm.version])
            if started == 6:
                all_sampled.set()
            return [row]

    async def update(trainer, shared, rows):
        # Let both sampling waves finish on version 0 before the first update.
        await all_sampled.wait()
        updates.append(rows)

    monkeypatch.setattr("platoon.train.tinker.ppo.update_batch", update)
    await asyncio.wait_for(trainer.train(Workflow()), timeout=10)
    assert started == 6 and [len(rows) for rows in updates] == [2, 2]
    assert all(row.loss_fn_inputs["checkpoint_version"].data == [0] for rows in updates for row in rows)


@pytest.mark.parametrize("field,value", [("gamma", 0.99), ("gae_lambda", 0.95)])
def test_nonunit_gae_is_supported(field, value):
    from platoon.train.tinker.config_defs import PPOConfig

    cfg = config(ppo=PPOConfig(**{field: value}))
    assert getattr(cfg.train.ppo, field) == value


@pytest.mark.asyncio
async def test_existing_grpo_loop_still_uses_only_actor(tmp_path, monkeypatch):
    from platoon.train.tinker.rl import PlatoonTinkerDataloader, TerminateTrainLoop, TrainLoopSharedState

    cfg = config()
    cfg.train.algorithm = cfg.train.workflow_config.algorithm = "grpo"
    cfg.train.loss_fn = "cispo"
    cfg.train.max_training_steps = 1
    actor, _, bus = clients()
    trainer = make_trainer(tmp_path, cfg)

    async def publish(**kwargs):
        return await actor.save_weights_and_get_sampling_client_async()

    monkeypatch.setattr(trainer, "_save_checkpoint_and_get_sampling_client", publish)
    shared = TrainLoopSharedState(
        config=cfg,
        shutdown_event=asyncio.Event(),
        train_step=0,
        train_dataloader=PlatoonTinkerDataloader(trainer.train_dataset, cfg.train.batch_size),
        eval_dataloader=None,
        optimizer_params=tinker.AdamParams(learning_rate=1e-5),
        service_client=None,
        training_client=actor,
        sampling_client=None,
        sampling_client_step=0,
        sampling_client_updated_event=asyncio.Event(),
        model_info=trainer.model_info,
        stats_logger=trainer._stats_logger,
        train_tracker=trainer._train_tracker,
    )

    class Workflow:
        async def arun_episode(self, task):
            row = datum()
            row.loss_fn_inputs.pop("returns")
            for key in list(row.loss_fn_inputs):
                if key.startswith("_ppo_"):
                    del row.loss_fn_inputs[key]
            return [row]

    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(trainer._train_loop(group, shared, 0, 1, Workflow()))
    except* TerminateTrainLoop:
        pass
    assert actor.steps == 1
    assert [e for e in bus["events"] if e[0] == "backward"] == [("backward", "policy", "cispo")]
    assert shared.critic_client is None
