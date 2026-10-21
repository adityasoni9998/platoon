"""Token/turn GAE and paired checkpoints, using ordinary Tinker clients."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tinker
from numpy.typing import NDArray
from tinker import TensorData

MANIFEST_NAME = "ppo_checkpoints.jsonl"


@dataclass
class PreparedDatum:
    """Keep PPO working arrays local until constructing each role's SDK request."""

    datum: tinker.Datum
    mask: NDArray[np.bool_]
    critic_mask: NDArray[np.bool_]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]
    old_values: NDArray[np.float32]


def compute_gae(values: np.ndarray, rewards: np.ndarray, gamma: float, gae_lambda: float):
    """GAE over logical decisions, with zero bootstrap at the rollout endpoint.

    Use the equivalent lambda-return recurrence so lambda=1 produces discounted
    reward-to-go directly, without cancellation of intermediate critic values.
    Observation tokens and prefix-split boundaries are not decisions.
    """
    # Python floats keep the recurrence in double precision until the final array cast.
    value_list, reward_list = values.tolist(), rewards.tolist()
    return_list = [0.0] * len(value_list)
    next_return = next_value = 0.0
    for index in reversed(range(len(value_list))):
        return_list[index] = reward_list[index] + gamma * ((1 - gae_lambda) * next_value + gae_lambda * next_return)
        next_return, next_value = return_list[index], value_list[index]
    returns = np.asarray(return_list, dtype=values.dtype)
    return returns - values, returns


def _action_blocks(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    # Target-aligned mask: the value at a block's start is computed from the
    # final prompt token, before generating the first action token.
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def prepare_batch(
    datums: list[tinker.Datum],
    outputs,
    normalization: str,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    value_granularity: str = "token",
    advantage_normalization_granularity: str | None = None,
) -> tuple[list[PreparedDatum], dict]:
    """Compute rollout-wide GAE, then freeze critic targets and whiten actor advantages.

    Datums may arrive in any order. Each carries a trajectory key, an inclusive
    turn range, and rewards aligned with contiguous action-mask blocks. Prefix
    breaks never reset GAE. Rollout endpoints retain the existing zero-bootstrap
    convention (including time/context limits); nonterminal bootstrapping would
    additionally require a recorded final observation and its critic value.

    Critic loss and value statistics count each trained value position once.
    Whitening and advantage statistics default to that same granularity, but
    turn critics can use token-level whitening. Actor loss remains a token mean.
    """
    if advantage_normalization_granularity is None:
        advantage_normalization_granularity = value_granularity
    if advantage_normalization_granularity not in ("token", "turn"):
        raise ValueError("advantage_normalization_granularity must be token or turn")
    if value_granularity == "token" and advantage_normalization_granularity == "turn":
        raise ValueError("Turn-level advantage normalization requires value_granularity=turn")
    advantage_mask_key = "mask" if advantage_normalization_granularity == "token" else "critic_mask"
    groups = {}
    values, masks, blocks = [], [], []
    for index, (datum, output) in enumerate(zip(datums, outputs, strict=True)):
        inputs = datum.loss_fn_inputs
        mask = inputs["mask"].to_numpy().astype(np.bool_, copy=False)
        value = output["values"].to_numpy().astype(np.float32, copy=False)
        logprobs = inputs["logprobs"]
        logprobs_shape = tuple(logprobs.shape if logprobs.shape is not None else [len(logprobs.data)])

        if not (value.shape == mask.shape == logprobs_shape) or len(logprobs.data) != mask.size:
            raise ValueError("PPO values, logprobs, and action masks must align with target tokens")
        values.append(value)
        masks.append(mask)
        blocks.append(_action_blocks(mask))
        if not mask.any():
            continue
        key = inputs["_ppo_trajectory_id"].data[0]
        first, last = inputs["_ppo_turn_range"].to_numpy().tolist()
        if not (
            len(blocks[index])
            == last - first + 1
            == len(inputs["_ppo_rewards"].data)
            == len(inputs["_ppo_train_turns"].data)
        ):
            raise ValueError("PPO turn range/rewards must match the action-mask blocks")
        groups.setdefault(key, []).append((first, last, index))

    advantages = [np.zeros_like(value) for value in values]
    returns = [np.zeros_like(value) for value in values]
    actor_masks = [np.zeros_like(mask) for mask in masks]
    critic_masks = [np.zeros_like(mask) for mask in masks]
    for segments in groups.values():
        turns = []
        expected_turn = 0
        for first, last, index in sorted(segments):
            if first != expected_turn:
                raise ValueError("PPO trajectory has missing or overlapping turns")
            expected_turn = last + 1
            inputs = datums[index].loss_fn_inputs
            rewards = inputs["_ppo_rewards"].to_numpy().tolist()
            train_turns = inputs["_ppo_train_turns"].to_numpy().tolist()
            turns.extend(
                (index, start, end, reward, train)
                for (start, end), reward, train in zip(blocks[index], rewards, train_turns, strict=True)
            )

        decision_values = np.concatenate(
            [
                values[index][start:end] if value_granularity == "token" else values[index][start : start + 1]
                for index, start, end, _, _ in turns
            ]
        )
        decision_rewards = np.zeros_like(decision_values)
        offset = 0
        for _, start, end, reward, _ in turns:
            offset += end - start if value_granularity == "token" else 1
            decision_rewards[offset - 1] = reward  # A turn's reward occurs after its last action token.
        adv, targets = compute_gae(decision_values, decision_rewards, gamma, gae_lambda)

        offset = 0
        for index, start, end, _, train in turns:
            size = end - start if value_granularity == "token" else 1
            advantages[index][start:end] = adv[offset : offset + size]
            returns[index][start : start + size] = targets[offset : offset + size]
            actor_masks[index][start:end] = bool(train)
            critic_masks[index][start : start + size] = bool(train)
            offset += size

    prepared = []
    for index, datum in enumerate(datums):
        if not actor_masks[index].any():
            continue
        prepared.append(
            PreparedDatum(
                datum=datum,
                mask=actor_masks[index],
                critic_mask=critic_masks[index],
                returns=np.where(critic_masks[index], returns[index], np.float32(0)),
                old_values=np.where(critic_masks[index], values[index], np.float32(0)),
                advantages=np.where(actor_masks[index], advantages[index], np.float32(0)),
            )
        )
    if not prepared:
        return [], {}
    metrics = {}
    for field in ("old_values", "returns", "advantages"):
        mask_key = advantage_mask_key if field == "advantages" else "critic_mask"
        flat = np.concatenate([getattr(d, field)[getattr(d, mask_key)] for d in prepared])
        label = "raw_advantages" if field == "advantages" else field
        metrics[f"critic/{label}_mean"] = flat.mean().item()
        metrics[f"critic/{label}_std"] = flat.std().item()
        metrics[f"critic/{label}_min"] = flat.min().item()
        metrics[f"critic/{label}_max"] = flat.max().item()
        if field == "advantages":
            adv_mean = flat.mean()
            adv_std = flat.std()
    if normalization == "batch":
        for datum in prepared:
            datum.advantages = np.where(
                datum.mask,
                (datum.advantages - adv_mean) / np.sqrt(adv_std * adv_std + np.float32(1e-8)),
                np.float32(0),
            )
    normalized = np.concatenate([d.advantages[getattr(d, advantage_mask_key)] for d in prepared])
    metrics["critic/normalized_advantages_mean"] = normalized.mean().item()
    metrics["critic/normalized_advantages_std"] = normalized.std().item()
    metrics["critic/normalized_advantages_min"] = normalized.min().item()
    metrics["critic/normalized_advantages_max"] = normalized.max().item()
    return prepared, metrics


def loss_datums(datums: list[PreparedDatum], role: str) -> list[tinker.Datum]:
    """Mean critic loss over value positions; mean actor loss over action tokens."""
    mask_key = "critic_mask" if role == "critic" else "mask"
    count = sum(np.count_nonzero(getattr(d, mask_key)) for d in datums)
    scale = np.float32(1.0 / (count + 1e-8))
    output = []
    for datum in datums:
        inputs = datum.datum.loss_fn_inputs
        fields = {"target_tokens": inputs["target_tokens"]}
        if role == "critic":
            fields.update(
                values=TensorData.from_numpy(datum.old_values),
                returns=TensorData.from_numpy(datum.returns),
                weights=TensorData.from_numpy(datum.critic_mask * scale),
            )
        else:
            fields.update(logprobs=inputs["logprobs"], advantages=TensorData.from_numpy(datum.advantages * scale))
        output.append(tinker.Datum(model_input=datum.datum.model_input, loss_fn_inputs=fields))
    return output


def _actor_metric_datums(datums: list[PreparedDatum]) -> list[tinker.Datum]:
    """Adapt only the fields consumed by the shared actor-metrics implementation."""
    return [
        tinker.Datum(
            model_input=d.datum.model_input,
            loss_fn_inputs={
                "mask": TensorData.from_numpy(d.mask.astype(np.float32)),
                "logprobs": d.datum.loss_fn_inputs["logprobs"],
            },
        )
        for d in datums
    ]


async def update_batch(trainer, shared_state, datums: list[tinker.Datum]) -> None:
    """Prepare one accepted batch, then finish critic updates before actor work."""
    from platoon.train.tinker.rl import compute_training_metrics

    config = trainer.config.train
    critic = shared_state.critic_client
    actor = shared_state.training_client
    tracker = shared_state.train_tracker

    async def resolve(future, name):
        return await trainer._await_with_heartbeat(
            future.result_async(), name=name, watchdog=shared_state.watchdog, total_timeout=1800.0
        )

    inputs = [
        tinker.Datum(
            model_input=datum.model_input, loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]}
        )
        for datum in datums
    ]
    start = time.perf_counter()
    future = await critic.forward_async(inputs, loss_fn="ppo")
    forward_submit = time.perf_counter() - start
    start = time.perf_counter()
    outputs = (await resolve(future, "critic values")).loss_fn_outputs
    forward_wait = time.perf_counter() - start
    tracker.scalar(
        **{
            "timing/critic_forward_submit": forward_submit,
            "timing/critic_forward_wait": forward_wait,
            "timing/critic_forward_total": forward_submit + forward_wait,
        }
    )
    start = time.perf_counter()
    prepared, stats = prepare_batch(
        datums,
        outputs,
        config.ppo.advantage_normalization,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        value_granularity=config.ppo.value_granularity,
        advantage_normalization_granularity=config.ppo.advantage_normalization_granularity,
    )
    tracker.scalar(**{"timing/prepare_batch": time.perf_counter() - start})
    if not prepared:
        return
    tracker.scalar(**stats)
    # These stay fixed for every optimization epoch on this batch.
    roles = [
        (
            "critic",
            critic,
            config.ppo.critic_epochs,
            config.ppo.critic.optimizer,
            {"value_clip": config.ppo.critic.value_clip},
        )
    ]
    if shared_state.train_step >= config.ppo.critic_warmup_batches:
        roles.append(("policy", actor, config.ppo.actor_epochs, config.optimizer, config.loss_fn_config))
    for role, client, epochs, optimizer, loss_config in roles:
        inputs = loss_datums(prepared, role)
        metric_inputs = _actor_metric_datums(prepared) if role == "policy" else []
        loss_fn = "ppo" if role == "critic" else config.loss_fn
        timings = dict(
            forward_backward_submit=0.0, forward_backward_wait=0.0, optim_step_submit=0.0, optim_step_wait=0.0
        )
        for _ in range(epochs):
            start = time.perf_counter()
            backward_future = await client.forward_backward_async(inputs, loss_fn=loss_fn, loss_fn_config=loss_config)
            timings["forward_backward_submit"] += time.perf_counter() - start
            start = time.perf_counter()
            optim_future = await client.optim_step_async(tinker.AdamParams(**asdict(optimizer)))
            timings["optim_step_submit"] += time.perf_counter() - start
            start = time.perf_counter()
            result = await resolve(backward_future, f"{role} forward_backward")
            timings["forward_backward_wait"] += time.perf_counter() - start
            loss = result.metrics["loss:sum"]
            if not math.isfinite(loss):
                raise ValueError(f"Nonfinite {role} loss")
            if role == "critic":
                deltas = [
                    (output["values"].to_numpy() - datum.old_values)[datum.critic_mask]
                    for datum, output in zip(prepared, result.loss_fn_outputs, strict=True)
                ]
                tracker.scalar(
                    **{
                        "critic/value_loss": loss,
                        "critic/value_clipfrac": (np.abs(np.concatenate(deltas)) > config.ppo.critic.value_clip)
                        .astype(np.float32)
                        .mean()
                        .item(),
                    }
                )
            else:
                tracker.scalar(**compute_training_metrics(metric_inputs, result, loss_fn, loss_config))
                tracker.scalar(**{"optim/pg_loss": loss})
            start = time.perf_counter()
            result = await resolve(optim_future, f"{role} optim_step")
            timings["optim_step_wait"] += time.perf_counter() - start
            metrics = result.metrics or {}
            if metrics.get("skipped"):
                raise RuntimeError(f"{role} optimizer skipped; resume the last complete PPO checkpoint pair")
            prefix = "critic" if role == "critic" else "optim"
            for key, value in metrics.items():
                tracker.scalar(**{f"{prefix}/{key}": value})
        # Sum repeated optimization epochs before logging; scalar() averages repeated calls.
        timings["forward_backward_total"] = timings["forward_backward_submit"] + timings["forward_backward_wait"]
        timings["optim_step_total"] = timings["optim_step_submit"] + timings["optim_step_wait"]
        tracker.scalar(**{f"timing/{role}_{key}": value for key, value in timings.items()})


def load_resume_manifest(config, run_log_path: str) -> dict | None:
    explicit = config.checkpoint.load_checkpoint_path
    path = Path(run_log_path) / MANIFEST_NAME
    lines = [line for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if not lines and explicit:
        path = Path(explicit)
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        if (Path(run_log_path) / "checkpoints.jsonl").exists():
            raise ValueError("This directory contains GRPO checkpoints; use a new run directory for PPO")
        return None
    state = json.loads(lines[-1])
    if state.get("algorithm") != "ppo":
        raise ValueError(f"Unsupported PPO checkpoint manifest: {path}")
    if set(state.get("paths", {})) != {"policy", "critic"} or not all(state["paths"].values()):
        raise ValueError("PPO resume requires a complete actor/critic checkpoint pair")
    return state


async def save_pair(actor, critic, run_log_path, *, batch, wandb_run_id):
    """Save both training states and actor sampling weights, then append their paths."""
    name = f"{batch:06d}"
    future = await critic.save_state_async(name)
    critic_path = (await future.result_async()).path
    actor_future = await actor.save_state_async(name)
    sampler_future = await actor.save_weights_for_sampler_async(name)
    actor_path = (await actor_future.result_async()).path
    sampler_path = (await sampler_future.result_async()).path
    state = {
        "algorithm": "ppo",
        "batch": batch,
        "paths": {"critic": critic_path, "policy": actor_path},
        "sampler_path": sampler_path,
        "wandb_run_id": wandb_run_id,
    }
    path = Path(run_log_path) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(state) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return state


async def restore_pair(actor, critic, state):
    for role, client in (("critic", critic), ("policy", actor)):
        future = await client.load_state_with_optimizer_async(state["paths"][role])
        await future.result_async()


def checkpoint_due(event, completed: int, start_batch: int, batches_per_epoch: int) -> bool:
    if event.strategy == "none" or event.every <= 0:
        return False
    interval = event.every * (batches_per_epoch if event.strategy == "epoch" else 1)
    return (completed - start_batch) % interval == 0
