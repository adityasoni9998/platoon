"""Terminal-return PPO math and paired state, using ordinary Tinker clients.

Gamma and lambda are one, and the final bootstrap is zero. Consequently
returns are the trajectory reward at every action token, including across
prefix splits. Actor advantages are whitened after freezing critic targets.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import tinker
import torch
from tinker import TensorData

MANIFEST_NAME = "ppo_checkpoints.jsonl"


def prepare_batch(datums: list[tinker.Datum], outputs, normalization: str) -> tuple[list[tinker.Datum], dict]:
    """Freeze token returns/old values, and whiten only the actor advantages.

    Assumes gamma = lambda = 1, terminal outcome rewards, and zero terminal
    bootstrap. The workflow supplies that outcome reward as each action token's
    return, so GAE reduces to advantages = returns - old_values.
    """
    prepared = []
    for datum, output in zip(datums, outputs, strict=True):
        inputs = datum.loss_fn_inputs
        if "returns" not in inputs:
            raise ValueError("PPO workflow must provide terminal returns, not group-centered advantages")
        mask = inputs["mask"].to_torch().bool()
        values = output["values"].to_torch().float()
        returns = inputs["returns"].to_torch().float()
        logprobs = inputs["logprobs"].to_torch()
        if not (values.shape == returns.shape == mask.shape == logprobs.shape):
            raise ValueError("PPO values, returns, logprobs, and action masks must align with target tokens")
        if not mask.any():
            continue
        values = torch.where(mask, values, 0.0)
        returns = torch.where(mask, returns, 0.0)
        advantages = torch.where(mask, returns - values, 0.0)
        prepared.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    **inputs,
                    "returns": TensorData.from_torch(returns.clone()),
                    "old_values": TensorData.from_torch(values.clone()),
                    "advantages": TensorData.from_torch(advantages),
                },
            )
        )
    if not prepared:
        return [], {}
    metrics = {}
    for field in ("old_values", "returns", "advantages"):
        flat = torch.cat(
            [d.loss_fn_inputs[field].to_torch()[d.loss_fn_inputs["mask"].to_torch().bool()] for d in prepared]
        )
        label = "raw_advantages" if field == "advantages" else field
        metrics[f"critic/{label}_mean"] = flat.mean().item()
        metrics[f"critic/{label}_std"] = flat.std(unbiased=False).item()
        metrics[f"critic/{label}_min"] = flat.min().item()
        metrics[f"critic/{label}_max"] = flat.max().item()
        if field == "advantages":
            adv_mean = flat.mean()
            adv_std = flat.std(unbiased=False)
    if normalization == "batch":
        for datum in prepared:
            mask = datum.loss_fn_inputs["mask"].to_torch().bool()
            adv = datum.loss_fn_inputs["advantages"].to_torch()
            datum.loss_fn_inputs["advantages"] = TensorData.from_torch(
                torch.where(mask, (adv - adv_mean) / torch.sqrt(adv_std.square() + 1e-8), 0.0)
            )
    normalized = torch.cat(
        [d.loss_fn_inputs["advantages"].to_torch()[d.loss_fn_inputs["mask"].to_torch().bool()] for d in prepared]
    )
    metrics["critic/normalized_advantages_mean"] = normalized.mean().item()
    metrics["critic/normalized_advantages_std"] = normalized.std(unbiased=False).item()
    metrics["critic/normalized_advantages_min"] = normalized.min().item()
    metrics["critic/normalized_advantages_max"] = normalized.max().item()
    return prepared, metrics


def loss_datums(datums: list[tinker.Datum], role: str) -> list[tinker.Datum]:
    """GRPO's active-token normalization, with exactly one microbatch per update."""
    count = sum(d.loss_fn_inputs["mask"].to_torch().sum().item() for d in datums)
    scale = 1.0 / (count + 1e-8)
    output = []
    for datum in datums:
        inputs = datum.loss_fn_inputs
        mask = inputs["mask"].to_torch().float()
        fields = {"target_tokens": inputs["target_tokens"]}
        if role == "critic":
            fields.update(
                values=inputs["old_values"],
                returns=inputs["returns"],
                weights=TensorData.from_torch(mask * scale),
            )
        else:
            fields.update(
                logprobs=inputs["logprobs"], advantages=TensorData.from_torch(inputs["advantages"].to_torch() * scale)
            )
        output.append(tinker.Datum(model_input=datum.model_input, loss_fn_inputs=fields))
    return output


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
    prepared, stats = prepare_batch(datums, outputs, config.ppo.advantage_normalization)
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
                    (output["values"].to_torch() - datum.loss_fn_inputs["old_values"].to_torch())[
                        datum.loss_fn_inputs["mask"].to_torch().bool()
                    ]
                    for datum, output in zip(prepared, result.loss_fn_outputs, strict=True)
                ]
                tracker.scalar(
                    **{
                        "critic/value_loss": loss,
                        "critic/value_clipfrac": (torch.cat(deltas).abs() > config.ppo.critic.value_clip)
                        .float()
                        .mean()
                        .item(),
                    }
                )
            else:
                tracker.scalar(**compute_training_metrics(prepared, result, loss_fn, loss_config))
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
