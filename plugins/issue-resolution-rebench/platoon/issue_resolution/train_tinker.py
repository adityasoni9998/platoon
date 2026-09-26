# ruff: noqa: E402
"""SWE-rebench issue-resolution training script with Tinker backend."""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path


def configure_plain_logging() -> None:
    """Install plain stderr logging before dependencies auto-configure handlers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logging.getLogger("platoon").setLevel(logging.INFO)
    logging.getLogger("openhands").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


configure_plain_logging()

from datasets import Dataset
from platoon.inference.inference_tunnel import InferenceTunnelConfig
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

from platoon.issue_resolution.rollout import run_rollout
from platoon.issue_resolution.tasks import get_task, load_data
from platoon.issue_resolution.tinker_workflow import TinkerProxyRollout, routed_tinker_proxy


@dataclass
class IssueResolutionTinkerConfig(PlatoonTinkerRLTrainerConfig):
    inference_tunnel: InferenceTunnelConfig = field(
        default_factory=lambda: InferenceTunnelConfig(backend="modal")
    )


def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "train_issue_resolution_fft_tinker.yaml"
    config, _ = load_config(
        args=args,
        config_class=IssueResolutionTinkerConfig,
        default_config_path=str(default_config),
    )
    train_datamap, _ = load_data()
    train_dataset = Dataset.from_list([{"task_id": x} for x in train_datamap.keys()])
    # Create trainer and run with context manager for proper cleanup
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    # Rollout children are reaped before the training loop exits and closes the
    # shared proxy and tunnel. Workflow concurrency bounds the number of children.
    with routed_tinker_proxy(trainer.model_info, config.inference_tunnel) as proxy:
        asyncio.run(train(trainer, TinkerProxyRollout(run_rollout, proxy)))


async def train(trainer, rollout_fn):
    async with trainer:
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=rollout_fn,
            get_task_fn=get_task,
            config=trainer.config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=False,
        )
        await trainer.train(train_workflow=train_workflow, eval_workflow=None)


if __name__ == "__main__":
    main(sys.argv[1:])
