# ruff: noqa: E402
"""CodeScout training with Platoon's current AReaL backend."""

from __future__ import annotations

import logging
import sys


def configure_plain_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.getLogger("openhands").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)


configure_plain_logging()

from areal.api.cli_args import load_expr_config
from datasets import Dataset
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import GroupRolloutWorkflow

from platoon.codescout.rollout import run_rollout
from platoon.codescout.tasks import get_task, load_data


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config

    train_datamap, val_datamap = load_data()
    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_datamap])
    val_dataset = Dataset.from_list([{"task_id": task_id} for task_id in val_datamap])

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.workflow_config,
            proxy_base_url=trainer.proxy_base_url,
            proxy_admin_api_key=trainer.proxy_admin_api_key,
            output_subdir="train_rollout",
            filter_errors=False,
        )

        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.workflow_config,
            proxy_base_url=trainer.eval_proxy_base_url or trainer.proxy_base_url,
            proxy_admin_api_key=trainer.proxy_admin_api_key,
            output_subdir="eval_rollout",
            filter_errors=False,
        )

        trainer.train(workflow=train_workflow, eval_workflow=eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
