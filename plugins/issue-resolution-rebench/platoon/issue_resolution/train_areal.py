# ruff: noqa: E402
"""SWE-rebench issue-resolution training script with the AReaL backend."""

import logging
import sys


def configure_plain_logging() -> None:
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

from areal.api.cli_args import load_expr_config
from datasets import Dataset
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig

from platoon.issue_resolution.areal_workflow import OgmaRoutedGroupRolloutWorkflow
from platoon.issue_resolution.ogma_tunnel import OgmaReverseTunnelManager
from platoon.issue_resolution.rollout import run_rollout
from platoon.issue_resolution.tasks import get_task, load_data


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config

    train_datamap, _ = load_data()
    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_datamap])
    print(f"Training dataset size: {len(train_dataset)}")

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=None,
    ) as trainer:
        with OgmaReverseTunnelManager(trainer.rollout.proxy_addrs) as tunnels:
            train_workflow = OgmaRoutedGroupRolloutWorkflow(
                rollout_fn=run_rollout,
                get_task_fn=get_task,
                config=config.workflow_config,
                proxy_base_url=trainer.proxy_base_url,
                proxy_admin_api_key=trainer.proxy_admin_api_key,
                output_subdir="train_rollout",
                filter_errors=False,
                proxy_endpoint_map=tunnels.endpoint_map,
            )
            trainer.train(workflow=train_workflow, eval_workflow=None)


if __name__ == "__main__":
    main(sys.argv[1:])
