"""Train DAPO-Math-17K through the existing async Tinker trainer."""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from datasets import Dataset
from platoon.dapo_math.grader import get_deepscaler_rule_based_reward
from platoon.dapo_math.rollout import reward_processor, run_rollout
from platoon.dapo_math.tasks import DataConfig, load_tasks

from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config


@dataclass
class DapoMathConfig(PlatoonTinkerRLTrainerConfig):
    data: DataConfig = field(default_factory=DataConfig)


async def main(args: list[str]) -> None:
    config, _ = load_config(
        args=args,
        config_class=DapoMathConfig,
        default_config_path=str(Path(__file__).with_name("dapo_math_ppo_tinker.yaml")),
    )
    tasks = load_tasks(config.data)
    if len(tasks) < config.train.batch_size:
        raise ValueError("The dataset must contain at least train.batch_size prompts")
    if config.eval.strategy != "none":
        raise ValueError("This smoke-test plugin has no held-out evaluation dataset; use eval.strategy: none")
    dataset = Dataset.from_list([{"task_id": task_id} for task_id in tasks])
    async with PlatoonTinkerRLTrainer(config=config, train_dataset=dataset) as trainer:
        workflow = GroupRolloutWorkflow(
            rollout_fn=partial(
                run_rollout,
                tokenizer=trainer.model_info.llm.tokenizer,
                grader=get_deepscaler_rule_based_reward,
            ),
            get_task_fn=tasks.__getitem__,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=False,
            reward_processor=reward_processor,
        )
        await trainer.train(workflow)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    asyncio.run(main(sys.argv[1:]))
