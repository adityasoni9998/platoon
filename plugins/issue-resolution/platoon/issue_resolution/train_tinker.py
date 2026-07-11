"""Issue resolution training script with Tinker backend."""
import asyncio
import logging
import sys
from pathlib import Path

from datasets import Dataset
from platoon.issue_resolution.rollout import run_rollout
from platoon.issue_resolution.tasks import get_task, load_data
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.fastapi_litellm_proxy import FastAPILiteLLMTinkerHTTPProxyServer
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("platoon").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)

async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "train_issue_resolution_fft_tinker.yaml"
    config, raw_config = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )
    train_datamap, val_datamap = load_data()
    train_dataset = Dataset.from_list([{ "task_id": x } for x in train_datamap.keys()])
    # val_dataset = Dataset.from_list([{ "task_id": x } for x in val_datamap.keys()])
    # Create trainer and run with context manager for proper cleanup
    print(f"Training dataset size: {len(train_dataset)}")
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    async with trainer:
        old_model_name = trainer.model_info.model_name
        old_base_url = trainer.model_info.base_url
        old_api_key = trainer.model_info.api_key
        tinker_proxy = FastAPILiteLLMTinkerHTTPProxyServer(
            litellm_model_name=old_model_name,
            context_window_length=trainer.model_info.llm.context_window_length,
        )
        tinker_proxy.start()
        trainer.model_info.model_name = tinker_proxy.model_name
        trainer.model_info.base_url = tinker_proxy.base_url
        trainer.model_info.api_key = tinker_proxy.api_key
        try:
            # Create workflows - use trainer.run_log_path for run-specific output
            train_workflow = GroupRolloutWorkflow(
                rollout_fn=run_rollout,
                get_task_fn=get_task,
                config=config.train.workflow_config,
                model_info=trainer.model_info,
                log_path=trainer.run_log_path,
                stats_scope="train",
                filter_errors=False,
            )
            # Run training
            await trainer.train(
                train_workflow=train_workflow,
                eval_workflow=None,
            )
        finally:
            trainer.model_info.model_name = old_model_name
            trainer.model_info.base_url = old_base_url
            trainer.model_info.api_key = old_api_key
            tinker_proxy.stop()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))