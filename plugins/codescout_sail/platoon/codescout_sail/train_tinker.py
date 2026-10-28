# ruff: noqa: E402
"""Codescout training script with Tinker backend."""

import asyncio
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field


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
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

# from datasets.utils.logging import disable_progress_bars
from platoon.codescout_sail.rollout import run_rollout
from platoon.codescout_sail.tasks import get_task, load_data
from platoon.codescout_sail.tinker_workflow import TinkerProxyRollout
from platoon.openhands.apptainer_namespace import shared_apptainer_namespace
from platoon.inference.inference_tunnel import InferenceTunnelConfig, inference_tunnels, wait_for_health
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.fastapi_litellm_proxy import FastAPILiteLLMTinkerHTTPProxyServer

# disable_progress_bars()


@dataclass
class CodeScoutTinkerConfig(PlatoonTinkerRLTrainerConfig):
    inference_tunnel: InferenceTunnelConfig = field(
        default_factory=lambda: InferenceTunnelConfig(backend="modal")
    )


async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "train_codescout_fft_tinker.yaml"
    config, _ = load_config(
        args=args,
        config_class=CodeScoutTinkerConfig,
        default_config_path=str(default_config),
    )
    workspace = config.train.workflow_config.rollout_config.extra.get("workspace", {})
    with shared_apptainer_namespace(enabled=workspace.get("type") == "apptainer"):
        await train(config)


async def train(config):
    train_datamap, _ = load_data()
    train_dataset = Dataset.from_list([{"task_id": x} for x in train_datamap.keys()])
    # Create trainer and run with context manager for proper cleanup
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
            host=os.environ.get("CODESCOUT_TINKER_PROXY_HOST", "127.0.0.1"),
        )
        rollout_adapter = TinkerProxyRollout(run_rollout, tinker_proxy)
        try:
            tinker_proxy.start()
            origin = tinker_proxy.base_url.removesuffix("/v1")
            wait_for_health(origin)
            with inference_tunnels([origin], config=config.inference_tunnel) as endpoints:
                trainer.model_info.model_name = tinker_proxy.model_name
                trainer.model_info.base_url = endpoints[origin] + "/v1"
                trainer.model_info.api_key = tinker_proxy.api_key
                # Create workflows - use trainer.run_log_path for run-specific output
                train_workflow = GroupRolloutWorkflow(
                    rollout_fn=rollout_adapter,
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
            try:
                await rollout_adapter.aclose()
            finally:
                trainer.model_info.model_name = old_model_name
                trainer.model_info.base_url = old_base_url
                trainer.model_info.api_key = old_api_key
                tinker_proxy.stop()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
