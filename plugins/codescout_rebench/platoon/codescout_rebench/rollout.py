"""CodeScout rollouts on prebuilt SWE-rebench Modal images."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import shlex
from pathlib import Path

# Keep the image backend consistent with the SWE-rebench image/evaluator path.
os.environ["MODAL_SANDBOX_V2"] = "1"

from jinja2 import Environment, FileSystemLoader
from openhands.sdk import LLM, Agent, Tool, get_logger
from openhands.tools.terminal import TerminalTool
from openhands.workspace import ModalWorkspace
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.openhands.agent import OpenHandsAgent
from platoon.visualization.event_sinks import JsonlFileSink

from platoon.codescout_rebench import custom_tools as codescout_custom_tools
from platoon.codescout_rebench.custom_tools.localization_finish import (  # noqa: F401
    LocalizationFinishTool,
)
from platoon.codescout_rebench.env import CodeScoutEnv
from platoon.codescout_rebench.runtime_assets import (
    hydrate_custom_tool,
    install_ripgrep,
    read_system_prompt,
)
from platoon.codescout_rebench.tasks import (
    NUM_RETRIES_SANDBOX_START,
    SDK_GIT_SHA,
    SYSTEM_PROMPT_FILENAME,
    USER_PROMPT_FILENAME,
    named_agent_server_image_for_instance,
)

logging.getLogger("openhands.sdk.conversation.impl.remote_conversation").setLevel(logging.CRITICAL)
logger = get_logger(__name__)


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _optional_int_env(name: str) -> int | None:
    value = _optional_env(name)
    return int(value) if value is not None else None


def _modal_workspace(
    instance: dict,
    *,
    rollout_timeout: int = 600,
) -> ModalWorkspace:
    """Start a Sandbox from the existing named image for this instance."""
    expected_sha = os.environ.get("MODAL_EXPECTED_SERVER_GIT_SHA", SDK_GIT_SHA)
    return ModalWorkspace(
        named_server_image=named_agent_server_image_for_instance(instance),
        target_type="source",
        app_name=os.environ.get(
            "CODESCOUT_REBENCH_MODAL_APP_NAME",
            "codescout-rebench-agent-server",
        ),
        modal_environment=_optional_env("MODAL_ENVIRONMENT"),
        working_dir="/testbed",
        timeout=int(os.environ.get("MODAL_SANDBOX_TIMEOUT", str(rollout_timeout + 300))),
        idle_timeout=_optional_int_env("MODAL_IDLE_TIMEOUT"),
        startup_timeout=float(os.environ.get("MODAL_STARTUP_TIMEOUT", "600")),
        cpu=float(os.environ.get("MODAL_CPU", "0.125")),
        memory=int(os.environ.get("MODAL_MEMORY", "288")),
        cloud=_optional_env("MODAL_CLOUD"),
        region=_optional_env("MODAL_REGION"),
        expected_server_git_sha=expected_sha or None,
        sandbox_tags={"purpose": "codescout-swerebench-localization"},
        verbose=os.environ.get("MODAL_VERBOSE", "0").lower() in {"1", "true", "yes"},
    )


def _custom_tool_package_path() -> Path:
    package_path = getattr(codescout_custom_tools, "__path__", None)
    if not package_path:
        raise RuntimeError("Could not locate the installed CodeScout custom tool package")
    return Path(next(iter(package_path))).resolve()


def prepare_workspace(
    instance: dict,
    *,
    rollout_timeout: int = 600,
) -> tuple[Path, ModalWorkspace]:
    """Start, reset, and hydrate a prebuilt SWE-rebench workspace."""
    instance_id = str(instance["instance_id"])
    repo_path = Path("/testbed")
    quoted_repo_path = shlex.quote(str(repo_path))

    last_error: Exception | None = None
    for attempt in range(1, NUM_RETRIES_SANDBOX_START + 1):
        workspace: ModalWorkspace | None = None
        try:
            workspace = _modal_workspace(instance, rollout_timeout=rollout_timeout)
            clean = workspace.execute_command(
                f"cd {quoted_repo_path} && "
                "git config --global --add safe.directory /testbed && "
                "git reset --hard HEAD && git clean -fd",
                timeout=300,
            )
            if clean.exit_code != 0:
                raise RuntimeError(f"Testbed cleanup failed: {clean.stderr}")

            hydrate_custom_tool(workspace, _custom_tool_package_path())
            install_ripgrep(workspace)
            instance["repo_path"] = str(repo_path)
            return repo_path, workspace
        except Exception as error:
            last_error = error
            if workspace is not None:
                workspace.cleanup()
            if attempt < NUM_RETRIES_SANDBOX_START:
                logger.warning(
                    "Modal workspace setup attempt %s/%s failed for %s: %s",
                    attempt,
                    NUM_RETRIES_SANDBOX_START,
                    instance_id,
                    error,
                )

    raise RuntimeError(f"Error preparing Modal workspace for instance {instance_id}: {last_error}")


def get_instruction(instance: dict, prompt_path: Path, workspace_path: str) -> str:
    """Render the host-side CodeScout user prompt."""
    template = Environment(loader=FileSystemLoader(prompt_path.parent)).get_template(
        prompt_path.name
    )
    return template.render(instance=instance, working_dir=workspace_path)


def prepare_llm(config: RolloutConfig) -> LLM:
    """Use the routed AReaL endpoint and its session-scoped bearer key."""
    model_name = config.model_name
    if not model_name:
        raise ValueError("RolloutConfig.model_name must be set")
    if not model_name.startswith(("openai/", "litellm_proxy/")):
        model_name = f"openai/{model_name}"

    return LLM(
        usage_id="agent",
        model=model_name,
        num_retries=2,
        base_url=config.model_endpoint,
        api_key=config.model_api_key or "sk-xxx",
        temperature=config.inference_params.temperature,
        # Preserve independent sampling across trajectories in a task group.
        seed=secrets.randbits(31),
        max_input_tokens=config.extra.get("max_input_tokens"),
        max_output_tokens=config.inference_params.max_completion_tokens,
        litellm_extra_body={
            "include_stop_str_in_output": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run one CodeScout trajectory and destroy its Modal Sandbox afterward."""
    env: CodeScoutEnv | None = None
    workspace: ModalWorkspace | None = None
    rollout_timeout = config.timeout or 600

    try:
        if config.verbose:
            print(
                f"[run_rollout] Starting SWE-rebench CodeScout task {task.id}",
                flush=True,
            )

        working_dir, workspace = await asyncio.to_thread(
            prepare_workspace,
            task.misc,
            rollout_timeout=rollout_timeout,
        )

        prompts_dir = Path(__file__).parent / "prompts"
        user_prompt_path = (prompts_dir / USER_PROMPT_FILENAME).resolve()
        system_prompt_path = (prompts_dir / SYSTEM_PROMPT_FILENAME).resolve()
        if not user_prompt_path.is_file():
            raise FileNotFoundError(f"User prompt not found: {user_prompt_path}")
        task.goal = get_instruction(task.misc, user_prompt_path, str(working_dir))
        task.max_steps = config.max_steps if config.max_steps is not None else 10

        agent = Agent(
            llm=prepare_llm(config),
            tools=[Tool(name=TerminalTool.name), Tool(name="LocalizationFinishTool")],
            # SDK 5acdf05 serializes this text with the Agent, so it does not
            # need to exist in the per-instance image.
            system_prompt=read_system_prompt(system_prompt_path),
            include_default_tools=[],
            condenser=None,
        )
        env = CodeScoutEnv(
            task=task,
            agent=agent,
            workspace=workspace,
            conversation_timeout=rollout_timeout,
        )
        agent_wrapper = OpenHandsAgent()

        trajectory_collection = TrajectoryCollection()
        current_trajectory_collection.set(trajectory_collection)
        events_path = os.path.join(
            config.output_dir,
            "events",
            f"events_{task.id}_{trajectory_collection.id}.jsonl",
        )
        trajectory_collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=trajectory_collection.id,
                process_id=os.getpid(),
            )
        )

        await asyncio.wait_for(
            # The SDK enforces max_iteration_per_run. Platoon's step tracker
            # also counts the initial observation, so enforcing it here would
            # cut the model off early and can mark a successful finish as an error.
            run_episode(agent_wrapper, env, timeout=config.step_timeout, enforce_step_budget=False),
            timeout=rollout_timeout + 30,
        )

        result = current_trajectory_collection.get()
        if config.return_dict:
            return result.to_dict()
        return result
    finally:
        if env is not None:
            try:
                await asyncio.wait_for(env.close(), timeout=180)
            except Exception as cleanup_error:
                logger.warning("Modal cleanup failed for task %s: %s", task.id, cleanup_error)
        if workspace is not None:
            try:
                await asyncio.to_thread(workspace.cleanup)
            except Exception as cleanup_error:
                logger.warning("Workspace cleanup failed for task %s: %s", task.id, cleanup_error)
