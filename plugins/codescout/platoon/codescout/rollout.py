"""CodeScout rollouts backed by OpenHands running in Modal Sandboxes."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid
from pathlib import Path

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

from platoon.codescout.custom_tools.localization_finish import LocalizationFinishTool  # noqa: F401
from platoon.codescout.env import CodeScoutEnv
from platoon.codescout.tasks import (
    EVAL_NAMED_AGENT_SERVER_IMAGE,
    NUM_RETRIES_SANDBOX_START,
    USER_PROMPT_FILENAME,
)

logging.getLogger("openhands.sdk.conversation.impl.remote_conversation").setLevel(logging.CRITICAL)
logger = get_logger(__name__)


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _optional_int_env(name: str) -> int | None:
    value = _optional_env(name)
    return int(value) if value is not None else None


def _get_active_tinker_proxy():
    """Return the in-process Tinker proxy without requiring Tinker for Areal."""
    try:
        from platoon.train.tinker.fastapi_litellm_proxy import (
            get_active_tinker_http_proxy,
        )
    except ImportError:
        return None
    return get_active_tinker_http_proxy()


def _modal_workspace(working_dir: str) -> ModalWorkspace:
    """Create one isolated Modal Sandbox for a CodeScout rollout."""
    return ModalWorkspace(
        named_server_image=os.environ.get(
            "CODESCOUT_NAMED_AGENT_SERVER_IMAGE",
            EVAL_NAMED_AGENT_SERVER_IMAGE,
        ),
        target_type="source",
        app_name=os.environ.get("MODAL_APP_NAME", "codescout-agent-server"),
        modal_environment=_optional_env("MODAL_ENVIRONMENT"),
        working_dir=working_dir,
        timeout=int(os.environ.get("MODAL_SANDBOX_TIMEOUT", "800")),
        idle_timeout=_optional_int_env("MODAL_IDLE_TIMEOUT"),
        startup_timeout=float(os.environ.get("MODAL_STARTUP_TIMEOUT", "300")),
        cpu=float(os.environ.get("MODAL_CPU", "0.5")),
        memory=int(os.environ.get("MODAL_MEMORY", "256")),
        cloud=_optional_env("MODAL_CLOUD"),
        region=_optional_env("MODAL_REGION"),
        expected_server_git_sha=_optional_env("MODAL_EXPECTED_SERVER_GIT_SHA"),
        sandbox_tags={"purpose": "codescout-localization"},
        verbose=os.environ.get("MODAL_VERBOSE", "0").lower() in {"1", "true", "yes"},
    )


def prepare_workspace(instance: dict) -> tuple[Path, ModalWorkspace]:
    """Start a Modal Sandbox and clone the task repository into it."""
    instance_id = str(instance["instance_id"])
    repo_name = str(instance["repo"])
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = Path("/workspace") / instance_dir_name

    repo_url = f"https://github.com/{repo_name}.git"
    clone_command = (
        "git clone --depth 1 --single-branch --branch "
        f"{shlex.quote(instance_id)} {shlex.quote(repo_url)} "
        f"{shlex.quote(str(instance_path))}"
    )

    last_error: Exception | None = None
    for attempt in range(1, NUM_RETRIES_SANDBOX_START + 1):
        workspace: ModalWorkspace | None = None
        try:
            workspace = _modal_workspace(str(instance_path))
            result = workspace.execute_command(
                clone_command,
                cwd="/workspace",
                timeout=300,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Repository clone failed (exit {result.exit_code}): {result.stderr}"
                )
            return instance_path, workspace
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


def get_instruction(instance: dict, prompt_path: str, workspace_path: str) -> str:
    """Render the CodeScout localization instruction."""
    prompts_dir = os.path.dirname(prompt_path)
    template = Environment(loader=FileSystemLoader(prompts_dir)).get_template(
        os.path.basename(prompt_path)
    )
    return template.render(instance=instance, working_dir=workspace_path)


def prepare_llm(config: RolloutConfig, tinker_proxy_session_id: str | None = None) -> LLM:
    """Build an OpenHands LLM configured for either Platoon backend proxy."""
    model_name = config.model_name
    if not model_name:
        raise ValueError("RolloutConfig.model_name must be set")
    if not model_name.startswith(("openai/", "litellm_proxy/")):
        model_name = f"openai/{model_name}"

    active_proxy = _get_active_tinker_proxy()
    extra_headers = None
    if tinker_proxy_session_id is not None:
        extra_headers = {"X-Platoon-Tinker-Session": tinker_proxy_session_id}

    return LLM(
        usage_id="agent",
        model=model_name,
        base_url=config.model_endpoint,
        api_key=config.model_api_key or "sk-xxx",
        temperature=config.inference_params.temperature,
        max_input_tokens=(active_proxy.context_window_length if active_proxy is not None else None),
        max_output_tokens=config.inference_params.max_completion_tokens,
        extra_headers=extra_headers,
        litellm_extra_body={
            "include_stop_str_in_output": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run one CodeScout trajectory and destroy its Modal Sandbox afterward."""
    env: CodeScoutEnv | None = None
    workspace: ModalWorkspace | None = None
    tinker_proxy = _get_active_tinker_proxy()
    tinker_session_id = str(uuid.uuid4()) if tinker_proxy is not None else None

    try:
        if config.verbose:
            print(f"[run_rollout] Starting Modal rollout for task {task.id}", flush=True)

        working_dir, workspace = await asyncio.to_thread(prepare_workspace, task.misc)

        prompt_path = (Path(__file__).parent / "prompts" / USER_PROMPT_FILENAME).resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"User prompt {prompt_path} not found")
        task.goal = get_instruction(task.misc, str(prompt_path), str(working_dir))
        task.max_steps = config.max_steps if config.max_steps is not None else 6

        agent = Agent(
            llm=prepare_llm(config, tinker_session_id),
            tools=[Tool(name=TerminalTool.name), Tool(name="LocalizationFinishTool")],
            system_prompt_filename="/app/prompts_codescout/system_prompt.j2",
            include_default_tools=[],
        )
        env = CodeScoutEnv(task=task, agent=agent, workspace=workspace)
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

        rollout_timeout = config.timeout or 600
        await asyncio.wait_for(
            run_episode(agent_wrapper, env, timeout=rollout_timeout),
            timeout=rollout_timeout + 30,
        )

        result = current_trajectory_collection.get()
        if config.return_dict:
            result_dict = result.to_dict()
            if tinker_proxy is not None and tinker_session_id is not None:
                result_dict["_tinker_interactions"] = tinker_proxy.pop_interactions(
                    tinker_session_id
                )
            return result_dict
        return result
    except Exception:
        if tinker_proxy is not None and tinker_session_id is not None:
            tinker_proxy.discard_session(tinker_session_id)
        raise
    finally:
        if env is not None:
            try:
                await asyncio.wait_for(env.close(), timeout=180)
            except Exception as cleanup_error:
                logger.warning("Modal cleanup failed for task %s: %s", task.id, cleanup_error)
        elif workspace is not None:
            await asyncio.to_thread(workspace.cleanup)
