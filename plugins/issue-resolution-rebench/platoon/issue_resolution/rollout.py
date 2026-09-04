import asyncio
import json
import os
import shlex
import time
from pathlib import Path

# Modal 1.5.5 selects the V2 backend through this client-side setting. Set it
# in code so every agent-server Sandbox uses V2 even outside the README launch
# command.
os.environ["MODAL_SANDBOX_V2"] = "1"

from jinja2 import Environment, FileSystemLoader
from openhands.sdk import LLM, Agent, get_logger
from openhands.tools import get_default_tools
from openhands.workspace import ModalWorkspace
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.openhands.agent import OpenHandsAgent
from platoon.visualization.event_sinks import JsonlFileSink

from platoon.issue_resolution.env import SWERebenchEnv
from platoon.issue_resolution.tasks import (
    NUM_RETRIES_SANDBOX_START,
    SDK_SHORT_SHA,
    USER_PROMPT_FILENAME,
    named_agent_server_image_for_instance,
)

logger = get_logger(__name__)


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _optional_int_env(name: str) -> int | None:
    value = _optional_env(name)
    return int(value) if value is not None else None


def _modal_workspace(instance: dict, *, rollout_timeout: int = 1800) -> ModalWorkspace:
    """Create an isolated workspace from the instance's published Modal image."""
    return ModalWorkspace(
        named_server_image=named_agent_server_image_for_instance(instance),
        target_type="source",
        app_name=os.environ.get("MODAL_APP_NAME", "swerebench-agent-server"),
        modal_environment=_optional_env("MODAL_ENVIRONMENT"),
        working_dir="/testbed",
        # The sandbox also covers workspace setup and post-episode cleanup.
        timeout=int(os.environ.get("MODAL_SANDBOX_TIMEOUT", str(rollout_timeout + 300))),
        idle_timeout=_optional_int_env("MODAL_IDLE_TIMEOUT"),
        startup_timeout=float(os.environ.get("MODAL_STARTUP_TIMEOUT", "600")),
        cpu=float(os.environ.get("MODAL_CPU", "0.125")),
        memory=int(os.environ.get("MODAL_MEMORY", "288")),
        cloud=_optional_env("MODAL_CLOUD"),
        region=_optional_env("MODAL_REGION"),
        expected_server_git_sha=_optional_env("MODAL_EXPECTED_SERVER_GIT_SHA"),
        sandbox_tags={"purpose": "swerebench-issue-resolution"},
        verbose=os.environ.get("MODAL_VERBOSE", "0").lower() in {"1", "true", "yes"},
    )


def prepare_workspace(instance: dict, *, rollout_timeout: int = 1800) -> ModalWorkspace:
    """Start a Modal workspace and validate its prebuilt SWE-rebench testbed."""
    instance_id = str(instance["instance_id"])
    repo_path = "/testbed"
    quoted_repo_path = shlex.quote(repo_path)

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

            instance["repo_path"] = repo_path
            logger.info(
                "Modal workspace for %s is ready at %s using SDK image %s",
                instance_id,
                repo_path,
                SDK_SHORT_SHA,
            )
            return workspace
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


def get_instruction(instance: dict, prompt_path: str) -> str:
    prompts_dir = os.path.dirname(prompt_path)
    template = Environment(loader=FileSystemLoader(prompts_dir)).get_template(
        os.path.basename(prompt_path)
    )
    return template.render(instance=instance)


def prepare_llm(config: RolloutConfig) -> LLM:
    model_name = config.model_name
    if not model_name:
        raise ValueError("RolloutConfig.model_name must be set")
    if not model_name.startswith(("openai/", "litellm_proxy/")):
        model_name = "openai/" + model_name

    return LLM(
        usage_id="agent",
        model=model_name,
        num_retries=2,
        base_url=config.model_endpoint,
        api_key=config.model_api_key or "sk-xxx",
        temperature=config.inference_params.temperature,
        max_input_tokens=config.extra.get("max_input_tokens"),
        max_output_tokens=config.inference_params.max_completion_tokens,
        litellm_extra_body={
            "include_stop_str_in_output": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def prepare_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=None,
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    env: SWERebenchEnv | None = None
    workspace: ModalWorkspace | None = None
    cleanup_done = False
    rollout_start = time.perf_counter()
    rollout_timeout = config.timeout or 1800
    prepare_workspace_s: float | None = None
    prompt_build_s: float | None = None
    agent_init_s: float | None = None
    agent_loop_s: float | None = None
    cleanup_total_s = 0.0
    status = "started"
    error_detail: str | None = None
    events_path: str | None = None
    collection_id: str | None = None

    async def run_cleanup() -> None:
        nonlocal cleanup_done, cleanup_total_s
        if cleanup_done:
            return
        cleanup_done = True
        cleanup_start = time.perf_counter()
        try:
            if env is not None:
                await asyncio.wait_for(env.close(), timeout=180)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as error:
            logger.warning("Environment cleanup failed for task %s: %s", task.id, error)
        finally:
            if workspace is not None:
                try:
                    await asyncio.to_thread(workspace.cleanup)
                except (asyncio.CancelledError, Exception) as error:
                    logger.warning("Modal cleanup failed for task %s: %s", task.id, error)
            cleanup_total_s += time.perf_counter() - cleanup_start

    try:
        if config.verbose:
            print(
                f"[run_rollout] Process {os.getpid()}: Starting rollout for task {task.id}",
                flush=True,
            )
        instance: dict = task.misc
        prepare_workspace_start = time.perf_counter()
        try:
            workspace = await asyncio.to_thread(prepare_workspace, instance, rollout_timeout=rollout_timeout)
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
        except Exception as error:
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
            status = "workspace_setup_failed"
            error_detail = str(error)
            raise RuntimeError(f"Workspace setup failed for task {task.id}: {error}") from error

        prompt_path = (Path(__file__).parent / "prompts" / USER_PROMPT_FILENAME).resolve()
        if not prompt_path.exists():
            raise FileNotFoundError(f"User prompt path {prompt_path} not found")
        prompt_build_start = time.perf_counter()
        task.goal = get_instruction(instance, str(prompt_path))
        prompt_build_s = time.perf_counter() - prompt_build_start
        task.max_steps = config.max_steps if config.max_steps is not None else 100

        agent_init_start = time.perf_counter()
        agent = prepare_agent(prepare_llm(config))
        agent_wrapper = OpenHandsAgent()
        env = SWERebenchEnv(
            task=task,
            agent=agent,
            workspace=workspace,
            conversation_timeout=rollout_timeout,
            enable_length_penalty=config.extra.get("enable_length_penalty", False),
            length_penalty_threshold=config.extra.get("length_penalty_threshold", 10),
        )
        agent_init_s = time.perf_counter() - agent_init_start

        trajectory_collection = TrajectoryCollection()
        current_trajectory_collection.set(trajectory_collection)
        collection_id = trajectory_collection.id
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

        episode_task = asyncio.create_task(
            run_episode(agent_wrapper, env, timeout=config.step_timeout)
        )
        agent_loop_start = time.perf_counter()
        try:
            await asyncio.wait_for(episode_task, timeout=rollout_timeout)
            agent_loop_s = time.perf_counter() - agent_loop_start
        except asyncio.TimeoutError:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "timeout"
            error_detail = f"Rollout timed out after {rollout_timeout} seconds"
            raise
        except Exception as error:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "run_episode_failed"
            error_detail = str(error)
            raise

        # Match issue-resolution-legacy: do not discard completed trajectories
        # based on finish/error/budget status. The workflow also uses
        # filter_errors=False so every emitted agent token remains trainable.
        status = "success"
        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        return current_trajectory_collection.get()
    except Exception as error:
        if status == "started":
            status = "error"
            error_detail = str(error)
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {error}", flush=True)
        await run_cleanup()
        raise
    finally:
        await run_cleanup()
        print(
            "ROLLOUT_TIMING "
            + json.dumps(
                {
                    "task_id": task.id,
                    "collection_id": collection_id,
                    "pid": os.getpid(),
                    "status": status,
                    "error": error_detail,
                    "prepare_workspace_s": prepare_workspace_s,
                    "prompt_build_s": prompt_build_s,
                    "agent_init_s": agent_init_s,
                    "agent_loop_s": agent_loop_s,
                    "cleanup_total_s": cleanup_total_s,
                    "total_rollout_s": time.perf_counter() - rollout_start,
                    "events_path": events_path,
                },
                sort_keys=True,
            ),
            flush=True,
        )
