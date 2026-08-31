import asyncio
import json
import os
import shlex
import time
from pathlib import Path

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

from platoon.issue_resolution.env import SWEBenchEnv
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


def _modal_workspace(instance: dict) -> ModalWorkspace:
    """Create one isolated Modal Sandbox from the instance's published image."""
    return ModalWorkspace(
        named_server_image=named_agent_server_image_for_instance(instance),
        target_type="source",
        app_name=os.environ.get("MODAL_APP_NAME", "swesmith-agent-server"),
        modal_environment=_optional_env("MODAL_ENVIRONMENT"),
        working_dir="/workspace",
        # Leave headroom around the legacy 1,230-second outer rollout timeout
        # for Sandbox startup, testbed setup, and cleanup.
        timeout=int(os.environ.get("MODAL_SANDBOX_TIMEOUT", "1400")),
        idle_timeout=_optional_int_env("MODAL_IDLE_TIMEOUT"),
        startup_timeout=float(os.environ.get("MODAL_STARTUP_TIMEOUT", "600")),
        cpu=float(os.environ.get("MODAL_CPU", "0.125")),
        memory=int(os.environ.get("MODAL_MEMORY", "288")),
        cloud=_optional_env("MODAL_CLOUD"),
        region=_optional_env("MODAL_REGION"),
        expected_server_git_sha=_optional_env("MODAL_EXPECTED_SERVER_GIT_SHA"),
        sandbox_tags={"purpose": "swesmith-issue-resolution"},
        verbose=os.environ.get("MODAL_VERBOSE", "0").lower() in {"1", "true", "yes"},
    )


def prepare_workspace(instance: dict) -> ModalWorkspace:
    """Start a Modal workspace and copy its prebuilt SWE-Smith testbed."""
    instance_id = str(instance["instance_id"])
    repo_name = str(instance["repo"]).rsplit("/", 1)[-1]
    repo_path = f"/workspace/{repo_name}/"
    quoted_repo_path = shlex.quote(repo_path)
    quoted_instance_id = shlex.quote(instance_id)

    last_error: Exception | None = None
    for attempt in range(1, NUM_RETRIES_SANDBOX_START + 1):
        workspace: ModalWorkspace | None = None
        try:
            workspace = _modal_workspace(instance)
            setup = workspace.execute_command(
                f"mkdir -p {quoted_repo_path} ; cp -r /testbed/. {quoted_repo_path}",
                timeout=900,
            )
            if setup.exit_code != 0:
                raise RuntimeError(f"Testbed copy failed: {setup.stderr}")

            fetch = workspace.execute_command(f"cd {quoted_repo_path} && git fetch", timeout=300)
            if fetch.exit_code != 0:
                raise RuntimeError(f"git fetch failed: {fetch.stderr}")
            checkout = workspace.execute_command(
                f"cd {quoted_repo_path} && git checkout {quoted_instance_id}",
                timeout=300,
            )
            if checkout.exit_code != 0:
                raise RuntimeError(f"git checkout failed: {checkout.stderr}")

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


def get_instruction(
    instance: dict,
    prompt_path: str,
) -> str:
    """Generate user instruction for the agent for SWE-Bench-style tasks."""
    # Set up Jinja2 environment
    # NOTE: Jinja template will not work for SWE-Smith as its base commit is None
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
    }

    # Render the instruction
    instruction = template.render(context)
    return instruction


def prepare_llm(config: RolloutConfig) -> LLM:
    model_name = config.model_name
    if not model_name:
        raise ValueError("RolloutConfig.model_name must be set")

    temperature = config.inference_params.temperature
    if not model_name.startswith("openai/") and not model_name.startswith("litellm_proxy/"):
        model_name = "openai/" + model_name

    llm = LLM(
        usage_id="agent",
        model=model_name,
        num_retries=2,
        base_url=config.model_endpoint,
        api_key=config.model_api_key or "sk-xxx",
        temperature=temperature,
        max_input_tokens=config.extra.get("max_input_tokens"),
        max_output_tokens=config.inference_params.max_completion_tokens,
        litellm_extra_body={
            "include_stop_str_in_output": False,
            "chat_template_kwargs": {
                # "add_generation_prompt": True, #NOTE: setting this to true raises errors
                "enable_thinking": False
            },
        },
    )
    return llm


def prepare_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=None,
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    env: SWEBenchEnv | None = None
    workspace: ModalWorkspace | None = None
    cleanup_done = False
    rollout_start = time.perf_counter()
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
            # env.close() normally cleans up the workspace. Always make the
            # idempotent Modal cleanup call as a fallback, even if env.close()
            # times out or is cancelled.
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
        try:
            prepare_workspace_start = time.perf_counter()
            workspace = await asyncio.to_thread(prepare_workspace, instance)
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
        except Exception as e:
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
            status = "workspace_setup_failed"
            error_detail = str(e)
            print(f"[run_rollout] Workspace setup failed for task {task.id}: {e}", flush=True)
            raise RuntimeError(f"Workspace setup failed for task {task.id}: {e}")
        user_prompt_filename = USER_PROMPT_FILENAME
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        user_prompt_path = prompt_dir / user_prompt_filename
        assert user_prompt_path.exists(), f"User prompt path {user_prompt_path} not found"
        prompt_build_start = time.perf_counter()
        input_message = get_instruction(instance, str(user_prompt_path))
        prompt_build_s = time.perf_counter() - prompt_build_start

        task.goal = input_message
        task.max_steps = config.max_steps if config.max_steps is not None else 100

        agent_init_start = time.perf_counter()
        llm: LLM = prepare_llm(config)
        agent: Agent = prepare_agent(llm)
        agent_wrapper_platoon: OpenHandsAgent = OpenHandsAgent()
        env: SWEBenchEnv = SWEBenchEnv(task=task, agent=agent, workspace=workspace)
        agent_init_s = time.perf_counter() - agent_init_start

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        collection_id = traj_collection.id

        events_path = os.path.join(
            config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl"
        )

        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        rollout_timeout = config.timeout or 1230
        rollout_task = asyncio.create_task(
            run_episode(agent_wrapper_platoon, env, timeout=config.step_timeout)
        )
        try:
            agent_loop_start = time.perf_counter()
            await asyncio.wait_for(rollout_task, timeout=rollout_timeout)
            agent_loop_s = time.perf_counter() - agent_loop_start
        except asyncio.TimeoutError:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "timeout"
            error_detail = f"Rollout timed out after {rollout_timeout} seconds"
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout timed out for task {task.id}", flush=True)
            raise
        except Exception as e:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "run_episode_failed"
            error_detail = str(e)
            if config.verbose:
                print(
                    f"Process {os.getpid()}: Rollout failed for task {task.id}: {str(e)}",
                    flush=True,
                )
            raise

        # The legacy 257-instance easy run intentionally did not mask completed
        # trajectories based on finish/error/budget status. Keep them trainable;
        # only an exception or hard timeout prevents a rollout from returning.
        status = "success"
        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        else:
            return current_trajectory_collection.get()
    except Exception as e:
        if status == "started":
            status = "error"
            error_detail = str(e)
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}", flush=True)
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
