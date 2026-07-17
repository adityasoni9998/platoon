import os
import logging

logging.getLogger(
    "openhands.sdk.conversation.impl.remote_conversation"
).setLevel(logging.CRITICAL)

import subprocess
import json
from jinja2 import Environment, FileSystemLoader
import asyncio
import time
from platoon.envs.base import Task
from platoon.issue_resolution.env import SWEBenchEnv
from pathlib import Path
from openhands.sdk import LLM, get_logger, Agent, Tool, AgentBase
from openhands.tools import get_default_tools
from openhands.workspace import ApptainerWorkspace
from platoon.episode.trajectory import TrajectoryCollection
from platoon.config_defs import RolloutConfig
from openhands.sdk.workspace import BaseWorkspace
from platoon.episode.loop import run_episode
from platoon.episode.context import current_trajectory_collection, finish_message, error_message
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.issue_resolution.tasks import USER_PROMPT_FILENAME, APPTAINER_CACHEDIR, ENV_SETUP_COMMANDS
from platoon.openhands.agent import OpenHandsAgent
import platform
import uuid
from openhands.tools.terminal import TerminalTool
from platoon.train.tinker.fastapi_litellm_proxy import SESSION_HEADER, get_active_tinker_http_proxy
import shutil
logger = get_logger(__name__)

_orig_start_container = ApptainerWorkspace._start_container


def _patched_start_container(self) -> None:
    """Drop apptainer child logs entirely when detach_logs is disabled."""
    overlay_root_dir = self.forward_env[-1]
    self.forward_env = self.forward_env[:-1]
    if self.detach_logs:
        return _orig_start_container(self)

    env_args: list[str] = []
    for key in self.forward_env:
        if key in os.environ:
            env_args += ["--env", f"{key}={os.environ[key]}"]

    bind_args: list[str] = []
    if self.mount_dir:
        mount_path = "/workspace"
        bind_args += ["--bind", f"{self.mount_dir}:{mount_path}"]
        logger.info(
            "Mounting host dir %s to container path %s",
            self.mount_dir,
            mount_path,
        )
    env_extra_binds = [
        item.strip()
        for item in os.getenv("OPENHANDS_APPTAINER_EXTRA_BINDS", "").split(",")
        if item.strip()
    ]
    for bind_spec in [*self.extra_bind_mounts, *env_extra_binds]:
        bind_args += ["--bind", bind_spec]
        logger.info("Adding Apptainer bind mount: %s", bind_spec)

    container_opts: list[str] = []
    if self.use_fakeroot:
        container_opts.append("--fakeroot")
    if self.enable_docker_compat:
        # container_opts.append("--compat")
        container_opts.append("--containall")
        container_opts.append("--no-eval")
        container_opts.append("--no-init")
        container_opts.append("--no-umask")
        container_opts += ["--overlay", str(overlay_root_dir)]
    if self.enable_gpu:
        container_opts.append("--nv")
    if self.disable_mount_locations:
        for loc in self.disable_mount_locations:
            container_opts += ["--no-mount", loc]

    server_cmd = [
        "apptainer",
        "run",
        *container_opts,
        *env_args,
        *bind_args,
        self._sif_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(self.host_port),
    ]

    self._process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


ApptainerWorkspace._start_container = _patched_start_container  # type: ignore[method-assign]

def detect_platform():
    """Detects the correct platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

def prepare_workspace(instance: dict, overlay_root_dir: str) -> BaseWorkspace:
    forward_env_vars = ["DEBUG", overlay_root_dir]
    workspace_kwargs = {
        "working_dir": "/testbed",
        "platform": detect_platform(),
        "cache_dir": os.environ.get("APPTAINER_CACHEDIR", APPTAINER_CACHEDIR),
        "detach_logs": False, #NOTE: Keep this False to use the patched _start_container method that suppresses logs and uses overlays instead of writable-tmpfs
        "health_check_timeout": 600,
        "forward_env": forward_env_vars,
    }
    image_name = instance["image_name"].split("/")[-1]
    sif_path = f"{APPTAINER_CACHEDIR}/43376f1-93c33d0-{image_name}-source-minimal.sif" #TODO: fix this
    if not os.path.exists(sif_path):
        raise FileNotFoundError(f"Apptainer image not found at {sif_path}. Please ensure the image is built and available.")
    workspace_kwargs["sif_file"] = sif_path
    NUM_RETRIES = 3
    workspace = None
    for i in range(NUM_RETRIES):
        try:
            workspace = ApptainerWorkspace(**workspace_kwargs)
            break
        except Exception as e:
            if i == NUM_RETRIES - 1:
                raise RuntimeError(f"Error preparing workspace for instance {instance['instance_id']}: {str(e)}")
            try:
                workspace.cleanup()
            except:
                pass
            logger.warning(f"Workspace setup attempt {i + 1} failed, retrying...")
    for cmd in ENV_SETUP_COMMANDS:
        res = workspace.execute_command(cmd)
        if res.exit_code != 0:
            raise RuntimeError(
                f"Failed to run env setup command '{cmd}': {res.stderr}"
            )
        logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
    repo_path = f"/testbed"
    # logger.info(f"Repo path in Remote workspace: {repo_path}")
    instance["repo_path"] = repo_path

    commit_id = instance["instance_id"]
    git_fetch = workspace.execute_command(f"cd {repo_path} && git fetch", timeout=300)
    assert git_fetch.exit_code == 0, f"git fetch failed: {git_fetch.stderr}"

    checkout_commit = workspace.execute_command(f"cd {repo_path} && git checkout {commit_id}", timeout=300)
    assert checkout_commit.exit_code == 0, f"git checkout failed: {checkout_commit.stderr}"
    
    # # NOTE: clean any uncommited tracked/untracked changes in repo so that they do not seep into our model patch and cause apply patch errors later
    # workspace.execute_command(f"cd {repo_path} && git reset --hard")
    # workspace.execute_command(f"cd {repo_path} && git clean -fd")
    return workspace

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

def prepare_llm(config: RolloutConfig, tinker_proxy_session_id: str | None = None) -> LLM:
    model_name = config.model_name
    temperature = config.inference_params.temperature
    if not model_name.startswith("openai/") and not model_name.startswith("litellm_proxy/"):
        model_name = "openai/" + model_name

    active_proxy = get_active_tinker_http_proxy()
    extra_headers = None
    if tinker_proxy_session_id is not None:
        extra_headers = {SESSION_HEADER: tinker_proxy_session_id}

    llm=LLM(
            usage_id="agent",
            model=model_name,
            num_retries=2,
            base_url=config.model_endpoint,
            api_key=config.model_api_key or "sk-xxx",
            temperature=temperature,
            max_input_tokens=active_proxy.context_window_length if active_proxy is not None else None,
            max_output_tokens=config.inference_params.max_completion_tokens,
            extra_headers=extra_headers,
            litellm_extra_body={
                "include_stop_str_in_output": False,
                "chat_template_kwargs": {
                    # "add_generation_prompt": True, #NOTE: setting this to true raises errors
                    "enable_thinking": False
                }
            }
        )
    return llm

def prepare_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=None,
    )

async def cleanup_resources(agent, env):
    if env is not None:
        await env.close()
        env = None

async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = agent_wrapper_platoon = None
    tinker_proxy = get_active_tinker_http_proxy()
    rollout_id = str(uuid.uuid4())
    overlay_root_dir = Path(f"/tmp/rollout_overlay_{rollout_id}")
    (overlay_root_dir / "upper").mkdir(parents=True, exist_ok=True)
    (overlay_root_dir / "work").mkdir(parents=True, exist_ok=True)
    overlay_root_dir = str(overlay_root_dir)

    tinker_proxy_session_id = str(uuid.uuid4()) if tinker_proxy is not None else None
    rollout_start = time.perf_counter()
    prepare_workspace_s: float | None = None
    agent_loop_s: float | None = None
    cleanup_total_s = 0.0
    status = "started"
    error_detail: str | None = None
    events_path: str | None = None
    collection_id: str | None = None

    async def run_cleanup() -> None:
        nonlocal cleanup_total_s
        cleanup_start = time.perf_counter()
        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        finally:
            cleanup_total_s += time.perf_counter() - cleanup_start

    try:
        # if config.verbose:
        #     print(f"[run_rollout] Process {os.getpid()}: Starting rollout for task {task.id}", flush=True)
        instance: dict = task.misc
        try:
            loop = asyncio.get_event_loop()
            # Run in a separate thread to avoid blocking the event loop.
            prepare_workspace_start = time.perf_counter()
            workspace = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                prepare_workspace,
                instance,
                overlay_root_dir
            )
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
        except Exception as e:
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
            status = "workspace_setup_failed"
            error_detail = str(e)
            # print(f"[run_rollout] Workspace setup failed for task {task.id}: {e}", flush=True)
            raise RuntimeError(
                f"Workspace setup failed for task {task.id}: {e}"
            )        
        user_prompt_filename = USER_PROMPT_FILENAME
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        user_prompt_path = prompt_dir / user_prompt_filename
        assert user_prompt_path.exists(), f"User prompt path {user_prompt_path} not found"
        input_message = get_instruction(instance, str(user_prompt_path))

        task.goal = input_message
        task.max_steps = config.max_steps if config.max_steps is not None else 100

        llm: LLM = prepare_llm(config, tinker_proxy_session_id=tinker_proxy_session_id)
        agent: Agent = prepare_agent(llm)
        agent_wrapper_platoon: OpenHandsAgent = OpenHandsAgent()
        env: SWEBenchEnv = SWEBenchEnv(task=task, agent=agent, workspace=workspace)

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        collection_id = traj_collection.id

        events_path = os.path.join(
            config.output_dir,
            "events",
            f"events_{task.id}_{traj_collection.id}.jsonl"
        )

        traj_collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=traj_collection.id,
                process_id=os.getpid()
            )
        )

        rollout_task = asyncio.create_task(run_episode(agent_wrapper_platoon, env, timeout=600))
        try:
            # Apply a hard timeout to the entire rollout, not just individual steps
            agent_loop_start = time.perf_counter()
            traj = await asyncio.wait_for(rollout_task, timeout=1530)
            agent_loop_s = time.perf_counter() - agent_loop_start
        except asyncio.TimeoutError:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "timeout"
            error_detail = "rollout_timed_out"
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout timed out for task {task.id}", flush=True)
            raise
        except Exception as e:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "run_episode_failed"
            error_detail = str(e)
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout failed for task {task.id}: {str(e)}", flush=True)
            raise

        await run_cleanup()
        ignore_rollout: bool = False
        if env.ignore_rollout:
            ignore_rollout = True
        finish_msg = traj.finish_message
        error_msg = traj.error_message
        if finish_msg is None or "Error in episode loop at step" in (error_msg or ""):
            ignore_rollout = True
        if ignore_rollout:
            status = "ignored_rollout"
            error_detail = error_msg or "internal_errors"
            raise RuntimeError(f"Rollout ignored for task {task.id} due to internal errors")
        status = "success"
        if config.return_dict:
            result = current_trajectory_collection.get().to_dict()
            if tinker_proxy is not None and tinker_proxy_session_id is not None:
                result["_tinker_interactions"] = tinker_proxy.pop_interactions(tinker_proxy_session_id)
            return result
        else:
            return current_trajectory_collection.get() 
    except Exception as e:
        if status == "started":
            status = "error"
            error_detail = str(e)
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}", flush=True)
        if tinker_proxy is not None and tinker_proxy_session_id is not None:
            tinker_proxy.discard_session(tinker_proxy_session_id)
        await run_cleanup()
        raise
    finally:
        await run_cleanup()
        try:
            shutil.rmtree(overlay_root_dir)
        except:
            pass
        print(
            "ROLLOUT_TIMING "
            + json.dumps(
                {
                    "task_id": task.id,
                    "status": status,
                    "error": error_detail,
                    "prepare_workspace_s": prepare_workspace_s,
                    "agent_loop_s": agent_loop_s,
                    "cleanup_total_s": cleanup_total_s,
                    "total_rollout_s": time.perf_counter() - rollout_start,
                    "events_path": events_path,
                },
                sort_keys=True,
            ),
            flush=True,
        )
