from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from tabnanny import verbose
import time
import traceback

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uuid

# This process creates the test Sandbox from inside the deployed Modal
# function, so the trainer process's environment does not reach this client.
os.environ["MODAL_SANDBOX_V2"] = "1"

import modal
import tenacity
from unidiff import PatchSet
from datasets import load_dataset
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    LOG_INSTANCE,
    DOCKER_WORKDIR,
    DOCKER_PATCH,
    TESTS_TIMEOUT,
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
)
from swebench.harness.eval import get_log_dir
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from typing import Any
from swebench.harness.reporting import make_run_report

SANDBOX_ENTRYPOINT = "run_evaluation_modal_entrypoint"
LOCAL_MODAL_TEST_EXECUTION_PATH = str(Path(__file__).resolve())
LOCAL_SANDBOX_ENTRYPOINT_PATH = str(
    (Path(__file__).resolve().parent / f"{SANDBOX_ENTRYPOINT}.py").resolve()
)
REMOTE_SANDBOX_ENTRYPOINT_PATH = f"/tmp/{SANDBOX_ENTRYPOINT}.py"
REMOTE_EVAL_SCRIPT_PATH = "/tmp/eval.sh"
DEFAULT_MODEL_NAME = "modal"
DEFAULT_TIMEOUT = 60 * 15  # 15 minutes
REMOTE_LOG_ROOT = Path("/tmp/swerebenchv1-modal-eval")
DOCKER_TEST_OUTPUT = "/tmp/swerebench_test_output.txt"
app = modal.App("swerebenchv1-evaluation")

swerebench_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "git+https://github.com/adityasoni9998/"
        "SWE-rebench-harness.git@1e5839bc9df38a4f495dac289241d49c4174db33",
        "modal==1.5.5",
        "tenacity",
    )
)

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

@dataclass
class TestOutput:
    instance_id: str
    resolved: bool
    test_output: str
    report: dict[str, Any]
    run_instance_log: str
    patch_diff: str
    log_dir: Path
    timed_out: bool
    errored: bool

class ModalSandboxRuntime:
    """
    Runtime for running instances in a Modal Sandbox.
    """
    def __init__(
        self,
        named_server_image: str,
        timeout: int | None = None,
        verbose: bool = True,
    ):
        self.image = ModalSandboxRuntime.get_instance_image(named_server_image)
        self.sandbox = self._get_sandbox(timeout)
        self.verbose = verbose
        self._stream_tasks = []

        # Hack for pylint
        # self.write_file("/sys/fs/cgroup/cpu/cpu.shares", "2048")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(7),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    )
    def _get_sandbox(self, timeout: int | None = None):
        # Sometimes network flakiness causes the image build to fail,
        # so we retry a few times.
        if timeout is None:
            # Default 30 minutes
            timeout = 60 * 30

        return modal.Sandbox.create(
            image=self.image,
            timeout=timeout,
            cpu=4,
        )
    async def _read_stream(
        self,
        stream: modal.io_streams.StreamReader,
        output_list: list[str],
        merged_output: list[str] | None = None,
    ):
        try:
            async for line in stream:
                output_list.append(line)
                if merged_output is not None:
                    merged_output.append(line)
                if self.verbose:
                    print(line)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error reading stream: {e}")

    async def _read_output(
        self,
        p: modal.container_process.ContainerProcess,
        stdout: list[str],
        stderr: list[str],
        merged_output: list[str],
    ):
        self._stream_tasks = [
            asyncio.create_task(self._read_stream(p.stdout, stdout, merged_output)),
            asyncio.create_task(self._read_stream(p.stderr, stderr, merged_output)),
        ]
        try:
            await asyncio.gather(*self._stream_tasks)
        except asyncio.CancelledError:
            pass    

    def exec(self, command: str) -> tuple[str, int]:
        p = self.sandbox.exec("python", REMOTE_SANDBOX_ENTRYPOINT_PATH, command)
        stdout = []
        stderr = []
        merged_output = []
        try:
            # We separate stdout/stderr because some tests rely on them being separate.
            # We still read stdout/stderr simultaneously to continuously
            # flush both streams and avoid blocking.
            asyncio.run(self._read_output(p, stdout, stderr, merged_output))
        except Exception as e:
            print(f"Error during command execution: {e}")
        p.wait()
        if merged_output:
            return "".join(merged_output), p.returncode
        return "".join(stdout + stderr), p.returncode
        
    def write_file(self, file_path: str, content: str):
        self.sandbox.filesystem.write_text(content, file_path)

    def read_file(self, file_path: str) -> str:
        return self.sandbox.filesystem.read_text(file_path)

    def close(self):
        try:
            for task in self._stream_tasks:
                if not task.done():
                    task.cancel()
            self.sandbox.terminate()
        except Exception:
            pass
        finally:
            self._stream_tasks = []
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @staticmethod
    def get_instance_image(named_server_image: str) -> modal.Image:
        return modal.Image.from_name(named_server_image).add_local_file(
                LOCAL_SANDBOX_ENTRYPOINT_PATH,
                REMOTE_SANDBOX_ENTRYPOINT_PATH,
            )

@app.function(
    image=(
        swerebench_image
        .add_local_file(
            LOCAL_MODAL_TEST_EXECUTION_PATH,
            "/root/modal_test_execution.py",
        )
        .add_local_file(
            LOCAL_MODAL_TEST_EXECUTION_PATH,
            "/root/platoon/issue_resolution/modal_test_execution.py",
        )
        .add_local_file(
            LOCAL_SANDBOX_ENTRYPOINT_PATH,
            REMOTE_SANDBOX_ENTRYPOINT_PATH,
        )
    ),
    timeout=120 * 60,  # Much larger than default timeout to account for image build time
    include_source=False,
)
def run_instance_modal(
    prediction: dict[str, Any] | str,
    instance: dict[str, Any],
    run_id: str,
    named_server_image: str,
    f2p_only: bool = False,
    is_gold: bool = False,
    timeout: int | None = None,
    verbose: bool = False,
) -> TestOutput:
    test_spec: TestSpec = make_test_spec(instance)
    runner: ModalSandboxRuntime | None = None
    log_lines: list[str] = []
    instance_id = test_spec.instance_id
    prediction = _normalize_prediction(prediction, instance_id)
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    patch_diff = prediction[KEY_PREDICTION] or ""
    def log(message: str):
        log_lines.append(f"{message}\n")
        if verbose:
            print(message)
    def finalize(
        *,
        test_output: str = "",
        errored: bool = False,
        timed_out: bool = False,
    ) -> TestOutput:
        return _write_result_artifacts(
            test_spec=test_spec,
            prediction=prediction,
            run_id=run_id,
            run_instance_log="".join(log_lines),
            patch_diff=patch_diff,
            test_output=test_output,
            timed_out=timed_out,
            timeout=timeout,
            errored=errored
        )
    try:
        runner = ModalSandboxRuntime(
            verbose=verbose,
            timeout=timeout,
            named_server_image=named_server_image,
        )
        runner.write_file(DOCKER_PATCH, patch_diff)
        apply_succeeded = False
        for i, git_apply_cmd in enumerate(GIT_APPLY_CMDS):
            apply_output, apply_code = runner.exec(
                f"cd {DOCKER_WORKDIR} && {git_apply_cmd} {DOCKER_PATCH}"
            )
            if apply_code == 0:
                    apply_succeeded = True
                    log(f"Patch applied successfully with command idx {i}: '{git_apply_cmd} {DOCKER_PATCH}'")
                    break
            else:
                # NOTE: even if the patch application command can return a non-zero exit code, there can still be partial patch application. Clean the state of the repo before trying the next command.
                # FIXME: this can interfere with the repository in cases if the repo had uncommitted tracked/untracked changes by default even before the agent started inside the docker image. But that corner case is ignored for now as it happens for <80/6542 images so this is treated as noise :)
                runner.exec(f"cd {DOCKER_WORKDIR} && git reset --hard")
                runner.exec(f"cd {DOCKER_WORKDIR} && git clean -fd")
                log(f"Patch apply command '{git_apply_cmd} {DOCKER_PATCH}' failed with exit code {apply_code}:\n{apply_output}")
        if not apply_succeeded:
            return finalize(errored=True)
        
        eval_commands = list(test_spec.eval_script_list)
        eval_commands.remove(test_spec.install_config["install"])
        eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_commands) + "\n"
        runner.write_file(REMOTE_EVAL_SCRIPT_PATH, eval_script)
        _, returncode = runner.exec(
            f"timeout {timeout}s /bin/bash {REMOTE_EVAL_SCRIPT_PATH} > {DOCKER_TEST_OUTPUT} 2>&1"
        )
        test_output = runner.read_file(DOCKER_TEST_OUTPUT)
        log(f"Test command output: {test_output}")
        timed_out = returncode == 124
        return finalize(test_output=test_output, timed_out=timed_out)
    except modal.exception.SandboxTimeoutError:
        log(f"Evaluation timed out after {timeout} seconds")
        return finalize(timed_out=True)
    except Exception:
        log(traceback.format_exc())
        return finalize(errored=True)
    finally:
        if runner is not None:
            runner.close()


def _normalize_prediction(prediction: dict[str, Any] | str, instance_id: str) -> dict[str, Any]:
    if isinstance(prediction, str):
        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: prediction,
            KEY_MODEL: DEFAULT_MODEL_NAME,
        }
    else:
        prediction = dict(prediction)

    prediction.setdefault(KEY_INSTANCE_ID, instance_id)
    prediction.setdefault(KEY_PREDICTION, "")
    prediction.setdefault(KEY_MODEL, DEFAULT_MODEL_NAME)
    return prediction

def _write_result_artifacts(
    test_spec: TestSpec,
    prediction: dict[str, Any],
    run_id: str,
    run_instance_log: str,
    patch_diff: str,
    test_output: str,
    timed_out: bool,
    timeout: int,
    errored: bool
) -> TestOutput:
    model_name = prediction.get(KEY_MODEL, "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name / test_spec.instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / LOG_INSTANCE).write_text(run_instance_log)
    (log_dir / "patch.diff").write_text(patch_diff)

    if timed_out:
        timeout_error = f"Timeout error: {timeout} seconds exceeded."
        test_output = (
            f"{test_output}\n\n{timeout_error}"
        )
        report = {
            "timed_out": True,
            "patch_exists": len(patch_diff.strip()) > 0,
            "resolved": False,
            KEY_MODEL: model_name,
        }
        return TestOutput(
            instance_id=test_spec.instance_id,
            resolved=False,
            test_output=test_output,
            report=report,
            run_instance_log=run_instance_log,
            patch_diff=patch_diff,
            log_dir=log_dir,
            timed_out=True,
            errored=False,
        )
    if errored:
        report = {
            "patch_exists": len(patch_diff.strip()) > 0,
            "resolved": False,
            "errored": True,
            KEY_MODEL: model_name,
        }
        return TestOutput(
            instance_id=test_spec.instance_id,
            resolved=False,
            test_output=test_output,
            report=report,
            run_instance_log=run_instance_log,
            patch_diff=patch_diff,
            log_dir=str(log_dir),
            errored=True,
            timed_out=False,
        )

    (log_dir / LOG_TEST_OUTPUT).write_text(test_output)    
    test_output_path = log_dir / LOG_TEST_OUTPUT
    report_path = log_dir / LOG_REPORT
    patch_exists = len(patch_diff.strip()) > 0

    instance_id = prediction[KEY_INSTANCE_ID]

    report = get_eval_report(
        test_spec=test_spec,
        prediction=prediction,
        test_log_path=test_output_path,
        include_tests_status=True,
    )[instance_id]
    return TestOutput(
            instance_id=instance_id,
            resolved=bool(report.get("resolved", False)),
            test_output=test_output,
            report=report,
            run_instance_log=run_instance_log,
            patch_diff=patch_diff,
            log_dir=str(log_dir),
            errored=False,
            timed_out=False,
        )

def sample_testing_code(instances):
    from platoon.issue_resolution.tasks import named_agent_server_image_for_instance

    start_time = time.time()
    test_specs = list(map(make_test_spec, instances))
    with modal.enable_output():
        with app.run():
            results = run_instance_modal.starmap(
                [
                    (
                        {
                            KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
                            KEY_PREDICTION: instance["patch"],
                            KEY_MODEL: "test_model",
                        },
                        instance,
                        uuid.uuid4().hex,
                        named_agent_server_image_for_instance(instance),
                        False,
                        True,
                        5*60, #NOTE: allow 5 minute timeout.
                        False,
                    )
                    for test_spec, instance in zip(test_specs, instances)
                ],
                return_exceptions=True,
            )
            err = 0
            fail = 0
            problematic_repos = []
            for res in results:
                try:
                    if not res.resolved:
                        print(f"Test failed for instance_id: {res.instance_id}")
                        fail += 1
                        run_id = uuid.uuid4().hex
                        with open(f"full_test/error_log_{run_id}.json", "w") as f:
                            from dataclasses import asdict
                            repo_name = res.instance_id
                            problematic_repos.append(repo_name)
                            json.dump(asdict(res), f, indent=4)
                    else:
                        print(f"Test passed for instance_id: {res.instance_id}", flush=True)
                except Exception as e:
                    try:
                        print(f"Error during evaluation: {res.instance_id}, error: {e}", flush=True)
                    except:
                        print(f"Error during evaluation: {e}", flush=True)
                    err += 1
    print(f"Total time for testing: {time.time() - start_time} seconds", flush=True)
    print(f"Total instances: {len(instances)}, Failed: {fail}, Errors: {err}", flush=True)


if __name__ == "__main__":
    max_cnt = 128
    batch_size = 32
    from datasets import load_dataset
    dataset = load_dataset("nebius/SWE-rebench", split="filtered")
    repo_seen = set()
    instances = []
    for instance in dataset:
        instances.append(instance)
    import random
    random.seed(42)
    random.shuffle(instances)
    for i in range(0, len(instances), batch_size):
        batch = instances[i:min(i + batch_size, len(instances))]
        print(len(batch))
        sample_testing_code(batch)
        break
