import asyncio
import logging
import shlex
from pathlib import Path

from openhands.sdk.workspace import BaseWorkspace
from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished
from platoon.issue_resolution.test_execution_reward.test_execution_reward import compute_test_execution_reward
from platoon.issue_resolution.localization_reward.localization_reward import compute_localization_reward, score_sources
from platoon.issue_resolution.tool_error_reward.tool_json_error import compute_tool_json_error_reward
from platoon.issue_resolution.tool_error_reward.agent_error_event import compute_agent_error_reward
from platoon.issue_resolution.tool_error_reward.str_replace_errors import compute_str_replace_reward
from platoon.issue_resolution.turn_snapshots import (
    EMPTY_PATCH_HASH, capture_command, completion_snapshots, download_snapshots,
)
logger = logging.getLogger(__name__)

def remove_binary_files_from_git():
    """
    Generate a bash command to remove binary files from git staging.
    Returns:
        str: A bash command that removes binary files from git staging
    """
    return """
    for file in $(git status --porcelain | grep -E "^(M| M|\\?\\?|A| A)" | cut -c4-); do
        if [ -f "$file" ] && (file "$file" | grep -q "executable" || \\
            git check-attr binary "$file" | grep -q "binary: set"); then
            git rm -f "$file" 2>/dev/null || rm -f "$file"
            echo "Removed: $file"
        fi
    done
    """.strip()


def extract_patch_from_environment(
    workspace: BaseWorkspace,
    repo_path: str,
    base_commit: str = "HEAD",
):
    workspace.execute_command(f"cd {repo_path} ; git add -A")
    workspace.execute_command(
        f"cd {repo_path} && "
        f"git config --global user.email 'evaluation@openhands.dev' && "
        f"git config --global user.name 'OpenHands Evaluation' && "
        f'git config --global core.pager ""'
    )
    workspace.execute_command(f"cd {repo_path} && {remove_binary_files_from_git()}")
    git_patch_result = workspace.execute_command(
        f"cd {repo_path}; git diff --no-color --cached {shlex.quote(base_commit)}"
    )
    if git_patch_result.exit_code != 0:
        logger.error(f"git diff command failed with exit code {git_patch_result.exit_code} and stderr: {git_patch_result.stderr} {git_patch_result.stdout}")
    git_patch = git_patch_result.stdout
    return git_patch

class SWERebenchEnv(OpenHandsEnv):
    def __init__(
        self, task, agent, workspace, *, use_binary_reward: bool = False,
        turn_rewards: bool = False, reward_artifact_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(task=task, agent=agent, workspace=workspace, **kwargs)
        self._terminal_result = None
        self._use_binary_reward = use_binary_reward
        self._turn_rewards_enabled = turn_rewards
        self._test_execution_semaphore = asyncio.Semaphore(8)
        self._snapshot_scores = {}
        self._completion_snapshots = {}
        self._previous_patch_hash = EMPTY_PATCH_HASH
        self._best_localization = self._best_test = 0.0
        self.turn_reward_error = None
        self._reward_artifact_dir = Path(reward_artifact_dir) if reward_artifact_dir else None
        if turn_rewards:
            if not self._wait_for_completion:
                raise ValueError("Turn rewards require wait_for_completion=True")
            if use_binary_reward:
                raise ValueError("Turn rewards require dense rewards")
            if self._reward_artifact_dir is None:
                raise ValueError("Turn rewards require a reward_artifact_dir")

    async def reset(self):
        if not self._turn_rewards_enabled:
            return await super().reset()
        self.turn_reward_error = None
        self._previous_patch_hash = EMPTY_PATCH_HASH
        self._best_localization = self._best_test = 0.0
        try:
            observation = await super().reset()
            # Generation has finished. Reuse the SDK action/completion IDs;
            # the normal step loop will handle action/observation pairing.
            self._completion_snapshots = completion_snapshots(
                self._state.conversation_state.events, capture_command(self._task.misc),
            )
            snapshots = await asyncio.to_thread(
                download_snapshots, self._workspace, self._reward_artifact_dir,
            )
            snapshots.setdefault(EMPTY_PATCH_HASH, {"patch": ""})
            results = await asyncio.gather(*(self._score_snapshot(s) for s in snapshots.values()))
            self._snapshot_scores = dict(zip(snapshots, results, strict=True))
            return observation
        except Exception as error:
            # run_episode logs ordinary exceptions and returns its partial
            # trajectory. run_rollout must reject that result, not train on zeros.
            self.turn_reward_error = str(error) or type(error).__name__
            raise

    async def _score_snapshot(self, snapshot):
        if not snapshot["patch"]:
            # Assume initial F2P failures, P2P passes, and no localized edits.
            return {"localization_reward": 0.0, "composite_reward": 0.0,
                    "binary_reward": 0.0, "f2p_pass_fraction": 0.0, "p2p_fail_fraction": 0.0}
        async with self._test_execution_semaphore:
            test, info = await compute_test_execution_reward(snapshot["patch"], self._task.misc)
        try:
            localization, details = await asyncio.to_thread(
                score_sources, snapshot["patch"], snapshot["before_sources"],
                snapshot["after_sources"], self._task.misc.get("file_changes"),
            )
        except Exception as error:
            # Preserve the existing localization processor's error-to-zero policy.
            localization, details = 0.0, {"status": "error", "error": str(error)}
            if snapshot.get("source_errors"):
                details["source_errors"] = snapshot["source_errors"]
        return {**info, "composite_reward": test,
                "localization_reward": localization, "localization": details}

    def _evaluate_turn(self):
        step = self._current_step
        completion = step.misc["action_misc"]["completion_id"]
        patch_hash = self._completion_snapshots.get(completion, self._previous_patch_hash)
        scores = self._snapshot_scores[patch_hash]
        localization, test = scores["localization_reward"], scores["composite_reward"]
        dl, dt = localization - self._best_localization, test - self._best_test
        final = is_finished(self._state)
        if not final:
            dl, dt = max(dl, 0.0), max(dt, 0.0)
        self._best_localization = max(self._best_localization, localization)
        self._best_test = max(self._best_test, test)

        events = [*(step.action_events or []), *(step.observation_events or [])]
        json_reward, json_info = compute_tool_json_error_reward(events)
        agent_reward, agent_info = compute_agent_error_reward(events)
        _, editor_info = compute_str_replace_reward(events)
        penalties = {
            "tool_json_penalty": -0.05 if json_reward == 0 else 0.0,
            "agent_error_penalty": -0.05 if agent_reward == 0 else 0.0,
            "file_editor_penalty": -0.05 if editor_info["file_editor_errors"] else 0.0,
        }
        reward = 0.15 * dl + 0.70 * dt + sum(penalties.values())
        info = {
            **json_info, **agent_info, **editor_info, **penalties,
            "reward_mode": "turn", "patch_hash": patch_hash,
            "snapshot_unchanged": patch_hash == self._previous_patch_hash,
            "localization_reward": localization, "composite_reward": test,
            "localization_delta": dl, "test_execution_delta": dt,
            "final_settlement": final, "total_reward": reward,
        }
        if final or patch_hash != self._previous_patch_hash:
            info.update(scores)
        self._previous_patch_hash = patch_hash
        return reward, info

    async def evaluate(self) -> tuple[float, dict]:
        if self._turn_rewards_enabled:
            try:
                return self._evaluate_turn()
            except Exception as error:
                self.turn_reward_error = str(error) or type(error).__name__
                raise
        if not is_finished(self._state):
            return 0.0, {}

        if self._terminal_result is not None:
            return self._terminal_result

        instance: dict = self._task.misc

        info = {}

        # --- extract model patch ---
        try:
            model_patch = extract_patch_from_environment(
                workspace=self._workspace,
                repo_path=instance["repo_path"],
                base_commit=instance.get("base_commit", "HEAD"),
            )
        except Exception as e:
            logger.warning("Failed to extract model patch: %s", e)
            model_patch = ""
            info["patch_extraction_error"] = "Failed to extract model patch: " + str(e)

        # Read dirty sources, reset to base_commit, and score locally.
        info["model_patch"] = model_patch
        localization_reward, localization_info = await asyncio.to_thread(
            compute_localization_reward, model_patch, instance, self._workspace
        )
        info["localization"] = localization_info
        info["localization_reward"] = localization_reward

        # Execute tests on Modal
        test_execution_reward, test_execution_info = await compute_test_execution_reward(model_patch, instance)
        info.update(test_execution_info)
        # Compute tool_json_error_reward
        tool_json_error_reward, tool_json_error_reward_info = compute_tool_json_error_reward(self._state.conversation_state.events)
        info.update(tool_json_error_reward_info)

        # Compute agent_error_event_reward
        agent_error_event_reward, agent_error_event_reward_info = compute_agent_error_reward(self._state.conversation_state.events)
        info.update(agent_error_event_reward_info)

        str_replace_reward, str_replace_reward_info = compute_str_replace_reward(self._state.conversation_state.events)
        info.update(str_replace_reward_info)

        # reward weights
        TEST_EXECUTION_REWARD_WEIGHT = 0.70
        LOCALIZATION_REWARD_WEIGHT = 0.15
        TOOL_JSON_ERROR_REWARD_WEIGHT = 0.05
        AGENT_ERROR_EVENT_REWARD_WEIGHT = 0.05
        STR_REPLACE_REWARD_WEIGHT = 0.05

        reward = (test_execution_reward * TEST_EXECUTION_REWARD_WEIGHT) + \
                 (localization_reward * LOCALIZATION_REWARD_WEIGHT) + \
                 (tool_json_error_reward * TOOL_JSON_ERROR_REWARD_WEIGHT) + \
                 (agent_error_event_reward * AGENT_ERROR_EVENT_REWARD_WEIGHT) + \
                 (str_replace_reward * STR_REPLACE_REWARD_WEIGHT)
        # Select the training reward after computing all component diagnostics.
        if self._use_binary_reward:
            reward = float(test_execution_info.get("binary_reward", 0.0))
        self._terminal_result = (reward, info)
        return self._terminal_result
