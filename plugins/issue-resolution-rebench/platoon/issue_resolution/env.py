import asyncio
import logging
import shlex

from openhands.sdk.workspace import BaseWorkspace
from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_action, is_finished
from platoon.issue_resolution.test_execution_reward.test_execution_reward import compute_test_execution_reward
from platoon.issue_resolution.localization_reward.localization_reward import compute_localization_reward
from platoon.issue_resolution.tool_error_reward.tool_json_error import compute_tool_json_error_reward
from platoon.issue_resolution.tool_error_reward.agent_error_event import compute_agent_error_reward
from platoon.issue_resolution.tool_error_reward.str_replace_errors import compute_str_replace_reward
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
        self, task, agent, workspace, *, length_penalty_threshold: int,
        enable_length_penalty: bool = False, **kwargs,
    ):
        max_turns = task.max_steps
        if enable_length_penalty and (
            not isinstance(length_penalty_threshold, int)
            or isinstance(length_penalty_threshold, bool)
            or max_turns is None
            or not 0 <= length_penalty_threshold < max_turns
        ):
            raise ValueError("length_penalty_threshold must be an integer >= 0 and < task.max_steps")
        super().__init__(task=task, agent=agent, workspace=workspace, **kwargs)
        self._terminal_result = None
        self._enable_length_penalty = enable_length_penalty
        self._length_penalty_threshold = length_penalty_threshold

    async def evaluate(self) -> tuple[float, dict]:
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
        if not self._enable_length_penalty:
            self._terminal_result = (reward, info)
            return self._terminal_result

        # One response may emit several tool calls or messages. Count it once.
        response_ids = {
            event.llm_response_id
            for event in self._state.conversation_state.events
            if is_action(event) and getattr(event, "llm_response_id", None)
        }
        num_turns = len(response_ids)
        length_reward = (self._length_penalty_threshold - num_turns) / (
            self._task.max_steps - self._length_penalty_threshold
        )
        length_reward_weight = 1.0
        reward += length_reward_weight * length_reward
        info.update(
            length_reward=length_reward,
            length_reward_weight=length_reward_weight,
            num_agent_turns=num_turns,
            length_penalty_threshold=self._length_penalty_threshold,
            total_reward=reward,
        )
        self._terminal_result = (reward, info)
        return self._terminal_result
