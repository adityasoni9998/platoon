import logging

from openhands.sdk.workspace import BaseWorkspace
from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_action, is_finished
from platoon.issue_resolution.test_execution_reward.test_execution_reward import compute_test_execution_reward
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
):
    workspace.execute_command(f"cd {repo_path} ; git add -A")
    workspace.execute_command(
        f"cd {repo_path} && "
        f"git config --global user.email 'evaluation@openhands.dev' && "
        f"git config --global user.name 'OpenHands Evaluation' && "
        f'git config --global core.pager ""'
    )
    workspace.execute_command(f"cd {repo_path} && {remove_binary_files_from_git()}")
    git_patch_result = workspace.execute_command(f"cd {repo_path}; git diff --no-color --cached")
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
        self._enable_length_penalty = enable_length_penalty
        self._length_penalty_threshold = length_penalty_threshold

    async def evaluate(self) -> tuple[float, dict]:
        if not is_finished(self._state):
            return 0.0, {}

        instance: dict = self._task.misc

        info = {}

        # --- extract model patch ---
        try:
            model_patch = extract_patch_from_environment(
                workspace=self._workspace,
                repo_path=instance["repo_path"],
            )
        except Exception as e:
            logger.warning("Failed to extract model patch: %s", e)
            model_patch = ""
            info["patch_extraction_error"] = "Failed to extract model patch: " + str(e)

        # Execute tests on Modal
        test_execution_reward, test_execution_info = await compute_test_execution_reward(model_patch, instance)
        info.update(test_execution_info)
        if not self._enable_length_penalty:
            return test_execution_reward, info

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
        reward = test_execution_reward + length_reward_weight * length_reward
        info.update(
            binary_reward=test_execution_reward,
            length_reward=length_reward,
            length_reward_weight=length_reward_weight,
            num_agent_turns=num_turns,
            length_penalty_threshold=self._length_penalty_threshold,
            total_reward=reward,
        )
        return reward, info
