import logging

from openhands.sdk.workspace import BaseWorkspace
from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished
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
    async def evaluate(self) -> tuple[float, dict]:
        if not is_finished(self._state):
            return 0.0, {}

        instance: dict = self._task.misc

        reward = 0.0
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
        reward = test_execution_reward
        info.update(test_execution_info)
        return reward, info
