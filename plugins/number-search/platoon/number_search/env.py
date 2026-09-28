from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.episode.context import finish_message


def guess_factory(target: int):
    def guess(number: int) -> str:
        if number == target:
            finish_message.set(f"You guessed the number {target} correctly!")
        elif number < target:
            return "Too low, try again."
        else:
            return "Too high, try again."

    return guess


class NumberSearchEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(
            task,
            IPythonCodeExecutor(
                task,
                actions=(finish, guess_factory(task.misc["target"])),
                detect_while_loops=True,
                detect_interactive_input=True,
            ),
        )

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = 0.0, {}
        if self._state.finished:
            message = finish_message.get(None)
            if message is not None and "correctly" in message:
                return 1.0, {}
            else:
                return 0.0, {}
        return score, reward_misc
