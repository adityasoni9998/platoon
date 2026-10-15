from dataclasses import dataclass
from typing import Callable

from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.envs.codeact.types import CodeActAction, CodeActObservation
from platoon.episode.context import finish_message


@dataclass
class RewardShapingConfig:
    enabled: bool = False
    midpoint_reward: float = 0.2
    invalid_guess_reward: float = -0.2


def guess_factory(target: int, on_guess: Callable[[int], None] | None = None):
    def guess(number: int) -> str:
        if on_guess is not None:
            on_guess(number)
        if number == target:
            finish_message.set(f"You guessed the number {target} correctly!")
        elif number < target:
            return "Too low, try again."
        else:
            return "Too high, try again."

    return guess


class NumberSearchEnv(CodeActEnv):
    def __init__(self, task: Task, reward_shaping: RewardShapingConfig | None = None):
        self.reward_shaping = reward_shaping or RewardShapingConfig()
        super().__init__(
            task,
            IPythonCodeExecutor(
                task,
                actions=(finish, guess_factory(task.misc["target"], self._record_guess)),
                detect_while_loops=True,
                detect_interactive_input=True,
            ),
        )
        self._reset_rewards()

    def _reset_rewards(self):
        self._lo, self._hi = self.task.misc["low"], self.task.misc["high"]
        self._success = False
        self._turn_success = False
        self._guessed = False
        self._shaping_reward = 0.0

    async def reset(self) -> CodeActObservation:
        self._reset_rewards()
        return await super().reset()

    async def step(self, action: CodeActAction) -> CodeActObservation:
        self._turn_success = False
        self._guessed = False
        self._shaping_reward = 0.0
        return await super().step(action)

    def _record_guess(self, number: int):
        # Enforce the same action rules regardless of reward shaping. Count the
        # attempt before validation so caught errors cannot buy another guess.
        if self._guessed:
            raise ValueError("Only one guess is allowed per agent turn.")
        self._guessed = True
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError("The guess must be an integer.")
        if self._success:
            return
        if self.reward_shaping.enabled:
            # Score against the interval BEFORE incorporating this guess's feedback.
            if number < self._lo or number > self._hi:
                self._shaping_reward = self.reward_shaping.invalid_guess_reward
            elif number in ((self._lo + self._hi) // 2, (self._lo + self._hi + 1) // 2):
                self._shaping_reward = self.reward_shaping.midpoint_reward

        target = self.task.misc["target"]
        if number == target:
            self._turn_success = self._turn_success or not self._success
            self._success = True
        elif number < target:
            self._lo = max(self._lo, number + 1)
        else:
            self._hi = min(self._hi, number - 1)

    async def evaluate(self) -> tuple[float, dict]:
        shaping = self._shaping_reward if self.reward_shaping.enabled else 0.0
        # Success is an increment on the successful turn, not another terminal reward.
        return float(self._turn_success) + shaping, {
            "success_rate": float(self._success),
            "shaping_reward": shaping,
        }


def reward_processor(trajectory: dict) -> tuple[float, dict]:
    components = [step.get("misc", {}).get("reward_misc", {}) for step in trajectory["steps"]]
    return trajectory["reward"], {
        "success_rate": float(any(info.get("success_rate", 0.0) for info in components)),
        "shaping_return": sum(info.get("shaping_reward", 0.0) for info in components),
    }
