from typing import Protocol

import tinker

from platoon.utils.stats_tracker import DeferredStats, StatsTracker


class RolloutDatums(list):
    """A rollout's training datums, carrying the rollout's own statistics.

    The step that trains the rollout records them (``record_stats``), so per-rollout metrics cover
    exactly the rollouts a step trains, not rollouts that finish during this step but train in the
    next one. A step records a stale result's statistics apart, under ``stale/``.
    """

    def __init__(self, datums: list[tinker.Datum], stats: DeferredStats):
        super().__init__(datums)
        self.stats = stats


def record_stats(rollout: list[tinker.Datum] | None, tracker: StatsTracker, prefix: str = "") -> None:
    """Record a consumed rollout's own statistics (a plain list carries none), keys prefixed by ``prefix``."""
    if isinstance(rollout, RolloutDatums):
        rollout.stats.record(tracker, prefix)


class RolloutWorkflow(Protocol):
    """Protocol for rollout workflows used in tinker RL training.

    Implementations should receive model_info and other dependencies via constructor.
    """

    async def arun_episode(self, data: dict) -> list[tinker.Datum] | None: ...
