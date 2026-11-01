"""Stats tracking utilities for training metrics.

A simplified stats tracker inspired by areal's DistributedStatsTracker:
https://github.com/inclusionAI/AReaL/blob/main/areal/utils/stats_tracker.py

This implementation is single-process and does not require torch.distributed.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Lock


class ReduceType(Enum):
    """Reduction types for aggregating statistics."""

    AVG_MIN_MAX = auto()  # Report average, min, and max
    AVG = auto()  # Report only average
    SUM = auto()  # Report sum
    MIN = auto()  # Report minimum
    MAX = auto()  # Report maximum
    SCALAR = auto()  # Single scalar value (averaged across calls)


@dataclass
class StatEntry:
    """A single statistic entry with value and optional denominator."""

    values: list[list | float] = field(default_factory=list)
    denominators: list[list[bool]] = field(default_factory=list)
    reduce_type: ReduceType = ReduceType.AVG


class StatsTracker:
    """Tracks and aggregates training statistics.

    Usage:
        tracker = StatsTracker("train")

        # Record a scalar
        tracker.scalar(loss=0.5)

        # Record per-item stats with a denominator (mask): plain lists
        tracker.denominator(batch_mask=[True] * batch_size)
        tracker.stat(rewards=rewards, denominator="batch_mask")

        # Export aggregated stats
        stats = tracker.export()
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.lock = Lock()
        self.scope_stack: list[str] = []
        if name:
            self.scope_stack.append(name.strip("/"))

        self.stats: dict[str, StatEntry] = defaultdict(StatEntry)
        self.denominators: dict[str, str] = {}  # stat_key -> denominator_key

    def _get_full_key(self, key: str) -> str:
        """Combine scope stack with current key."""
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def scope(self, name: str):
        """Context manager for hierarchical scoping."""
        with self.lock:
            self.scope_stack.append(name.strip("/"))
        try:
            yield
        finally:
            with self.lock:
                self.scope_stack.pop()

    @contextmanager
    def record_timing(self, key: str):
        """Record timing for a code block."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            with self.lock:
                full_key = f"timing/{key}"
                entry = self.stats[full_key]
                entry.values.append(elapsed)
                entry.reduce_type = ReduceType.SCALAR

    def scalar(self, **kwargs: float):
        """Record scalar values."""
        with self.lock:
            for key, value in kwargs.items():
                full_key = self._get_full_key(key)
                entry = self.stats[full_key]
                entry.values.append(float(value))
                entry.reduce_type = ReduceType.SCALAR

    def denominator(self, **kwargs: list[bool]):
        """Register denominator masks (lists of bools) for subsequent stat calls.

        Plain lists, not torch tensors: these stats are a few numbers per rollout, recorded on the
        trainer's event loop, where each torch call releases and re-acquires the GIL.
        """
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, list) or not all(isinstance(x, bool) for x in value):
                    raise ValueError(f"`{key}` must be a list of bools")
                full_key = self._get_full_key(key)
                entry = self.stats[full_key]
                entry.values.append(list(value))
                entry.reduce_type = ReduceType.SUM

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        **kwargs: list[float],
    ):
        """Record statistics (lists of numbers) with a denominator mask."""
        with self.lock:
            denom_key = self._get_full_key(denominator)
            if denom_key not in self.stats:
                raise ValueError(f"Denominator `{denom_key}` not registered")

            for key, value in kwargs.items():
                if not isinstance(value, list):
                    raise ValueError(f"`{key}` must be a list of numbers")

                full_key = self._get_full_key(key)
                entry = self.stats[full_key]
                entry.values.append([float(x) for x in value])
                entry.reduce_type = reduce_type or ReduceType.AVG_MIN_MAX
                self.denominators[full_key] = denom_key

    def export(self, reset: bool = True) -> dict[str, float]:
        """Export aggregated statistics."""
        with self.lock:
            results: dict[str, float] = {}

            for key, entry in list(self.stats.items()):
                if not entry.values:
                    continue

                aggregated = self._aggregate(key, entry)
                results.update(aggregated)

            if reset:
                self.stats.clear()
                self.denominators.clear()

            return results

    def _aggregate(self, key: str, entry: StatEntry) -> dict[str, float]:
        """Aggregate a single stat entry."""
        result: dict[str, float] = {}

        if entry.reduce_type == ReduceType.SCALAR:
            # Average of scalar values
            if entry.values:
                result[key] = sum(entry.values) / len(entry.values)
                result[f"{key}__count"] = len(entry.values)

        elif entry.reduce_type == ReduceType.SUM:
            # Sum of masks: the number of True entries
            result[key] = sum(sum(v) for v in entry.values)

        elif entry.reduce_type in (ReduceType.AVG, ReduceType.AVG_MIN_MAX):
            # Weighted average using denominator
            denom_key = self.denominators.get(key)
            if denom_key and denom_key in self.stats:
                denom_entry = self.stats[denom_key]

                numerator = 0.0
                denominator = 0.0
                mins = []
                maxs = []

                for v, d in zip(entry.values, denom_entry.values):
                    valid = [x for x, keep in zip(v, d, strict=True) if keep]
                    numerator += sum(valid)
                    denominator += len(valid)

                    if entry.reduce_type == ReduceType.AVG_MIN_MAX and valid:
                        mins.append(min(valid))
                        maxs.append(max(valid))

                if denominator > 0:
                    avg = numerator / denominator
                    if entry.reduce_type == ReduceType.AVG_MIN_MAX:
                        result[f"{key}/avg"] = avg
                        if mins:
                            result[f"{key}/min"] = min(mins)
                        if maxs:
                            result[f"{key}/max"] = max(maxs)
                    else:
                        result[key] = avg

        elif entry.reduce_type == ReduceType.MIN:
            mins = [min(v) for v in entry.values]
            if mins:
                result[key] = min(mins)

        elif entry.reduce_type == ReduceType.MAX:
            maxs = [max(v) for v in entry.values]
            if maxs:
                result[key] = max(maxs)

        return result


class DeferredStats:
    """Stat calls kept now and recorded into a tracker later, with the same arguments.

    A rollout's statistics belong to the step that trains it, not to the step during which it
    happened to finish: with rollouts running ahead, some finish during one step and train in the next.
    """

    def __init__(self):
        self._calls: list[tuple[str, tuple, dict]] = []

    def denominator(self, **kwargs: list[bool]):
        self._calls.append(("denominator", (), kwargs))

    def stat(self, denominator: str, reduce_type: ReduceType | None = None, **kwargs: list[float]):
        self._calls.append(("stat", (denominator, reduce_type), kwargs))

    def scalar(self, **kwargs: float):
        self._calls.append(("scalar", (), kwargs))

    def record(self, tracker: StatsTracker, prefix: str = ""):
        """Replay the calls; ``prefix`` (e.g. "stale/") renames every key and denominator."""
        for method, args, kwargs in self._calls:
            kwargs = {prefix + key: value for key, value in kwargs.items()}
            if method == "stat":
                denominator, reduce_type = args
                args = (prefix + denominator, reduce_type)
            getattr(tracker, method)(*args, **kwargs)


# Global trackers registry
_TRACKERS: dict[str, StatsTracker] = {}
_LOCK = Lock()


def get(name: str = "") -> StatsTracker:
    """Get or create a named tracker."""
    global _TRACKERS
    with _LOCK:
        if name not in _TRACKERS:
            _TRACKERS[name] = StatsTracker(name)
        return _TRACKERS[name]


def export_all(reset: bool = True) -> dict[str, float]:
    """Export stats from all trackers."""
    global _TRACKERS
    results: dict[str, float] = {}
    with _LOCK:
        for tracker in _TRACKERS.values():
            results.update(tracker.export(reset=reset))
    return results
