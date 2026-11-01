"""StatsTracker records per-rollout stats as plain lists and exports the same metrics as before."""

import pytest

from platoon.utils.stats_tracker import DeferredStats, ReduceType, StatsTracker

# One entry per recorded call (e.g. per rollout): (values, mask)
CALLS = [
    ([14.0], [True]),
    ([3.0, 40.0], [True, False]),
    ([7.5, 0.25, 9.0], [True, True, True]),
]


def test_masked_avg_min_max_sum_min_and_max():
    tracker = StatsTracker("train")
    for values, mask in CALLS:
        tracker.denominator(x_mask=mask)
        tracker.stat(x=values, denominator="x_mask")
        tracker.stat(x_avg=values, denominator="x_mask", reduce_type=ReduceType.AVG)
        tracker.stat(x_min=values, denominator="x_mask", reduce_type=ReduceType.MIN)
        tracker.stat(x_max=values, denominator="x_mask", reduce_type=ReduceType.MAX)
    tracker.scalar(loss=0.5)
    stats = tracker.export()

    kept = [14.0, 3.0, 7.5, 0.25, 9.0]  # masked-out 40.0 excluded from avg/min/max
    assert stats["train/x/avg"] == pytest.approx(sum(kept) / len(kept))
    assert stats["train/x/min"] == 0.25
    assert stats["train/x/max"] == 14.0
    assert stats["train/x_avg"] == pytest.approx(sum(kept) / len(kept))
    assert stats["train/x_mask"] == 5  # a mask sums to its True count
    # MIN/MAX reduce over all recorded values, as before (no masking).
    assert stats["train/x_min"] == 0.25
    assert stats["train/x_max"] == 40.0
    assert stats["train/loss"] == 0.5 and stats["train/loss__count"] == 1
    assert tracker.export() == {}  # export resets


@pytest.mark.parametrize("mask", [[1, 0], ["yes"], (True,)])
def test_mask_must_be_a_list_of_bools(mask):
    with pytest.raises(ValueError, match="list of bools"):
        StatsTracker().denominator(m=mask)


def test_values_must_be_a_list_and_match_the_mask():
    tracker = StatsTracker()
    tracker.denominator(m=[True, True])
    with pytest.raises(ValueError, match="list of numbers"):
        tracker.stat(x=(1.0, 2.0), denominator="m")
    tracker.stat(x=[1.0], denominator="m")
    with pytest.raises(ValueError):
        tracker.export()


def test_deferred_stats_replay_later_and_can_be_prefixed():
    deferred = DeferredStats()
    deferred.denominator(x_mask=[True, False])
    deferred.stat(x=[2.0, 9.0], denominator="x_mask")
    deferred.stat(x_min=[2.0, 9.0], denominator="x_mask", reduce_type=ReduceType.MIN)
    deferred.scalar(n=1.0)
    tracker = StatsTracker("train")
    assert tracker.export() == {}  # nothing until replayed
    deferred.record(tracker)
    deferred.record(tracker, prefix="stale/")
    stats = tracker.export()
    for prefix in ("", "stale/"):
        assert stats[f"train/{prefix}x/avg"] == 2.0
        assert stats[f"train/{prefix}x_mask"] == 1
        assert stats[f"train/{prefix}x_min"] == 2.0
        assert stats[f"train/{prefix}n"] == 1.0
