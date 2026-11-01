"""Per-trajectory reward breakdown for training metrics (reward/* keys).

Reads the reward info each turn stores in ``step.misc["reward_misc"]``:

- terminal fields, taken from the end of the trajectory: localization reward and its three F1 scores,
  test reward (composite), binary reward, and the F2P/P2P pass rates. Defaults are 0, except the P2P
  pass rate, which defaults to 1 (an unchanged repository breaks no passing test);
- each tool-formatting penalty summed over all turns, its cumulative contribution to the reward. Default 0.

Every trajectory reports every key: the workflow pairs root_* metrics with a mask shared by all groups,
so a key missing from one group would misalign the others.
"""

TOOL_PENALTIES = ("tool_json_penalty", "agent_error_penalty", "file_editor_penalty")
F1_SCORES = {"file_f1": "file_f1_reward", "module_f1": "module_f1_reward", "function_f1": "entity_f1_reward"}


def reward_breakdown(trajectory: dict) -> tuple[float, dict[str, float]]:
    """Return the trajectory's reward unchanged, with its reward breakdown as metrics."""
    infos = [(step.get("misc") or {}).get("reward_misc") or {} for step in trajectory.get("steps", [])]
    # Each turn reports its current patch's localization and test rewards; the full scores (test fractions,
    # localization details) come with turns whose patch changed and with final settlement. So the last turn
    # that reports each describes the final patch. Without them, the empty-patch scores apply: no F2P test
    # passes, every P2P test still passes, and nothing is localized.
    rewards = next((info for info in reversed(infos) if "composite_reward" in info), {})
    scores = next((info for info in reversed(infos) if "binary_reward" in info), {})
    # A scoring error keeps no localization details either: its F1 scores are 0.
    details = scores.get("localization") if isinstance(scores.get("localization"), dict) else {}

    metrics = {
        "reward/final/localization_reward": float(rewards.get("localization_reward", 0.0)),
        "reward/final/test_reward": float(rewards.get("composite_reward", 0.0)),
        "reward/final/binary_reward": float(scores.get("binary_reward", 0.0)),
        "reward/final/f2p_pass_rate": float(scores.get("f2p_pass_fraction", 0.0)),
        "reward/final/p2p_pass_rate": 1.0 - float(scores.get("p2p_fail_fraction", 0.0)),
        **{f"reward/final/{name}": float(details.get(key, 0.0)) for name, key in F1_SCORES.items()},
        **{f"reward/{key}": sum(float(info.get(key, 0.0)) for info in infos) for key in TOOL_PENALTIES},
    }
    return trajectory["reward"], metrics
