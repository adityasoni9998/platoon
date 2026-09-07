"""Local source parsing and file/module/entity F1, without system dependencies."""

from .processing import capture_sources_and_reset, generate_edited_locations_for_patch


def parse_edits(file_changes):
    files, modules, entities = set(), set(), set()
    for change in file_changes:
        files.add(change["file"])
        edits = change.get("changes") or {}
        modules.update(edits.get("edited_modules") or [])
        entities.update(edits.get("edited_entities") or [])
    return files, modules, entities


def compute_f1_score(prediction, ground_truth):
    # Preserve the legacy convention: an empty gold set earns zero.
    if not ground_truth:
        return 0.0
    return 2 * len(prediction & ground_truth) / (len(prediction) + len(ground_truth))


def score_sources(model_patch, before_sources, after_sources, ground_truth):
    if ground_truth is None:
        raise ValueError("Missing file_changes ground truth; use adityasoni17/SWE-rebench")
    result = generate_edited_locations_for_patch(model_patch, before_sources, after_sources)
    if result["status"] != "success":
        raise ValueError(f"Localization processing failed: {result['status']}")
    changes = result["file_changes"]
    predicted = parse_edits(changes)
    gold = parse_edits(ground_truth)
    info = {"status": "success", "file_changes": changes}
    scores = []
    for name, pred, true in zip(("file", "module", "entity"), predicted, gold):
        score = compute_f1_score(pred, true)
        scores.append(score)
        plural = "entities" if name == "entity" else name + "s"
        info[f"edited_{plural}"] = sorted(pred)
        info[f"gt_{plural}"] = sorted(true)
        info[f"{name}_f1_reward"] = score
    reward = sum(scores) / 3
    info["localization_reward"] = reward
    return reward, info


def compute_localization_reward(model_patch, instance, workspace):
    """Capture/reset remotely, then parse locally; preserve errors in reward info."""
    try:
        gold = instance.get("file_changes")
        if gold is None:
            raise ValueError("Missing file_changes ground truth")
        before, after = capture_sources_and_reset(workspace, model_patch, instance)
        return score_sources(model_patch, before, after, gold)
    except Exception as error:
        return 0.0, {"status": "error", "error": str(error), "localization_reward": 0.0}
