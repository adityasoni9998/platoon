import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

APPTAINER_CACHEDIR = os.environ.get("APPTAINER_CACHEDIR", "/data/user_data/adityabs/apptainer_cache")
LOCALIZATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCALIZATION_DIR.parent.parent.parent
HOST_PATCH = Path("/usr/bin/patch")
HOST_PATCH_LIBS = [
    Path("/lib64/libattr.so.1"),
    Path("/lib64/libselinux.so.1"),
    Path("/lib64/libc.so.6"),
    Path("/lib64/libpcre2-8.so.0"),
    Path("/lib64/ld-linux-x86-64.so.2"),
]

def stage_host_patch_tool():
    if not HOST_PATCH.exists():
        raise FileNotFoundError(f"Host patch binary not found: {HOST_PATCH}")
    for lib in HOST_PATCH_LIBS:
        if not lib.exists():
            raise FileNotFoundError(f"Host patch dependency not found: {lib}")


def resolve_sif_path(instance):
    image_name = instance["image_name"].split("/")[-1]
    sif_path = Path(APPTAINER_CACHEDIR) / f"43376f1-93c33d0-{image_name}-source-minimal.sif"
    if not sif_path.exists():
        raise FileNotFoundError(f"Apptainer image not found: {sif_path}")
    return sif_path


def localization_processing(model_patch: str, instance: dict):
    try:
        sif_path = resolve_sif_path(instance)
    except FileNotFoundError:
        return {}, "Apptainer SIF does not exist!", "error"

    random_id = str(uuid.uuid4())
    patch_file_path = f"/tmp/{random_id}.patch"
    with open(patch_file_path, "w", newline="") as f:
        f.write(model_patch)

    overlay_root = Path(f"/tmp/overlay_{random_id}")
    (overlay_root / "upper").mkdir(parents=True, exist_ok=True)
    (overlay_root / "work").mkdir(parents=True, exist_ok=True)

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python_executable = venv_python if venv_python.exists() else Path(sys.executable)
    command = (
        "cd /testbed; git fetch origin; "
        f"git checkout {instance['instance_id']}; "
        "/host_tools/lib64/ld-linux-x86-64.so.2 --library-path /host_tools/lib64 "
        "/host_tools/patch --version >/tmp/patch_version.txt || true; "
        f"{python_executable} {LOCALIZATION_DIR / 'localization_patch_processing.py'} --id {random_id}"
    )
    apptainer_cmd = [
        "apptainer",
        "exec",
        "--fakeroot",
        "--cleanenv",
        "--overlay",
        str(overlay_root),          # replaces "--writable-tmpfs"
        "--bind",
        f"{REPO_ROOT}:{REPO_ROOT}",
        "--bind",
        "/tmp:/tmp",
        "--bind",
        f"{HOST_PATCH}:/host_tools/patch:ro",
        "--bind",
        "/lib64/libattr.so.1:/host_tools/lib64/libattr.so.1:ro",
        "--bind",
        "/lib64/libselinux.so.1:/host_tools/lib64/libselinux.so.1:ro",
        "--bind",
        "/lib64/libc.so.6:/host_tools/lib64/libc.so.6:ro",
        "--bind",
        "/lib64/libpcre2-8.so.0:/host_tools/lib64/libpcre2-8.so.0:ro",
        "--bind",
        "/lib64/ld-linux-x86-64.so.2:/host_tools/lib64/ld-linux-x86-64.so.2:ro",
        str(sif_path),
        "env",
        f"PYTHONPATH={LOCALIZATION_DIR}",
        "bash",
        "-c",
        command,
    ]
    result = subprocess.run(
        apptainer_cmd,
        text=True,
        capture_output=True,
        check=False,
        timeout=5 * 60,
    )

    # Try processing the output from the JSON log
    try:
        with open(f"/tmp/{random_id}_edited_locations.json", "r") as f:
            edited_locations = json.load(f)
    except Exception:
        edited_locations = {}
    cleanup_tmp_files(random_id)
    cleanup_overlay(overlay_root)
    file_changes = edited_locations.get("file_changes", [])
    status = edited_locations.get("status", "error")
    return file_changes, result.stdout, status

def cleanup_overlay(overlay_root):
    import shutil
    shutil.rmtree(overlay_root, ignore_errors=True)

def cleanup_tmp_files(random_id):
    for suffix in (".patch", "_edited_locations.json"):
        try:
            Path(f"/tmp/{random_id}{suffix}").unlink()
        except FileNotFoundError:
            pass

def parse_edits(file_changes):
    files = []
    modules = []
    entities = []
    if file_changes is None:
        file_changes = []
    for change in file_changes:
        if "file" in change:
            files.append(change["file"])
        if "changes" in change:
            edited_modules = change["changes"].get("edited_modules", [])
            edited_modules = [] if edited_modules is None else edited_modules
            for module in edited_modules:
                modules.append(module)

            edited_entities = change["changes"].get("edited_entities", [])
            edited_entities = [] if edited_entities is None else edited_entities
            for entity in edited_entities:
                entities.append(entity)
    return set(files), set(modules), set(entities)

def compute_f1_score(prediction, ground_truth):
    pred, true = set(prediction), set(ground_truth)
    if not true:
        return 0.0 # return 0 reward if ground truth is empty
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_localization_reward(model_patch: str, instance: dict):
    reward = 0.0
    reward_info = {"edited_files": [], "edited_modules": [], "edited_entities": [], "gt_files": [], "gt_modules": [], "gt_entities": [], "file_f1_reward": 0.0, "module_f1_reward": 0.0, "entity_f1_reward": 0.0, "localization_reward": 0.0, "logs": ""}
    if model_patch is None or len(model_patch.strip()) == 0:
        # Localization reward for an empty patch is guaranteed to be 0 so need to run spawn apptainer.
        return reward, reward_info
    try:
        file_changes, logs, status = localization_processing(model_patch, instance) 
        reward_info["status"] = status
        reward_info["logs"] = logs
        gt_files, gt_modules, gt_entities = parse_edits(instance.get("file_changes"))
        pred_files, pred_modules, pred_entities = parse_edits(file_changes)
        reward_info["edited_files"] = list(pred_files)
        reward_info["edited_modules"] = list(pred_modules)
        reward_info["edited_entities"] = list(pred_entities)
        reward_info["gt_files"] = list(gt_files)
        reward_info["gt_modules"] = list(gt_modules)
        reward_info["gt_entities"] = list(gt_entities)
        file_f1_reward = compute_f1_score(pred_files, gt_files)
        module_f1_reward = compute_f1_score(pred_modules, gt_modules)
        entity_f1_reward = compute_f1_score(pred_entities, gt_entities)
        reward_info["file_f1_reward"] = file_f1_reward
        reward_info["module_f1_reward"] = module_f1_reward
        reward_info["entity_f1_reward"] = entity_f1_reward
        reward = (file_f1_reward + module_f1_reward + entity_f1_reward) / 3.0
        reward_info["localization_reward"] = reward
        return reward, reward_info
    except Exception as e:
        logger.warning(f"Failed to compute localization reward: {e}")
        reward_info["status"] = "error"
        reward_info["logs"] = str(e)
        return 0.0, reward_info
