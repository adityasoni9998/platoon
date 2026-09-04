import logging
import sys
import uuid
from pathlib import Path

import modal
from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION

from platoon.issue_resolution.tasks import named_agent_server_image_for_instance

logger = logging.getLogger(__name__)
from dataclasses import asdict

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    # Modal deserializes the remote return value by importing the defining module
    # by name, which in this deployment is `modal_test_execution`.
    sys.path.insert(0, _THIS_DIR)


# Test execution reward: run evaluation of this patch on Modal
async def compute_test_execution_reward(model_patch: str, instance: dict):
    # Empty model patch is guaranteed to fail, so skip evaluation on Modal.
    binary_reward = 0.0
    if len(model_patch.strip()) == 0:
        return binary_reward, {
            "error": "Empty model patch ==> guaranteed to not resolve issues.",
            "binary_reward": binary_reward,
        }

    # Run tests on Modal.
    try:
        run_id = f"rl-{uuid.uuid4().hex}"
        with modal.enable_output():
            modal_fn = modal.Function.from_name(
                "swerebenchv1-evaluation", "run_instance_modal"
            )
            res = await modal_fn.remote.aio(
                prediction={
                    KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
                    KEY_PREDICTION: model_patch,
                    KEY_MODEL: "test_model",
                },
                instance=instance,
                run_id=run_id,
                f2p_only=False,
                is_gold=False,
                timeout=5 * 60,
                verbose=False,
                named_server_image=named_agent_server_image_for_instance(instance),
            )
        info = {"model_patch": model_patch, "evaluation_logs": asdict(res)}
        try:
            binary_reward = 1.0 if res.resolved else 0.0
            info = {"model_patch": model_patch, "evaluation_logs": asdict(res)}
        except Exception as e:
            binary_reward = 0.0
            info = {"model_patch": model_patch, "evaluation_logs": str(e)}
    except Exception as e:
        binary_reward = 0.0
        info = {"model_patch": model_patch, "evaluation_logs": str(e)}
    except:
        binary_reward = 0.0
        info = {
            "model_patch": model_patch,
            "evaluation_logs": (
                "Failed to evaluate patch on Modal due to an unknown error."
            ),
        }

    info["binary_reward"] = binary_reward
    return binary_reward, info
