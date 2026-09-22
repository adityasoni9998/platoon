"""Single-use rollout child. No trainer, proxy, or dataset initialization here."""

import asyncio
import importlib
import logging
import pickle
import signal
import sys


async def run(module_name, function_name, request_path, response_path):
    # SIGTERM requests ordinary rollout cleanup; the parent enforces a deadline
    # if a blocking SDK call prevents this loop from handling the signal.
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    rollout_fn = getattr(importlib.import_module(module_name), function_name)
    with open(request_path, "rb") as stream:
        rollout_task, config = pickle.load(stream)
    result = await rollout_fn(rollout_task, config)
    with open(response_path, "wb") as stream:
        pickle.dump(result, stream)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for name in ("openhands", "httpx", "litellm", "LiteLLM"):
        logging.getLogger(name).setLevel(logging.WARNING)
    asyncio.run(run(*sys.argv[1:]))
