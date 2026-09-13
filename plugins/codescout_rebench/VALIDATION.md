# Port validation — 2026-09-10

The source was copied from the working tree at
`babel-compute-node:/home/adityabs/hyperion_modal/plugins/codescout_rebench`.
That plugin was untracked on Babel. Task loading, reward scoring, both prompts,
the custom finish tool, and runtime asset upload/install code were preserved.
AReaL trainer, workflow routing, tunnel manager, and launcher were adapted from
the local issue-resolution-rebench plugin.

The locked AReaL environment was successfully installed at
`/tmp/adityabs-codescout-rebench/areal-venv`. No plugin `.venv` was created
inside the repository.

## Automated checks

```bash
/tmp/adityabs-codescout-rebench/areal-venv/bin/python -m pytest \
  tests/test_codescout_rebench.py \
  tests/test_areal_registry_and_workflow.py \
  tests/test_areal_openai_total_token_patch.py -q
```

Result: **36 passed, 1 skipped**. The skipped case is an impossible valid finish
with zero model turns. Coverage includes image naming parity, remote tool upload
failures, isolated import checks, reward boundaries, turn limits, independent
sampling seeds, timeout cleanup, worker-specific public routing, session key
preservation, and serialized workflow reconstruction. Ruff checks and formatting,
launcher shell syntax, and whitespace checks passed.

An additional check using the installed AReaL and OpenHands packages loaded the
YAML through `load_expr_config(..., PlatoonArealRLTrainerConfig)`, verified CISPO,
batch/group sizes, learning rate, and off-policy limit, reconstructed the actual
workflow from its serialized kwargs, and built real SDK LLM objects for all four
worker endpoint mappings while preserving their session keys.

## Live Modal / Ogma check

One real sandbox used task `ARMmbed__mbed-tools-138` and its existing named
SWE-rebench image. It successfully reset `/testbed`, uploaded and import-checked
the custom finish tool, and installed/verified ripgrep. Four temporary local
HTTP health servers stood in for the inference proxies. All four automatic
Ogma reverse forwards passed health checks, and the Modal sandbox successfully
made HTTP requests through each of them. The sandbox, servers, and SSH forwards
were cleaned up afterward.

This checked live network reachability and runtime asset hydration. It did not
run model inference, execute the finish tool in a conversation, or perform an
optimizer step. Use the README's training smoke command to exercise the complete
GPU training path.

## Existing test failures

A broader check also ran `tests/test_areal_batch_transforms.py`: three tests
passed and two failed while importing the trainer because the test's mocked
`platoon.train.areal` package omits `preallocated_slurm`. Both failures reproduce
when that existing test file is run alone, without the new plugin tests. The
actual installed trainer imports successfully. No shared backend or existing
test files were changed by this port.
