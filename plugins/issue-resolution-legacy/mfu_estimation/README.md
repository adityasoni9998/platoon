# MFU estimator

This folder contains a read-only estimator for AReaL training MFU and SGLang
inference MFU. It does not attach to, signal, profile, or import the trainer.

## Dependencies

The estimator is **pure Python** and uses only the Python standard library. It
has no `pip` package dependencies. Python 3.10 or newer is sufficient; this
plugin already uses Python 3.12.

The standard-library modules used include `argparse`, `ast`, `datetime`,
`json`, `pathlib`, `re`, `subprocess`, and `urllib`.

Creating a virtual environment is optional:

```bash
python3 -m venv /tmp/mfu-estimation-venv
```

## Rerun this RL estimate

Run from the repository root. Replace `LOG_DIR` with the trial log directory
that contains `main.log` and `actor.log`:

```bash
LOG_DIR=/tmp/platoon_skyrl_areal/experiments/logs/adityabs/issue-resolution-platoon-areal/areal-flame-swesmith-fft-cispo-20260902-104322

/tmp/mfu-estimation-venv/bin/python \
  plugins/issue-resolution-legacy/mfu_estimation/estimate_mfu.py \
  --log-dir "$LOG_DIR" \
  --model-config plugins/issue-resolution-legacy/mfu_estimation/model_configs/qwen3_4b_instruct_2507.json \
  --train-gpus 4 \
  --context-parallel-size 2 \
  --gpu-peak-tflops 989 \
  --exclude-training-step 1 \
  --expected-sglang-servers 4
```

The default output is Markdown. Add `--format json` for machine-readable output.
Add `--training-only` when SGLang is not running.

The bundled model file makes this run reproducible without network access. For
another decoder-only Qwen/Llama-style model, either pass its local Hugging Face
`config.json`, pass `--model-id organization/model` to fetch it from Hugging
Face, or pass a precomputed matrix-parameter count with `--matrix-parameters`.

If other SGLang jobs share the host, use `--sglang-process-match TEXT` to select
only processes whose command contains `TEXT`, or repeat `--sglang-metrics-url`
for the desired discovered endpoints. `--expected-sglang-servers` prevents an
estimate from silently using the wrong number of inference workers.

## What is calculated

### Training metric

The reported training value is cumulative, idle-inclusive conventional 6N MFU:

```text
training MFU = sum(6 * N * non-padding tokens)
               / (training GPUs * peak FLOP/s/GPU * sum(complete cycle seconds))
```

`N` is the number of matrix weights involved in the model's dense forward and
backward passes. The factor 6 consists of approximately 2N forward FLOPs plus
4N backward FLOPs per token. Gradient-checkpoint recomputation is intentionally
not counted as useful model work.

The estimator reads completed-step timing from `main.log`. It reads non-padding
microbatch tokens from `actor.log` and selects one representative rank per
context-parallel group, preventing duplicated token counts. For the current
`d2c2` layout, those ranks are 0 and 2. Complete cycle time includes rollout,
training, weight synchronization, waiting, and other logged time. Repeat
`--exclude-training-step` to omit known startup or invalid steps.

### Inference metric

The reported inference value is cumulative, idle-inclusive conventional 2N MFU:

```text
inference MFU = 2 * N * (prompt tokens + generated tokens)
                / sum(SGLang GPU process seconds * peak FLOP/s/GPU)
```

The token counts come from SGLang's Prometheus counters. Process elapsed time
comes from read-only `ps` output. This denominator includes SGLang startup,
idle periods, and pauses for weight updates.

### Peak FLOPs

Pass the dense peak for the datatype actually used by the model kernels. This
run uses BF16 on H100 SXM GPUs, so the value is **989 TFLOP/s per GPU**. Do not
use the FP32 CUDA-core peak or the 2:1 structured-sparsity figure unless those
actually describe the workload.

## Important limitations

- The model parameter calculation supports the Qwen/Llama-style decoder shape.
  Use `--matrix-parameters` for an architecture with different projections.
- The 6N/2N convention does not add sequence-length-dependent attention FLOPs.
  This keeps the training and inference headline metrics conventional and
  comparable, but long-context workloads perform more real work than it counts.
- Only fully logged training steps are included. An in-flight step enters the
  estimate after its completion record is written.
- SGLang counters and process elapsed time must cover the same process lifetime.
  Restarted/reset counters paired with an older process start time would
  underestimate inference MFU.
