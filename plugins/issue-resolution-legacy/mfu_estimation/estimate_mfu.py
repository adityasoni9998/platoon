#!/usr/bin/env python3
"""Estimate idle-inclusive training and inference MFU from existing logs.

This program is intentionally read-only with respect to the trainer. It reads
AReaL log files, process metadata from ``ps``, and SGLang Prometheus counters.
It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"


def load_model_config(path: Path | None, model_id: str | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        return json.loads(path.read_text()), str(path)
    if model_id is None:
        raise ValueError("Provide --model-config, --model-id, or --matrix-parameters")
    quoted_model_id = urllib.parse.quote(model_id, safe="/")
    url = f"https://huggingface.co/{quoted_model_id}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode()), model_id


def matrix_parameter_count(config: dict[str, Any]) -> int:
    """Return matmul weights used by a Llama/Qwen-style decoder.

    Norms and biases are deliberately excluded. The input embedding lookup is
    not a matmul; one vocabulary projection is included for the LM head even
    when its weight is tied to the embedding.
    """

    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Model config is missing: {', '.join(missing)}")

    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    layers = int(config["num_hidden_layers"])
    attention_heads = int(config["num_attention_heads"])
    key_value_heads = int(config.get("num_key_value_heads", attention_heads))
    head_dim = int(config.get("head_dim", hidden // attention_heads))
    vocab = int(config["vocab_size"])

    query_width = attention_heads * head_dim
    key_value_width = key_value_heads * head_dim
    attention = hidden * query_width + 2 * hidden * key_value_width + query_width * hidden
    gated_mlp = 3 * hidden * intermediate
    lm_head = hidden * vocab
    return layers * (attention + gated_mlp) + lm_head


def parse_step_metrics(main_log: Path) -> dict[int, dict[str, float]]:
    text = ANSI_RE.sub("", main_log.read_text(errors="replace"))
    markers = list(re.finditer(r"Train step (\d+)/\d+ done\.", text))
    result: dict[int, dict[str, float]] = {}
    cell_re = re.compile(rf"│\s*([^│]+?)\s*│\s*({NUMBER_RE})\s*(?=│)", re.I)
    for index, marker in enumerate(markers):
        step = int(marker.group(1))
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        metrics = {
            key.strip(): float(value)
            for key, value in cell_re.findall(text[marker.end() : end])
        }
        result[step] = metrics
    return result


def parse_microbatches(actor_log: Path) -> dict[int, dict[int, tuple[list[int], list[int]]]]:
    text = ANSI_RE.sub("", actor_log.read_text(errors="replace")).replace("\r", "\n")
    version: int | None = None
    result: dict[int, dict[int, tuple[list[int], list[int]]]] = defaultdict(dict)
    for line in text.splitlines():
        version_match = re.search(r"current_version=(\d+)", line)
        if version_match:
            version = int(version_match.group(1))
        batch_match = re.search(
            r"Microbatch #tokens \(rank (\d+)\): (\[[^]]+\]), padded to: (\[[^]]+\])",
            line,
        )
        if version is not None and batch_match:
            rank = int(batch_match.group(1))
            tokens = ast.literal_eval(batch_match.group(2))
            padded = ast.literal_eval(batch_match.group(3))
            result[version][rank] = (tokens, padded)
    return dict(result)


def training_estimate(
    log_dir: Path,
    matrix_parameters: int,
    train_gpus: int,
    context_parallel_size: int,
    gpu_peak_tflops: float,
    excluded_steps: set[int],
    representative_ranks: list[int] | None,
) -> dict[str, Any]:
    stats = parse_step_metrics(log_dir / "main.log")
    batches = parse_microbatches(log_dir / "actor.log")
    peak_flops_per_second = train_gpus * gpu_peak_tflops * 1e12
    expected_representatives = train_gpus // context_parallel_size
    if train_gpus % context_parallel_size:
        raise ValueError("--train-gpus must be divisible by --context-parallel-size")

    steps: list[dict[str, Any]] = []
    skipped: dict[int, str] = {}
    for step, metrics in sorted(stats.items()):
        if step in excluded_steps:
            skipped[step] = "explicitly excluded"
            continue
        ranks = batches.get(step - 1, {})
        selected_ranks = representative_ranks
        if selected_ranks is None:
            selected_ranks = [
                rank for rank in sorted(ranks) if rank % context_parallel_size == 0
            ]
        if len(selected_ranks) != expected_representatives:
            skipped[step] = (
                f"expected {expected_representatives} representative ranks, "
                f"found {len(selected_ranks)}"
            )
            continue
        if any(rank not in ranks for rank in selected_ranks):
            skipped[step] = "one or more representative ranks are absent from actor.log"
            continue
        if "timeperf/train_step" not in metrics:
            skipped[step] = "timeperf/train_step is absent from main.log"
            continue

        tokens = sum(sum(ranks[rank][0]) for rank in selected_ranks)
        padded_tokens = sum(sum(ranks[rank][1]) for rank in selected_ranks)
        train_seconds = metrics["timeperf/train_step"]
        cycle_seconds = sum(
            value for key, value in metrics.items() if key.startswith("timeperf/")
        )
        useful_flops = 6.0 * matrix_parameters * tokens
        steps.append(
            {
                "step": step,
                "tokens": tokens,
                "padded_tokens": padded_tokens,
                "padding_fraction": (padded_tokens - tokens) / padded_tokens,
                "train_seconds": train_seconds,
                "cycle_seconds_including_idle": cycle_seconds,
                "active_6N_mfu": useful_flops / (peak_flops_per_second * train_seconds),
                "end_to_end_6N_mfu": useful_flops
                / (peak_flops_per_second * cycle_seconds),
            }
        )

    if not steps:
        reasons = "; ".join(f"step {step}: {reason}" for step, reason in skipped.items())
        raise RuntimeError(f"No usable completed training steps. {reasons}")

    tokens = sum(item["tokens"] for item in steps)
    padded_tokens = sum(item["padded_tokens"] for item in steps)
    train_seconds = sum(item["train_seconds"] for item in steps)
    cycle_seconds = sum(item["cycle_seconds_including_idle"] for item in steps)
    useful_flops = 6.0 * matrix_parameters * tokens
    return {
        "basis": "all usable completed steps after exclusions",
        "included_steps": [item["step"] for item in steps],
        "excluded_or_skipped_steps": skipped,
        "tokens": tokens,
        "padded_tokens": padded_tokens,
        "padding_fraction": (padded_tokens - tokens) / padded_tokens,
        "active_seconds": train_seconds,
        "cycle_seconds_including_idle": cycle_seconds,
        "standard_6N_active_mfu": useful_flops / (peak_flops_per_second * train_seconds),
        "standard_6N_end_to_end_mfu": useful_flops
        / (peak_flops_per_second * cycle_seconds),
        "steps": steps,
    }


def discover_sglang(process_match: str | None) -> list[dict[str, Any]]:
    output = subprocess.check_output(["ps", "-eo", "pid=,etimes=,args="], text=True)
    servers: list[dict[str, Any]] = []
    for line in output.splitlines():
        if "sglang.launch_server" not in line:
            continue
        if process_match is not None and process_match not in line:
            continue
        prefix = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        endpoint = re.search(r"--host\s+(\S+)\s+--port\s+(\d+)", line)
        if prefix and endpoint:
            servers.append(
                {
                    "pid": int(prefix.group(1)),
                    "elapsed_seconds": int(prefix.group(2)),
                    "url": f"http://{endpoint.group(1)}:{endpoint.group(2)}/metrics",
                }
            )
    return servers


def prometheus_value(metrics_text: str, metric: str) -> float:
    values = []
    for line in metrics_text.splitlines():
        if line.startswith(metric + "{") or line.startswith(metric + " "):
            values.append(float(line.rsplit(None, 1)[1]))
    if not values:
        raise KeyError(f"Prometheus metric not found: {metric}")
    return sum(values)


def inference_estimate(
    matrix_parameters: int,
    gpu_peak_tflops: float,
    process_match: str | None,
    explicit_urls: list[str],
    expected_servers: int | None,
) -> dict[str, Any]:
    discovered = discover_sglang(process_match)
    if explicit_urls:
        by_url = {server["url"]: server for server in discovered}
        missing = [url for url in explicit_urls if url not in by_url]
        if missing:
            raise RuntimeError(
                "Cannot determine process elapsed time for explicit URL(s): "
                + ", ".join(missing)
            )
        servers = [by_url[url] for url in explicit_urls]
    else:
        servers = discovered
    if not servers:
        raise RuntimeError("No SGLang launch_server processes found")
    if expected_servers is not None and len(servers) != expected_servers:
        raise RuntimeError(f"Expected {expected_servers} SGLang servers, found {len(servers)}")

    prompt_tokens = 0.0
    generation_tokens = 0.0
    requests = 0.0
    gpu_seconds = 0.0
    for server in servers:
        with urllib.request.urlopen(server["url"], timeout=5) as response:
            metrics_text = response.read().decode()
        server_prompt = prometheus_value(metrics_text, "sglang:prompt_tokens_total")
        server_generation = prometheus_value(metrics_text, "sglang:generation_tokens_total")
        server_requests = prometheus_value(metrics_text, "sglang:num_requests_total")
        server.update(
            {
                "prompt_tokens": server_prompt,
                "generation_tokens": server_generation,
                "requests": server_requests,
            }
        )
        prompt_tokens += server_prompt
        generation_tokens += server_generation
        requests += server_requests
        gpu_seconds += server["elapsed_seconds"]

    useful_flops = 2.0 * matrix_parameters * (prompt_tokens + generation_tokens)
    capacity_flops = gpu_peak_tflops * 1e12 * gpu_seconds
    return {
        "basis": "cumulative SGLang counters divided by cumulative GPU process wall time",
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "requests": requests,
        "gpu_seconds_including_idle_and_startup": gpu_seconds,
        "standard_2N_end_to_end_mfu": useful_flops / capacity_flops,
        "servers": servers,
    }


def format_markdown(result: dict[str, Any]) -> str:
    assumptions = result["assumptions"]
    training = result["training"]
    inference = result.get("inference")
    lines = [
        "# MFU estimate",
        "",
        f"Sample time: **{result['sample_time']}**",
        "",
        "| GPU pool | Idle-inclusive MFU |",
        "|---|---:|",
        f"| Training | **{100 * training['standard_6N_end_to_end_mfu']:.2f}%** |",
    ]
    if inference is not None:
        lines.append(f"| Inference | **{100 * inference['standard_2N_end_to_end_mfu']:.2f}%** |")
    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            f"- Matrix parameters, `N`: **{assumptions['matrix_parameters']:,}**",
            f"- Peak per GPU: **{assumptions['gpu_peak_tflops']:,.0f} TFLOP/s**",
            f"- Training GPUs: **{assumptions['train_gpus']}**",
            f"- Training pool peak: **{assumptions['training_pool_peak_petaflops']:.3f} PFLOP/s**",
            "- Peak is dense BF16 Tensor Core throughput; no sparsity multiplier.",
            "",
            "## Training",
            "",
            f"- Included completed steps: **{training['included_steps']}**",
            f"- Non-padding tokens: **{training['tokens']:,}**",
            f"- Complete cycle time: **{training['cycle_seconds_including_idle']:,.2f} s**",
            f"- Active-only 6N MFU: **{100 * training['standard_6N_active_mfu']:.2f}%**",
            f"- End-to-end 6N MFU: **{100 * training['standard_6N_end_to_end_mfu']:.2f}%**",
            "",
            "Training uses `6 * N * tokens / (training GPU peak * complete cycle seconds)`.",
        ]
    )
    if inference is not None:
        lines.extend(
            [
                "",
                "## Inference",
                "",
                f"- SGLang servers/GPUs: **{len(inference['servers'])}**",
                f"- Prompt tokens: **{int(inference['prompt_tokens']):,}**",
                f"- Generated tokens: **{int(inference['generation_tokens']):,}**",
                f"- End-to-end 2N MFU: **{100 * inference['standard_2N_end_to_end_mfu']:.2f}%**",
                "",
                "Inference uses `2 * N * (prompt + generated tokens) / cumulative GPU capacity`.",
            ]
        )
    lines.extend(
        [
            "",
            "Both end-to-end metrics include observed startup, waiting, synchronization, and idle time.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-config", type=Path, help="Local Hugging Face config.json")
    model.add_argument("--model-id", help="Hugging Face model ID; downloads config.json")
    model.add_argument("--matrix-parameters", type=int, help="Use a precomputed N directly")
    parser.add_argument("--train-gpus", type=int, required=True)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--representative-ranks", help="Comma-separated ranks, e.g. 0,2")
    parser.add_argument(
        "--gpu-peak-tflops",
        type=float,
        required=True,
        help="Dense peak for the actual training/inference dtype, per GPU",
    )
    parser.add_argument(
        "--exclude-training-step",
        type=int,
        action="append",
        default=[],
        help="May be repeated; useful for excluding startup step 1",
    )
    parser.add_argument("--training-only", action="store_true")
    parser.add_argument("--sglang-process-match")
    parser.add_argument(
        "--sglang-metrics-url",
        action="append",
        default=[],
        help="Restrict inference to this discovered URL; may be repeated",
    )
    parser.add_argument("--expected-sglang-servers", type=int)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.matrix_parameters is not None:
        matrix_parameters = args.matrix_parameters
        model_source = "command-line --matrix-parameters"
    else:
        model_config, model_source = load_model_config(args.model_config, args.model_id)
        matrix_parameters = matrix_parameter_count(model_config)

    ranks = None
    if args.representative_ranks:
        ranks = [int(value) for value in args.representative_ranks.split(",")]

    result: dict[str, Any] = {
        "sample_time": dt.datetime.fromtimestamp(time.time()).astimezone().isoformat(),
        "assumptions": {
            "model_source": model_source,
            "matrix_parameters": matrix_parameters,
            "gpu_peak_tflops": args.gpu_peak_tflops,
            "train_gpus": args.train_gpus,
            "training_pool_peak_petaflops": args.train_gpus
            * args.gpu_peak_tflops
            / 1000.0,
            "context_parallel_size": args.context_parallel_size,
        },
        "training": training_estimate(
            log_dir=args.log_dir,
            matrix_parameters=matrix_parameters,
            train_gpus=args.train_gpus,
            context_parallel_size=args.context_parallel_size,
            gpu_peak_tflops=args.gpu_peak_tflops,
            excluded_steps=set(args.exclude_training_step),
            representative_ranks=ranks,
        ),
    }
    if not args.training_only:
        result["inference"] = inference_estimate(
            matrix_parameters=matrix_parameters,
            gpu_peak_tflops=args.gpu_peak_tflops,
            process_match=args.sglang_process_match,
            explicit_urls=args.sglang_metrics_url,
            expected_servers=args.expected_sglang_servers,
        )

    if args.format == "json":
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(format_markdown(result))


if __name__ == "__main__":
    main()
