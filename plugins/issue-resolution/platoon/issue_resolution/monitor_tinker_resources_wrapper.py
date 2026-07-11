"""Standalone periodic resource monitor for issue-resolution Tinker runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


METADATA_KEYS = {"timestamp", "timestamp_epoch", "monitor_user"}


@dataclass(frozen=True)
class MonitorPaths:
    output_dir: Path
    jsonl_path: Path
    png_path: Path


def default_output_dir() -> Path:
    return Path(
        "/data/user_data/adityabs/platoon_skyrl_tinker/logs/"
        "issue_resolution-platoon-tinker/"
        "skyrl-flame-swerebench-fft-cispo-20260710-212549"
    )


def human_gib_from_kib(kib: int | float) -> float:
    return round(float(kib) / 1024.0 / 1024.0, 1)


def percent(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), 1)


def human_duration(seconds: int) -> str:
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f"{days}d{hours:02d}h{minutes:02d}m{seconds:02d}s"
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def run_command(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def parse_int(value: str) -> int:
    return int(float(value))


def count_trainer_fds(trainer_pids: str) -> int:
    total = 0
    for pid_text in trainer_pids.split(","):
        pid_text = pid_text.strip()
        if not pid_text:
            continue
        fd_dir = Path("/proc") / pid_text / "fd"
        if not fd_dir.exists():
            continue
        try:
            total += sum(1 for child in fd_dir.iterdir() if child.is_symlink())
        except OSError:
            continue
    return total


def summarize_ps(monitor_user: str) -> dict[str, Any]:
    output = run_command(
        ["ps", "-u", monitor_user, "-o", "pid=,stat=,nlwp=,pcpu=,rss=,etimes=,comm=,args="]
    )
    procs = threads = rss_kib = zombies = dstate = 0
    apptainer = starter = squash = overlay = agent = tmux = bash = 0
    apptainer_runtime_sum = apptainer_runtime_max = 0
    trainer_threads = trainer_rss_kib = 0
    total_pcpu = trainer_cpu = 0.0
    trainer_pids: list[str] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 7:
            continue
        pid, stat, nlwp, pcpu, rss, elapsed, comm = parts[:7]
        args = parts[7] if len(parts) > 7 else ""

        procs += 1
        threads += int(nlwp)
        total_pcpu += float(pcpu)
        rss_kib += int(rss)

        if stat.startswith("Z"):
            zombies += 1
        if stat.startswith("D"):
            dstate += 1
        if comm == "apptainer" and "apptainer run" in raw_line:
            apptainer += 1
            apptainer_runtime_sum += int(elapsed)
            apptainer_runtime_max = max(apptainer_runtime_max, int(elapsed))
        if comm == "starter":
            starter += 1
        if comm == "squashfuse_ll":
            squash += 1
        if comm == "fuse-overlayfs":
            overlay += 1
        if comm == "tmux:":
            tmux += 1
        if comm == "bash":
            bash += 1
        if comm == "python" and "openhands.agent_server" in raw_line:
            agent += 1
        if comm == "python3" and "platoon.issue_resolution.train_tinker" in raw_line:
            trainer_threads += int(nlwp)
            trainer_cpu += float(pcpu)
            trainer_rss_kib += int(rss)
            trainer_pids.append(pid)

    apptainer_runtime_avg_sec = int(apptainer_runtime_sum / apptainer) if apptainer else 0
    return {
        "procs": procs,
        "threads": threads,
        "cpu_cores": round(total_pcpu / 100.0, 1),
        "rss_kib": rss_kib,
        "zombies": zombies,
        "dstate": dstate,
        "apptainer": apptainer,
        "apptainer_runtime_avg_sec": apptainer_runtime_avg_sec,
        "apptainer_runtime_max_sec": apptainer_runtime_max,
        "starter": starter,
        "squash": squash,
        "overlay": overlay,
        "agent": agent,
        "tmux": tmux,
        "bash": bash,
        "trainer_pids": ",".join(trainer_pids),
        "trainer_threads": trainer_threads,
        "trainer_cpu_cores": round(trainer_cpu / 100.0, 1),
        "trainer_rss_kib": trainer_rss_kib,
    }


def summarize_lwps(monitor_user: str) -> dict[str, int]:
    output = run_command(["ps", "-u", monitor_user, "-L", "-o", "stat="])
    runnable = zombies = dstate = 0
    for raw_line in output.splitlines():
        stat = raw_line.strip()
        if not stat:
            continue
        if stat.startswith("R"):
            runnable += 1
        if stat.startswith("Z"):
            zombies += 1
        if stat.startswith("D"):
            dstate += 1
    return {
        "lwp_runnable": runnable,
        "lwp_zombie": zombies,
        "lwp_dstate": dstate,
    }


def summarize_vmstat() -> dict[str, int]:
    output = run_command(["vmstat", "1", "2"])
    lines = [line for line in output.splitlines() if line.strip()]
    fields = lines[-1].split()
    return {
        "runq": parse_int(fields[0]),
        "blocked": parse_int(fields[1]),
        "swap_kib": parse_int(fields[2]),
        "free_kib": parse_int(fields[3]),
        "buff_cache_kib": parse_int(fields[4]) + parse_int(fields[5]),
        "cpu_user": parse_int(fields[12]),
        "cpu_sys": parse_int(fields[13]),
        "cpu_idle": parse_int(fields[14]),
        "cpu_wait": parse_int(fields[15]),
    }


def summarize_iostat() -> dict[str, Any]:
    if shutil.which("iostat") is None:
        return {
            "disk": "na",
            "disk_util": None,
            "disk_await": None,
        }

    output = run_command(["iostat", "-xz", "1", "2"])
    sample = 0
    max_util = 0.0
    max_await = 0.0
    hot = "none"

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Device"):
            sample += 1
            continue
        fields = line.split()
        if sample < 2 or len(fields) < 23:
            continue
        read_await = float(fields[5])
        write_await = float(fields[11])
        await_ms = max(read_await, write_await)
        util = float(fields[-1])
        if util > max_util:
            max_util = util
            max_await = await_ms
            hot = fields[0]

    return {
        "disk": hot,
        "disk_util": round(max_util, 1),
        "disk_await": round(max_await, 1),
    }


def read_mem_total_kib() -> int:
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def read_online_cpus() -> int:
    try:
        return int(run_command(["getconf", "_NPROCESSORS_ONLN"]).strip())
    except Exception:
        return os.cpu_count() or 1


def collect_metrics(monitor_user: str) -> dict[str, Any]:
    record: dict[str, Any] = {}
    record.update(summarize_ps(monitor_user))
    record.update(summarize_lwps(monitor_user))
    record.update(summarize_vmstat())
    record.update(summarize_iostat())

    mem_total_kib = read_mem_total_kib()
    online_cpus = read_online_cpus()
    trainer_fds = count_trainer_fds(record["trainer_pids"])

    timestamp = datetime.now(tz=UTC)
    record.update(
        {
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "timestamp_epoch": timestamp.timestamp(),
            "monitor_user": monitor_user,
            "rss_gib": human_gib_from_kib(record["rss_kib"]),
            "trainer_rss_gib": human_gib_from_kib(record["trainer_rss_kib"]),
            "free_gib": human_gib_from_kib(record["free_kib"]),
            "buff_cache_gib": human_gib_from_kib(record["buff_cache_kib"]),
            "swap_gib": human_gib_from_kib(record["swap_kib"]),
            "trainer_fds": trainer_fds,
            "cpu_util_pct": percent(record["cpu_cores"], online_cpus),
            "ram_util_pct": percent(record["rss_kib"], mem_total_kib),
            "apptainer_runtime_avg": human_duration(record["apptainer_runtime_avg_sec"]),
            "apptainer_runtime_max": human_duration(record["apptainer_runtime_max_sec"]),
        }
    )
    return record


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect issue-resolution Tinker resource metrics every N seconds, append JSONL "
            "records, and refresh a multi-subplot PNG."
        )
    )
    parser.add_argument(
        "--monitor-user",
        default=os.environ.get("MONITOR_USER") or os.environ.get("USER") or "",
        help="User whose processes should be monitored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory where monitor_resources.jsonl and monitor_resources.png are written.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help="Override the JSONL output path.",
    )
    parser.add_argument(
        "--png-path",
        type=Path,
        default=None,
        help="Override the PNG output path.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Optional number of samples to collect before exiting.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable PNG generation. JSONL logging still runs.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> MonitorPaths:
    output_dir = args.output_dir
    jsonl_path = args.jsonl_path or output_dir / "monitor_resources.jsonl"
    png_path = args.png_path or output_dir / "monitor_resources.png"
    return MonitorPaths(output_dir=output_dir, jsonl_path=jsonl_path, png_path=png_path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def numeric_metric_names(records: list[dict[str, Any]]) -> list[str]:
    metric_names: set[str] = set()
    for record in records:
        for key, value in record.items():
            if key in METADATA_KEYS or isinstance(value, str):
                continue
            if value is None or isinstance(value, (int, float, bool)):
                metric_names.add(key)
    return sorted(metric_names)


def configure_matplotlib(output_dir: Path) -> None:
    mplconfigdir = output_dir / ".mplconfig"
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))


def plot_specs() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("owned_threads_plus_processes", [("owned_threads_plus_processes", "owned threads + processes")]),
        ("zombies", [("zombies", "zombies")]),
        ("cpu_util_pct", [("cpu_util_pct", "CPU core util %")]),
        ("ram_util_pct", [("ram_util_pct", "RAM util %")]),
        (
            "apptainer_and_agent_server",
            [
                ("apptainer", "active apptainer"),
                ("agent", "agent_server"),
            ],
        ),
        (
            "apptainer_runtime_sec",
            [
                ("apptainer_runtime_avg_sec", "avg runtime"),
                ("apptainer_runtime_max_sec", "max runtime"),
            ],
        ),
    ]


def plot_records(records: list[dict[str, Any]], png_path: Path) -> None:
    if not records:
        return

    configure_matplotlib(png_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = plot_specs()
    if not specs:
        return

    for record in records:
        record["owned_threads_plus_processes"] = record.get("threads", 0) + record.get("procs", 0)

    timestamps = [datetime.fromtimestamp(float(record["timestamp_epoch"]), tz=UTC) for record in records]
    cols = 2
    rows = math.ceil(len(specs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, max(rows * 2.8, 3.5)), squeeze=False)
    flat_axes = axes.flatten()

    for axis, (title, series_specs) in zip(flat_axes, specs):
        for metric_name, label in series_specs:
            series = []
            for record in records:
                value = record.get(metric_name)
                if value is None:
                    series.append(float("nan"))
                elif isinstance(value, bool):
                    series.append(int(value))
                else:
                    series.append(float(value))
            axis.plot(timestamps, series, linewidth=1.5, label=label)
        axis.set_title(title, fontsize=10)
        axis.grid(True, alpha=0.3)
        axis.set_xticks([])
        if len(series_specs) > 1:
            axis.legend(fontsize=8)

    for axis in flat_axes[len(specs):]:
        axis.set_visible(False)

    fig.suptitle("Tinker Resource Metrics", fontsize=14)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def monitor_loop(
    monitor_user: str,
    jsonl_path: Path,
    png_path: Path,
    interval_seconds: float,
    count: int | None,
    plot_enabled: bool,
) -> int:
    samples_taken = 0
    while count is None or samples_taken < count:
        sample_started_at = time.monotonic()
        record = collect_metrics(monitor_user)
        append_jsonl(jsonl_path, record)
        if plot_enabled:
            plot_records(load_records(jsonl_path), png_path)

        samples_taken += 1
        if count is not None and samples_taken >= count:
            break

        sleep_for = interval_seconds - (time.monotonic() - sample_started_at)
        if sleep_for > 0:
            time.sleep(sleep_for)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.monitor_user:
        raise SystemExit("Unable to determine monitor user; pass --monitor-user.")

    paths = resolve_paths(args)
    return monitor_loop(
        monitor_user=args.monitor_user,
        jsonl_path=paths.jsonl_path,
        png_path=paths.png_path,
        interval_seconds=args.interval_seconds,
        count=args.count,
        plot_enabled=not args.no_plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
