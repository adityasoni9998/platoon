#!/usr/bin/env python3
"""Summarize opt-in inference JSONL traces, or inspect one request's timeline.

Examples (run from the repository root):
  python scripts/profile_inference.py /path/to/trial/profiling/trace-*.jsonl
  python scripts/profile_inference.py /path/to/trace.jsonl --trace-id TRACE_ID

Times are local monotonic durations. Nested spans overlap: do not add their means.
HTTP wait includes remote work; it is not a measurement of network latency alone.
Dropped events make traces incomplete and are reported explicitly.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(paths, trace_id=None):
    timings = defaultdict(list)
    starts = {}
    backend_ids = defaultdict(set)
    requests = []
    timeline = []
    dropped = 0
    peak_requests = 0
    errors = 0
    incomplete_lines = 0
    for path in paths:
        file_dropped = 0
        with Path(path).open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    incomplete_lines += 1
                    continue
                trace = record.get("trace_id")
                if trace_id and trace == trace_id:
                    timeline.append(record)
                if record.get("tinker_request_id") and trace:
                    backend_ids[trace].add(record["tinker_request_id"])
                event = record.get("event")
                if event == "span_start":
                    starts[(str(path), record["span_id"])] = record
                elif event == "span_end":
                    start = starts.pop((str(path), record["span_id"]), {})
                    name = record["span"]
                    if name == "sdk.http":
                        name += " " + start.get("path", "unknown")
                    if name == "training.wait":
                        name += " " + start.get("phase", "unknown")
                    timings[name].append(record["duration_s"])
                    errors += record.get("error_type") is not None
                    if name == "proxy.total":
                        requests.append((record["duration_s"], trace))
                elif event == "thread_queue":
                    timings[record["stage"] + ".queue"].append(record["wait_s"])
                elif event == "sdk_scheduled" and record.get("wait_s") is not None:
                    timings["sdk.scheduling_wait"].append(record["wait_s"])
                elif event == "sdk_limit_wait":
                    timings[record["stage"]].append(record["wait_s"])
                elif event == "loop_lag":
                    timings["loop_lag." + record["loop"]].append(record["lag_s"])
                elif event in ("writer_status", "writer_closed"):
                    file_dropped = max(file_dropped, record["dropped"])
                peak_requests = max(peak_requests, record.get("active_requests", 0))
        dropped += file_dropped
    if trace_id:
        timeline.sort(key=lambda r: r["mono_s"])
        if not timeline:
            raise SystemExit(f"No records for trace {trace_id}")
        origin = timeline[0]["mono_s"]
        for record in timeline:
            print(json.dumps({"offset_s": round(record["mono_s"] - origin, 6), **record}))
        return
    print(
        f"Peak active proxy requests: {peak_requests}; dropped events: {dropped}; "
        f"error/cancelled spans: {errors}; unclosed spans: {len(starts)}; partial/invalid lines: {incomplete_lines}"
    )
    print(
        f"{'Stage (nested spans overlap)':<60} {'Count':>7} {'Mean ms':>10} "
        f"{'P50 ms':>10} {'P95 ms':>10} {'Max ms':>10}"
    )
    for name, values in sorted(timings.items()):
        values.sort()
        print(
            f"{name:<60} {len(values):7d} {statistics.mean(values) * 1000:10.2f} "
            f"{statistics.median(values) * 1000:10.2f} {values[round((len(values) - 1) * 0.95)] * 1000:10.2f} "
            f"{values[-1] * 1000:10.2f}"
        )
    print("\nSlowest proxy requests (seconds, trace ID, Tinker request IDs):")
    for duration, trace in sorted(requests, reverse=True)[:10]:
        print(f"{duration:9.3f}  {trace}  {','.join(sorted(backend_ids[trace])) or 'not recorded'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--trace-id")
    args = parser.parse_args()
    summarize(args.paths, args.trace_id)


if __name__ == "__main__":
    main()
