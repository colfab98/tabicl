#!/usr/bin/env python3
"""Monitor a running TabICL training job and summarize efficiency bottlenecks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


TRAINING_METRIC_RE = re.compile(
    r"(?:\|\s*(?P<step>\d+)/(?P<total>\d+)\s*\[.*?)?"
    r"accuracy=(?P<accuracy>[-+0-9.eE]+),\s*"
    r"ce=(?P<ce>[-+0-9.eE]+),\s*"
    r"prior_time=(?P<prior_time>[-+0-9.eE]+),\s*"
    r"train_time=(?P<train_time>[-+0-9.eE]+)"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: Iterable[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    weight = idx - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def parse_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "[N/A]", "N/A", "Not Supported"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="PID of the training process to monitor")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Training stderr/stdout file to parse for prior_time and train_time",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Sampling interval in seconds for GPU and process snapshots",
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=15,
        help="Print a rolling console summary every N samples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_efficiency_reports"),
        help="Directory where CSV/JSON/Markdown reports will be written",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label used in generated filenames",
    )
    return parser.parse_args()


def run_command(cmd: List[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def read_status_value(pid: int, prefix: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(prefix):
                fields = line.split()
                return int(fields[1])
    except FileNotFoundError:
        return None
    return None


def read_cpu_times(pid: int) -> tuple[int, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except FileNotFoundError:
        return None
    return int(fields[13]), int(fields[14])


def parse_csv_lines(output: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return []
    reader = csv.reader(lines)
    rows = []
    for row in reader:
        rows.append({str(i): cell.strip() for i, cell in enumerate(row)})
    return rows


def query_gpu_rows() -> List[dict]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for row in parse_csv_lines(output):
        rows.append(
            {
                "gpu_index": int(row["0"]),
                "gpu_uuid": row["1"],
                "gpu_name": row["2"],
                "utilization_gpu": parse_float(row["3"]),
                "utilization_memory": parse_float(row["4"]),
                "memory_used_mb": parse_float(row["5"]),
                "memory_total_mb": parse_float(row["6"]),
                "power_draw_w": parse_float(row["7"]),
                "temperature_c": parse_float(row["8"]),
                "sm_clock_mhz": parse_float(row["9"]),
            }
        )
    return rows


def query_compute_rows() -> List[dict]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for row in parse_csv_lines(output):
        gpu_uuid = row["0"]
        if gpu_uuid == "[Not Supported]":
            continue
        rows.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": int(row["1"]),
                "process_name": row["2"],
                "used_gpu_memory_mb": parse_float(row["3"]) or 0.0,
            }
        )
    return rows


@dataclass
class LogMetric:
    observed_at: str
    step: int | None
    total_steps: int | None
    accuracy: float
    ce: float
    prior_time: float
    train_time: float


class TrainingLogParser:
    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.seen = set()
        self.metrics: List[LogMetric] = []

    def poll(self) -> List[LogMetric]:
        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()

        if not chunk:
            return []

        found: List[LogMetric] = []
        for match in TRAINING_METRIC_RE.finditer(chunk.replace("\r", "\n")):
            key = (
                match.group("step"),
                match.group("accuracy"),
                match.group("ce"),
                match.group("prior_time"),
                match.group("train_time"),
            )
            if key in self.seen:
                continue
            self.seen.add(key)
            metric = LogMetric(
                observed_at=utc_now().isoformat(),
                step=int(match.group("step")) if match.group("step") is not None else None,
                total_steps=int(match.group("total")) if match.group("total") is not None else None,
                accuracy=float(match.group("accuracy")),
                ce=float(match.group("ce")),
                prior_time=float(match.group("prior_time")),
                train_time=float(match.group("train_time")),
            )
            self.metrics.append(metric)
            found.append(metric)
        return found


def build_label(args: argparse.Namespace) -> str:
    if args.label:
        return args.label
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pid{args.pid}_{timestamp}"


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(gpu_rows: List[dict], proc_rows: List[dict], log_metrics: List[LogMetric]) -> dict:
    gpu_util = [row["utilization_gpu"] for row in gpu_rows if row["utilization_gpu"] is not None]
    mem_util = [row["utilization_memory"] for row in gpu_rows if row["utilization_memory"] is not None]
    mem_used = [row["memory_used_mb"] for row in gpu_rows if row["memory_used_mb"] is not None]
    power = [row["power_draw_w"] for row in gpu_rows if row["power_draw_w"] is not None]
    temp = [row["temperature_c"] for row in gpu_rows if row["temperature_c"] is not None]

    proc_gpu_mem = [row["used_gpu_memory_mb"] for row in proc_rows]
    proc_cpu = [row["cpu_percent"] for row in proc_rows if row["cpu_percent"] is not None]
    proc_rss = [row["rss_mb"] for row in proc_rows if row["rss_mb"] is not None]

    prior_times = [metric.prior_time for metric in log_metrics]
    train_times = [metric.train_time for metric in log_metrics]

    total_prior = sum(prior_times)
    total_train = sum(train_times)
    total_observed = total_prior + total_train
    prior_share = (total_prior / total_observed) if total_observed else None
    train_share = (total_train / total_observed) if total_observed else None

    latest_step = None
    total_steps = None
    for metric in reversed(log_metrics):
        if metric.step is not None:
            latest_step = metric.step
            total_steps = metric.total_steps
            break

    return {
        "gpu": {
            "samples": len(gpu_rows),
            "avg_utilization_gpu": mean(gpu_util),
            "p95_utilization_gpu": percentile(gpu_util, 0.95),
            "avg_utilization_memory": mean(mem_util),
            "avg_memory_used_mb": mean(mem_used),
            "peak_memory_used_mb": max(mem_used) if mem_used else None,
            "avg_power_draw_w": mean(power),
            "avg_temperature_c": mean(temp),
        },
        "process": {
            "samples": len(proc_rows),
            "avg_gpu_memory_mb": mean(proc_gpu_mem),
            "peak_gpu_memory_mb": max(proc_gpu_mem) if proc_gpu_mem else None,
            "avg_cpu_percent": mean(proc_cpu),
            "peak_cpu_percent": max(proc_cpu) if proc_cpu else None,
            "avg_rss_mb": mean(proc_rss),
            "peak_rss_mb": max(proc_rss) if proc_rss else None,
        },
        "training_log": {
            "observations": len(log_metrics),
            "latest_step": latest_step,
            "total_steps": total_steps,
            "avg_prior_time_s": mean(prior_times),
            "avg_train_time_s": mean(train_times),
            "p95_prior_time_s": percentile(prior_times, 0.95),
            "p95_train_time_s": percentile(train_times, 0.95),
            "total_prior_time_s": total_prior if log_metrics else None,
            "total_train_time_s": total_train if log_metrics else None,
            "prior_share_of_step_time": prior_share,
            "train_share_of_step_time": train_share,
            "avg_accuracy": mean([metric.accuracy for metric in log_metrics]),
            "avg_ce": mean([metric.ce for metric in log_metrics]),
        },
    }


def build_findings(summary: dict) -> List[str]:
    findings: List[str] = []

    gpu_avg = summary["gpu"]["avg_utilization_gpu"]
    prior_share = summary["training_log"]["prior_share_of_step_time"]
    train_share = summary["training_log"]["train_share_of_step_time"]
    cpu_avg = summary["process"]["avg_cpu_percent"]
    gpu_mem_peak = summary["process"]["peak_gpu_memory_mb"]

    if gpu_avg is not None and gpu_avg < 40:
        findings.append(
            f"Average GPU utilization is low at {fmt(gpu_avg)}%, which usually means the accelerator is underfed."
        )
    elif gpu_avg is not None and gpu_avg > 85:
        findings.append(f"Average GPU utilization is strong at {fmt(gpu_avg)}%, so the run looks mostly compute-bound.")

    if prior_share is not None and prior_share > 0.6:
        findings.append(
            f"Prior/data generation accounts for {fmt(prior_share * 100)}% of observed step time, which points to a CPU/input bottleneck."
        )
    elif train_share is not None and train_share > 0.7:
        findings.append(
            f"Model compute accounts for {fmt(train_share * 100)}% of observed step time, so optimization should focus on kernels, precision, or batch sizing."
        )

    if cpu_avg is not None and cpu_avg > 85:
        findings.append(
            f"The monitored process averages {fmt(cpu_avg)}% CPU usage, reinforcing that host-side work is a meaningful part of the runtime."
        )

    if gpu_mem_peak is not None and gpu_mem_peak < 0.35 * (summary["gpu"]["peak_memory_used_mb"] or gpu_mem_peak):
        findings.append(
            "Only a small share of visible GPU memory appears tied to the training process, so background GPU consumers may be present."
        )

    if not findings:
        findings.append("No single dominant bottleneck stands out from the collected samples.")

    return findings


def render_report(label: str, pid: int, cmdline: str, summary: dict, findings: List[str]) -> str:
    lines = [
        f"# Training efficiency report: {label}",
        "",
        f"- PID: `{pid}`",
        f"- Command: `{cmdline or 'n/a'}`",
        "",
        "## Summary",
        "",
        f"- GPU avg utilization: `{fmt(summary['gpu']['avg_utilization_gpu'])}%`",
        f"- GPU p95 utilization: `{fmt(summary['gpu']['p95_utilization_gpu'])}%`",
        f"- GPU avg memory used: `{fmt(summary['gpu']['avg_memory_used_mb'])} MB`",
        f"- GPU peak memory used: `{fmt(summary['gpu']['peak_memory_used_mb'])} MB`",
        f"- Process avg CPU: `{fmt(summary['process']['avg_cpu_percent'])}%`",
        f"- Process peak GPU memory: `{fmt(summary['process']['peak_gpu_memory_mb'])} MB`",
        f"- Avg prior_time: `{fmt(summary['training_log']['avg_prior_time_s'])} s`",
        f"- Avg train_time: `{fmt(summary['training_log']['avg_train_time_s'])} s`",
        f"- Prior share of step time: `{fmt(None if summary['training_log']['prior_share_of_step_time'] is None else summary['training_log']['prior_share_of_step_time'] * 100)}%`",
        f"- Train share of step time: `{fmt(None if summary['training_log']['train_share_of_step_time'] is None else summary['training_log']['train_share_of_step_time'] * 100)}%`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    label = build_label(args)
    output_dir = ensure_output_dir(args.output_dir / label)

    log_parser = TrainingLogParser(args.log_file) if args.log_file is not None else None
    gpu_samples: List[dict] = []
    proc_samples: List[dict] = []
    last_cpu_times = read_cpu_times(args.pid)
    last_cpu_wall = time.monotonic()
    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cmdline = read_cmdline(args.pid)
    stop_requested = False

    def handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\nStopping monitor after signal {signum}...", file=sys.stderr)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Monitoring PID {args.pid} every {args.interval:.1f}s", file=sys.stderr)
    if args.log_file is not None:
        print(f"Parsing training log {args.log_file}", file=sys.stderr)
        log_parser.poll()

    sample_idx = 0
    while not stop_requested and pid_exists(args.pid):
        sample_time = utc_now().isoformat()
        try:
            gpu_rows = query_gpu_rows()
            compute_rows = query_compute_rows()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"Failed to query GPU metrics: {exc}", file=sys.stderr)
            break

        training_compute = [row for row in compute_rows if row["pid"] == args.pid]
        gpu_uuid_to_index = {row["gpu_uuid"]: row["gpu_index"] for row in gpu_rows}

        for row in gpu_rows:
            gpu_samples.append({"sample_time": sample_time, **row})

        current_cpu_times = read_cpu_times(args.pid)
        rss_kb = read_status_value(args.pid, "VmRSS:")
        thread_count = read_status_value(args.pid, "Threads:")
        cpu_percent = None
        if current_cpu_times is not None and last_cpu_times is not None:
            delta_ticks = (current_cpu_times[0] + current_cpu_times[1]) - (last_cpu_times[0] + last_cpu_times[1])
            delta_time = time.monotonic() - last_cpu_wall
            if delta_time > 0:
                cpu_percent = 100.0 * (delta_ticks / clk_tck) / delta_time
        last_cpu_times = current_cpu_times
        last_cpu_wall = time.monotonic()

        if training_compute:
            for row in training_compute:
                proc_samples.append(
                    {
                        "sample_time": sample_time,
                        "pid": args.pid,
                        "gpu_index": gpu_uuid_to_index.get(row["gpu_uuid"]),
                        "process_name": row["process_name"],
                        "used_gpu_memory_mb": row["used_gpu_memory_mb"],
                        "cpu_percent": cpu_percent,
                        "rss_mb": None if rss_kb is None else rss_kb / 1024,
                        "threads": thread_count,
                    }
                )
        else:
            proc_samples.append(
                {
                    "sample_time": sample_time,
                    "pid": args.pid,
                    "gpu_index": None,
                    "process_name": Path(cmdline.split()[0]).name if cmdline else None,
                    "used_gpu_memory_mb": 0.0,
                    "cpu_percent": cpu_percent,
                    "rss_mb": None if rss_kb is None else rss_kb / 1024,
                    "threads": thread_count,
                }
            )

        new_metrics = log_parser.poll() if log_parser is not None else []
        sample_idx += 1

        if args.summary_every > 0 and sample_idx % args.summary_every == 0:
            summary = summarize(gpu_samples, proc_samples, log_parser.metrics if log_parser is not None else [])
            latest_gpu = summary["gpu"]["avg_utilization_gpu"]
            latest_prior = summary["training_log"]["avg_prior_time_s"]
            latest_train = summary["training_log"]["avg_train_time_s"]
            print(
                f"[sample {sample_idx}] avg_gpu={fmt(latest_gpu)}% avg_prior={fmt(latest_prior)}s "
                f"avg_train={fmt(latest_train)}s new_log_metrics={len(new_metrics)}",
                file=sys.stderr,
            )

        time.sleep(args.interval)

    log_metrics = log_parser.metrics if log_parser is not None else []
    summary = summarize(gpu_samples, proc_samples, log_metrics)
    findings = build_findings(summary)
    report = render_report(label, args.pid, cmdline, summary, findings)

    write_csv(output_dir / "gpu_samples.csv", gpu_samples)
    write_csv(output_dir / "process_samples.csv", proc_samples)
    write_csv(
        output_dir / "training_log_metrics.csv",
        [
            {
                "observed_at": metric.observed_at,
                "step": metric.step,
                "total_steps": metric.total_steps,
                "accuracy": metric.accuracy,
                "ce": metric.ce,
                "prior_time": metric.prior_time,
                "train_time": metric.train_time,
            }
            for metric in log_metrics
        ],
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Artifacts written to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
