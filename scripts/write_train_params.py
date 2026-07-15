#!/usr/bin/env python3
"""Record the exact training command and run metadata before launching a model."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NUMERIC_INT = re.compile(r"^[+-]?\d+$")
NUMERIC_FLOAT = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)$|^[+-]?(?:\d+\.\d*|\.\d+)$"
)


def coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    if NUMERIC_INT.match(value):
        return int(value)
    if NUMERIC_FLOAT.match(value):
        return float(value)
    return value


def simplify_values(values: list[str]) -> Any:
    if not values:
        return True
    converted = [coerce_value(value) for value in values]
    if len(converted) == 1:
        return converted[0]
    return converted


def store_param(params: dict[str, Any], key: str, value: Any) -> None:
    if key not in params:
        params[key] = value
        return
    if not isinstance(params[key], list) or (
        params[key] and not isinstance(params[key][0], list)
    ):
        params[key] = [params[key]]
    params[key].append(value)


def parse_command(command: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    params_raw: dict[str, Any] = {}
    positional: list[str] = []
    command_prefix: list[str] = []
    seen_first_option = False

    i = 0
    while i < len(command):
        token = command[i]
        if token == "--":
            i += 1
            continue
        if token.startswith("--") and token != "--":
            seen_first_option = True
            raw_key = token[2:]
            values: list[str] = []
            if "=" in raw_key:
                raw_key, value = raw_key.split("=", 1)
                values = [value]
                i += 1
            else:
                i += 1
                while i < len(command) and not command[i].startswith("--"):
                    values.append(command[i])
                    i += 1

            key = raw_key.replace("-", "_")
            store_param(params, key, simplify_values(values))
            store_param(params_raw, key, True if not values else values[0] if len(values) == 1 else values)
            continue

        if seen_first_option:
            positional.append(token)
        else:
            command_prefix.append(token)
        i += 1

    return {
        "params": params,
        "params_raw": params_raw,
        "command_prefix": command_prefix,
        "positional_after_options": positional,
    }


def run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def collect_git_info(cwd: Path) -> dict[str, Any]:
    status = run_git(["status", "--short"], cwd)
    return {
        "root": run_git(["rev-parse", "--show-toplevel"], cwd),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "commit": run_git(["rev-parse", "HEAD"], cwd),
        "dirty": bool(status),
        "status_short": status or "",
    }


def collect_env() -> dict[str, str]:
    keep = {
        "CUDA_VISIBLE_DEVICES",
        "HOSTNAME",
        "USER",
        "VIRTUAL_ENV",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("SLURM_") or key in keep
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--params-root",
        default="/home/fcolanto/projects/tabicl/model_params",
        help="Root directory for recorded model params.",
    )
    parser.add_argument("--script-path", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("pass the training command after --")
    return args


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    params_dir = Path(args.params_root) / args.run_name
    params_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_command(args.command)
    git_info = collect_git_info(cwd)
    env = collect_env()
    command_text = shlex.join(args.command)

    record = {
        "run_name": args.run_name,
        "label": args.label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": str(cwd),
        "script_path": args.script_path,
        "checkpoint_dir": args.checkpoint_dir,
        "params_dir": str(params_dir),
        "command": args.command,
        "command_text": command_text,
        "parsed_command": parsed,
        "environment": env,
        "git": git_info,
    }

    (params_dir / "train_params.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (params_dir / "train_command.txt").write_text(command_text + "\n", encoding="utf-8")
    (params_dir / "git_status.txt").write_text(git_info["status_short"] + "\n", encoding="utf-8")
    (params_dir / "slurm_env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote training params to {params_dir}")


if __name__ == "__main__":
    main()
