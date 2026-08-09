"""Hash and verify frozen EPIT pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLIT_LOCK_NAME = "split_lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSplit:
    manifest: dict[str, Any]
    lock: dict[str, Any]
    manifest_path: Path
    lock_path: Path
    manifest_sha256: str
    lock_sha256: str


def build_split_lock(
    *,
    manifest_path: Path,
    assignments_path: Path,
    report_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "epit_split_lock_v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "manifest": {
            "file": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "assignments": {
            "file": assignments_path.name,
            "sha256": sha256_file(assignments_path),
        },
        "report": {
            "file": report_path.name,
            "sha256": sha256_file(report_path),
        },
    }


def load_frozen_split(
    manifest_path: Path,
    *,
    expected_manifest_schema: str = "epit_split_manifest_v2",
) -> FrozenSplit:
    manifest_path = manifest_path.resolve()
    lock_path = manifest_path.parent / SPLIT_LOCK_NAME
    if not lock_path.exists():
        raise RuntimeError(f"Split is not frozen: missing {lock_path}.")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "epit_split_lock_v1":
        raise RuntimeError("Unsupported split lock schema.")

    for name in ("manifest", "assignments", "report"):
        record = lock.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"Split lock is missing the {name} record.")
        artifact_path = manifest_path.parent / str(record.get("file", ""))
        if name == "manifest" and artifact_path != manifest_path:
            raise RuntimeError("Split lock names a different manifest file.")
        if not artifact_path.exists():
            raise RuntimeError(f"Frozen split artifact is missing: {artifact_path}.")
        if sha256_file(artifact_path) != record.get("sha256"):
            raise RuntimeError(f"Frozen split artifact changed: {artifact_path}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != expected_manifest_schema:
        raise RuntimeError("Unsupported split manifest schema.")
    if manifest.get("dataset", {}).get("source_sha256") != lock.get("source_sha256"):
        raise RuntimeError("Split lock and manifest source hashes differ.")

    return FrozenSplit(
        manifest=manifest,
        lock=lock,
        manifest_path=manifest_path,
        lock_path=lock_path,
        manifest_sha256=sha256_file(manifest_path),
        lock_sha256=sha256_file(lock_path),
    )
