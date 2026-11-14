#!/usr/bin/env python
"""Package only the replacement scripts and development data for offline sv3000."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "standalone/epit_pysr"
UPDATE_FILES = (
    "run_pysr.py", "run_pysr.sbatch", "run.sh", "bundle_manifest.json",
    "epit_development.csv", "README.md",
)


def package(output: Path) -> Path:
    canonical = REPO_ROOT / "scripts/epit_pipeline/run_pysr_discovery.py"
    if canonical.read_bytes() != (BUNDLE / "run_pysr.py").read_bytes():
        raise RuntimeError("Standalone runner differs from the canonical runner; sync it before packaging.")
    payloads = {name: (BUNDLE / name).read_bytes() for name in UPDATE_FILES}
    checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                        for name, data in payloads.items())
    payloads["UPDATE_SHA256SUMS"] = checksums.encode()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for name, data in payloads.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o755 if name.endswith((".sh", ".sbatch")) else 0o644
            archive.addfile(entry, io.BytesIO(data))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/epit_pysr_expanded_update.tar.gz"))
    print(package(parser.parse_args().output))


if __name__ == "__main__":
    main()
