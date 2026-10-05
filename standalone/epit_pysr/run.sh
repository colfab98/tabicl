#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export JULIA_DEPOT_PATH="${BUNDLE_DIR}/runtime/julia_depot"
export PYTHON_JULIAPKG_PROJECT="${BUNDLE_DIR}/runtime/julia_project"
export PYTHON_JULIAPKG_EXE="${BUNDLE_DIR}/runtime/julia/bin/julia"
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIACALL_THREADS="${PYTHON_JULIACALL_THREADS:-auto}"
exec "${BUNDLE_DIR}/.venv/bin/python" "${BUNDLE_DIR}/run_pysr.py" "$@"
