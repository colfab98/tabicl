#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${BUNDLE_DIR}/runtime"

(
  cd "${BUNDLE_DIR}"
  sha256sum --check OFFLINE_SHA256SUMS
)

if [[ ! -x "${RUNTIME_DIR}/julia/bin/julia" ]]; then
  mkdir -p "${RUNTIME_DIR}/julia"
  tar -xzf "${RUNTIME_DIR}/julia-linux-x86_64.tar.gz" \
    -C "${RUNTIME_DIR}/julia"
fi

if [[ ! -d "${RUNTIME_DIR}/julia_depot/packages" ]]; then
  mkdir -p "${RUNTIME_DIR}/julia_depot"
  tar -xzf "${RUNTIME_DIR}/julia-depot-pysr.tar.gz" \
    -C "${RUNTIME_DIR}/julia_depot"
fi

python3 -m venv "${BUNDLE_DIR}/.venv"
"${BUNDLE_DIR}/.venv/bin/pip" install \
  --no-index \
  --find-links "${BUNDLE_DIR}/wheelhouse" \
  --requirement "${BUNDLE_DIR}/requirements.txt"

export JULIA_DEPOT_PATH="${RUNTIME_DIR}/julia_depot"
export PYTHON_JULIAPKG_PROJECT="${RUNTIME_DIR}/julia_project"
export PYTHON_JULIAPKG_EXE="${RUNTIME_DIR}/julia/bin/julia"
export PYTHON_JULIAPKG_OFFLINE=yes
export PYTHON_JULIACALL_THREADS="${PYTHON_JULIACALL_THREADS:-auto}"
"${BUNDLE_DIR}/.venv/bin/python" -c \
  'import pysr; print("PySR", pysr.__version__, "offline runtime ready")'
