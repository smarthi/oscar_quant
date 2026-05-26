#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSCAR_DIR="${OSCAR_DIR:-${REPO_ROOT}/third_party/OScaR-KV-Quant}"
OSCAR_REPO="${OSCAR_REPO:-https://github.com/ZunhaiSu/OScaR-KV-Quant.git}"

mkdir -p "$(dirname "${OSCAR_DIR}")"

if [[ ! -d "${OSCAR_DIR}/.git" ]]; then
  git clone "${OSCAR_REPO}" "${OSCAR_DIR}"
fi

git -C "${OSCAR_DIR}" submodule update --init --recursive

if command -v uv >/dev/null 2>&1; then
  uv pip install "torch==2.6.0+cu124" psutil \
    --index-url https://download.pytorch.org/whl/cu124
  uv pip install --no-build-isolation -e "${OSCAR_DIR}"
else
  python -m pip install "torch==2.6.0+cu124" psutil \
    --index-url https://download.pytorch.org/whl/cu124
  python -m pip install --no-build-isolation -e "${OSCAR_DIR}"
fi

echo "Installed OScaR-KV-Quant from ${OSCAR_DIR}"

