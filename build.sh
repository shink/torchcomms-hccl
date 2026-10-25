#!/usr/bin/env bash
# torchcomms-hccl: build / lint / test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE=build

usage() {
  echo "Usage: $0 [--lint|--test|--clean|-h]"
  echo "  (default)  build the wheel into dist/"
  echo "  --lint     clang-format (--Werror) + pylint"
  echo "  --test     pytest tests/"
  echo "  --clean    remove build artifacts"
  echo "  -h         show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lint) MODE=lint ;;
    --test) MODE=test ;;
    --clean) MODE=clean ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" >&2; exit 1; }
}

clean() {
  rm -rf build dist build_* *.egg-info .pytest_cache
  find src torchcomms_hccl tests examples -name "__pycache__" -type d -prune \
    -exec rm -rf {} + 2>/dev/null || true
}

lint() {
  need clang-format
  need pylint
  find src \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' \) \
    -exec clang-format --dry-run --Werror {} +
  pylint torchcomms_hccl/
}

run_tests() {
  need pytest
  pytest tests/ -v
}

build() {
  need python3
  [[ -d .git ]] && git submodule update --init --recursive
  python3 -c "import torch" 2>/dev/null \
    || { echo "missing torch; run: pip install -r requirements_dev.txt" >&2; exit 1; }
  python3 -c "import build" 2>/dev/null \
    || { echo "missing 'build'; run: pip install -r requirements_dev.txt" >&2; exit 1; }
  python3 -m build --no-isolation --wheel --outdir dist
  ls -1 dist/
}

case "$MODE" in
  clean) clean ;;
  lint) lint ;;
  test) run_tests ;;
  build) build ;;
esac