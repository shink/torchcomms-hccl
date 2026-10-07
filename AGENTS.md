# AGENTS.md

Guidance for OpenCode sessions working in `torchcomms-hccl`.

## What this repo is

CANN HCCL backend plugin for [torchcomms](https://github.com/meta-pytorch/torchcomms).
The Python package `torchcomms_hccl` is a thin shim; the real work is the
`_comms_hccl` C++ extension built via CMake. Only `all_reduce` is implemented;
`all_gather`/`broadcast`/`reduce_scatter`/`send`/`recv`/`barrier` are stubs.

## Backend registration wiring (load-bearing)

Understand this before touching `csrc/`:

- `pyproject.toml` declares entry point `torchcomms.backends = "torchcomms_hccl"`.
- `import torchcomms_hccl` (`torchcomms_hccl/__init__.py`) pre-loads
  `libtorchcomms.so` via `ctypes.CDLL(..., RTLD_LOCAL)` so the C++ extension can
  resolve `TorchCommFactory` symbols, then imports `_comms_hccl`.
- `csrc/module.cpp` has a file-scope static initializer that calls
  `TorchCommFactory::get().register_backend("hccl", ...)`. The pybind11 module
  exports **nothing** — registration is a side effect of `.so` load. Do not
  move registration into `PYBIND11_MODULE`; it must run at static-init time.
- Vendored torchcomms headers live in `csrc/include/comms/torchcomms/`. They
  are the contract this plugin compiles against — do not substitute headers
  from a different torchcomms version.

## Build requires CANN + no isolation

`pip install -v .` **fails under build isolation**: CMake introspects the
already-installed `torch`, `torchcomms`, and `torch_npu` packages at configure
time. Always:

```bash
source /usr/local/Ascend/cann/set_env.sh   # sets ASCEND_HOME_PATH
pip install torch torch_npu torchcomms
pip install --no-build-isolation -v .
```

CMake hard-fails if `$ASCEND_HOME_PATH/include/hccl.h` or
`<torchcomms pkg>/libtorchcomms.so` is missing. There is no fallback.

## Developer commands

```bash
pip install -e ".[dev]"                        # pytest, pylint, clang-format

# Lint — CI enforces clang-format hard, pylint soft:
find csrc -name '*.hpp' -o -name '*.cpp' -o -name '*.h' \
  | xargs clang-format --dry-run --Werror      # HARD FAIL in CI
pylint torchcomms_hccl/                         # CI runs with || true

# Format C++ in place:
find csrc -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# Tests:
pytest tests/test_import.py -v                  # no NPU needed (metadata only)
torchrun --nproc_per_node=2 tests/test_all_reduce.py   # needs real NPU + CANN
```

CI (`.github/workflows/ci.yml`) has **no NPU and no CANN**, runs only
`test_import.py`, and the wheel-build step in `release.yml` is
`continue-on-error`. Anything requiring the extension to actually load must be
verified on an NPU box — never trust CI green as proof of runtime correctness.

## Style

- **C++**: C++20, Google clang-format, `ColumnLimit: 80`, `IndentWidth: 2`,
  `PointerAlignment: Left`, `SortIncludes: true`. Config in `.clang-format`.
  CI hard-fails on format drift — run clang-format before committing.
- **Python**: pylint max-line-length 88. `_comms_hccl` is in the ignore list
  (generated extension, do not lint). Docstring / `import-outside-toplevel`
  / `line-too-long` are disabled.
- Comments and docstrings are bilingual (Chinese + English). Match the
  surrounding file's language when editing.

## Adding a collective op

Follow the existing split: sync path in `csrc/TorchCommHCCL.cpp`, async work
handle in `csrc/TorchWorkHCCL.cpp`, raw HCCL wrapper in `csrc/HcclApi.hpp`.
The op must satisfy the `TorchCommBackend` contract defined by the vendored
headers in `csrc/include/comms/torchcomms/` — check `TorchCommBackend.hpp`
and `TorchWork.hpp` for the method signatures to override.
