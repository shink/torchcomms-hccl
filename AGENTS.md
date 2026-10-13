# AGENTS.md

Guidance for OpenCode sessions working in `torchcomms-hccl`.

## What this repo is

CANN HCCL backend plugin for [torchcomms](https://github.com/meta-pytorch/torchcomms).
The Python package `torchcomms_hccl` is a thin shim; the real work is the
`_comms_hccl` C++ extension built via CMake. Only `all_reduce` is implemented;
`all_gather`/`broadcast`/`reduce_scatter`/`send`/`recv`/`barrier` are stubs.

## Backend registration wiring (load-bearing)

Understand this before touching `src/`:

- `pyproject.toml` declares entry point `hccl = "torchcomms_hccl"` in group
  `torchcomms.backends` (discovered via `importlib.metadata.entry_points`).
- `import torchcomms_hccl` (`torchcomms_hccl/__init__.py`) pre-loads
  `libtorchcomms.so` via `ctypes.CDLL(..., RTLD_LOCAL)` so the C++ extension can
  resolve `TorchCommFactory` symbols, then imports `_comms_hccl`.
- `src/module.cpp` has a file-scope static initializer that calls
  `TorchCommFactory::get().register_backend("hccl", factory,
  TORCHCOMM_BACKEND_ABI_VERSION)`. The pybind11 module exports **nothing** —
  registration is a side effect of `.so` load. Do not move registration into
  `PYBIND11_MODULE`; it must run at static-init time.
- torchcomms headers come from the git submodule at
  `third_party/torchcomms/` (tracked at `comms/torchcomms/`). The contract
  this plugin compiles against is the submodule's `main` branch. Do not
  substitute headers from a different torchcomms version.

## Git submodules (CANN + torchcomms)

The build uses four git submodules under `third_party/`:

| Submodule | URL | Branch | Provides |
|-----------|-----|--------|----------|
| `third_party/hccl` | `gitcode.com/cann/hccl` | master | `<hccl.h>`, `<hccl_mc2.h>` |
| `third_party/hcomm` | `gitcode.com/cann/hcomm` | master | `<hccl/hccl_types.h>`, `<hccl/hccl_comm.h>`, `<hccl/hccl_res.h>`, `<hccl/hccl_launch.h>` |
| `third_party/runtime` | `gitcode.com/cann/runtime` | master | `<acl/acl.h>`, `<acl/acl_rt.h>`, `<acl/acl_base_rt.h>` |
| `third_party/torchcomms` | `github.com/meta-pytorch/torchcomms` | main | `<comms/torchcomms/*.hpp>` |

After cloning: `git submodule update --init --recursive`

## Build modes: with CANN vs. without CANN

The CMake build supports two modes:

### With CANN Toolkit installed (NPU dev box)

```bash
source /usr/local/Ascend/cann/set_env.sh   # sets ASCEND_HOME_PATH
pip install torch torch_npu torchcomms
pip install --no-build-isolation -v .
```

CMake detects `$ASCEND_HOME_PATH/include/hccl/hccl.h` and uses the real
CANN headers + libraries. Behavior is unchanged from before.

### Without CANN (CI, dev laptops, PyPI wheel build)

```bash
pip install torch torchcomms
pip install --no-build-isolation -v .
```

CMake falls back to submodule headers + stub `.so` libraries
(`third_party/stubs/hccl_stub.c`, `third_party/stubs/acl_stub.c`).
The stub libs satisfy link-time symbols; at runtime the dynamic linker
loads the real `libhccl.so` / `libascendcl.so` from the installed CANN.

**No `hccl.h` existence check** — the build never hard-fails on missing CANN.

## Developer commands

```bash
pip install -e ".[dev]"                        # pytest, pylint, clang-format

# Lint — CI enforces clang-format hard, pylint soft:
find src -name '*.hpp' -o -name '*.cpp' -o -name '*.h' \
  | xargs clang-format --dry-run --Werror      # HARD FAIL in CI
pylint torchcomms_hccl/                         # CI runs with || true

# Format C++ in place:
find src -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# Tests:
pytest tests/test_import.py -v                  # no NPU needed (metadata only)
torchrun --nproc_per_node=2 tests/test_all_reduce.py   # needs real NPU + CANN
```

CI has **no NPU and no CANN**, so the extension can never build or load there.
Two workflows on push/PR to `main`:

- `.github/workflows/lint.yml` — clang-format `--Werror` (HARD fail), pylint
  `|| true` (soft).
- `.github/workflows/build.yml` — installs CPU-only torch, runs
  `python setup.py --version` for metadata, then `pytest tests/test_import.py
  -v || true` (soft). Does **not** invoke CMake.

`.github/workflows/release.yml` (on `v*` tags) builds sdist + wheel; the wheel
step is `continue-on-error: true` because CANN is absent on the runner.

Anything requiring the extension to actually load must be verified on an NPU
box — never trust CI green as proof of runtime correctness.

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

Follow the existing split: sync path in `src/TorchCommHCCL.cpp`, async work
handle in `src/TorchWorkHCCL.cpp`, raw HCCL wrapper in `src/HcclApi.hpp`.
The op must satisfy the `TorchCommBackend` contract defined by the
submodule headers in `third_party/torchcomms/comms/torchcomms/` — check
`TorchCommBackend.hpp` and `TorchWork.hpp` for the method signatures to
override.
