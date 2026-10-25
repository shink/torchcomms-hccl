# torchcomms-hccl

[CANN HCCL](https://ascend.huawei.com/hccl) backend for
[torchcomms](https://github.com/meta-pytorch/torchcomms).

```python
import torch
import torch_npu        # register the "npu" backend
import torchcomms
import torchcomms_hccl  # registers the "hccl" backend

comm = torchcomms.new_comm("hccl", torch.device("npu"))
```

Only `all_reduce` is implemented; other collectives raise "not supported".

## Supported versions

| Package | Version |
|---------|---------|
| torch | 2.12.0 |
| torch-npu | 2.12.0 |
| torchcomms | 0.3.1 |
| Python | 3.10 – 3.13 |

`torch-npu` and `torchcomms` both pin `torch==2.12.0`; install them as this set.

## Features

- `all_reduce` collective
- Auto-registers `hccl` via the `torchcomms.backends` entry point
- Builds **without** a CANN Toolkit (submodule headers + stub libs)
- Ships as a standalone wheel

## Requirements

- Python >= 3.10, CMake >= 3.22, a C++17 compiler
- Build: `torch==2.12.0` (CPU build) + `torchcomms==0.3.1`
- Runtime: CANN Toolkit, `torch-npu==2.12.0`, NPU hardware

> **`torchcomms` is not on PyPI** (the PyPI name is a placeholder). Install it
> from the PyTorch index (it pins `torch==2.12.0`):
>
> ```bash
> pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
> pip install torchcomms==0.3.1 --index-url https://download.pytorch.org/whl/cu126 --no-deps
> ```

## Install

```bash
git clone --recursive https://github.com/cann-hccl/torchcomms-hccl

cd torchcomms-hccl
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchcomms==0.3.1 --index-url https://download.pytorch.org/whl/cu126 --no-deps
pip install -r requirements_dev.txt
bash build.sh
pip install dist/*.whl
```

From PyPI: install `torchcomms==0.3.1` (PyTorch index) and `torch-npu==2.12.0`
first, then `pip install torchcomms-hccl`.

## Usage

```bash
bash build.sh          # build the wheel
bash build.sh --lint   # clang-format (--Werror) + pylint
bash build.sh --test   # pytest (NPU tests skip without CANN)
```

```bash
torchrun --nproc_per_node=2 examples/all_reduce.py
```

## Supported collectives

| Op | Status |
|----|--------|
| `all_reduce` | ✅ |
| others (`all_gather`, `broadcast`, …) | 🚧 planned |

## Development

```bash
pip install -r requirements_dev.txt
bash build.sh --lint   # clang-format + pylint
pytest tests/       # the all_reduce test needs NPU + CANN
```

## License

[MIT License](./LICENSE)
