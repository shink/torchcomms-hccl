# torchcomms-hccl

[CANN HCCL](https://ascend.huawei.com/hccl) 后端,用于
[torchcomms](https://github.com/meta-pytorch/torchcomms)。

```python
import torch
import torch_npu        # 注册 "npu" 后端
import torchcomms
import torchcomms_hccl  # 注册 "hccl" 后端

comm = torchcomms.new_comm("hccl", torch.device("npu"))
```

当前仅实现 `all_reduce`,其余集合操作会抛出"不支持"。

## 支持的版本

| 包 | 版本 |
|----|------|
| torch | 2.12.0 |
| torch-npu | 2.12.0 |
| torchcomms | 0.3.1 |
| Python | 3.10 – 3.13 |

`torch-npu` 与 `torchcomms` 均依赖 `torch==2.12.0`,请按此组合安装。

## 特性

- 支持 `all_reduce`
- 通过 `torchcomms.backends` entry point 自动注册 `hccl`
- 编译**不需要** CANN Toolkit(submodule 头文件 + 桩库)
- 以独立 wheel 发布

## 依赖

- Python >= 3.10、CMake >= 3.22、支持 C++17 的编译器
- 编译期：`torch==2.12.0`(CPU 版)+ `torchcomms==0.3.1`
- 运行期：CANN Toolkit、`torch-npu==2.12.0`、NPU 硬件

> **`torchcomms` 不在 PyPI 上**(PyPI 同名包是占位包),需从 PyTorch 官方源安装
> (它依赖 `torch==2.12.0`):
>
> ```bash
> pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
> pip install torchcomms==0.3.1 --index-url https://download.pytorch.org/whl/cu126 --no-deps
> ```

## 安装

```bash
git clone --recursive https://github.com/cann-hccl/torchcomms-hccl

cd torchcomms-hccl
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
pip install torchcomms==0.3.1 --index-url https://download.pytorch.org/whl/cu126 --no-deps
pip install -r requirements_dev.txt
bash build.sh
pip install dist/*.whl
```

从 PyPI 安装:先装 `torchcomms==0.3.1`(PyTorch 源)与 `torch-npu==2.12.0`,再
`pip install torchcomms-hccl`。

## 用法

```bash
bash build.sh          # 构建 wheel
bash build.sh --lint   # clang-format(--Werror)+ pylint
bash build.sh --test   # pytest(无 CANN 时 NPU 用例跳过)
```

```bash
torchrun --nproc_per_node=2 examples/all_reduce.py
```

## 支持的集合操作

| 操作 | 状态 |
|------|------|
| `all_reduce` | ✅ |
| 其它(`all_gather`、`broadcast` 等) | 🚧 规划中 |

## 开发

```bash
pip install -r requirements_dev.txt
bash build.sh --lint   # clang-format + pylint
pytest tests/          # all_reduce 用例需要 NPU + CANN
```

## License

[MIT License](./LICENSE)
