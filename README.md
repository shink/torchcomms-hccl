# torchcomms-hccl

CANN HCCL backend for [torchcomms](https://github.com/meta-pytorch/torchcomms).

将华为昇腾 HCCL (Huawei Collective Communication Library) 接入 torchcomms 框架，
通过 `torchcomms.new_comm("hccl", device)` 即可使用。

## 特性

- ✅ **AllReduce** 集合通信
- ✅ 通过 `entry_points` 自动注册到 torchcomms
- ✅ 依赖 `torch_npu` 提供 NPU 设备支持
- ✅ 可作为独立 wheel 包发布 (`pip install torchcomms-hccl`)
- 🚧 其他集合操作 (all_gather, broadcast, reduce_scatter 等) 逐步扩展中

## 依赖

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.10 | |
| PyTorch ≥ 2.8 | |
| torchcomms | Meta 的集合通信框架 |
| torch_npu | 昇腾 NPU 设备注册层 |
| CANN Toolkit | 提供 `hccl.h` 和 `libhccl.so` |

## 安装

### 前置条件

安装 CANN Toolkit 并设置环境变量：

```bash
# 安装 CANN 后
source /usr/local/Ascend/cann/set_env.sh
echo $ASCEND_HOME_PATH  # 应指向 CANN 安装目录
```

### 从源码构建

```bash
pip install torch torch_npu torchcomms
pip install --no-build-isolation -v .
```

### 从 PyPI 安装

```bash
pip install torchcomms-hccl
```

## 快速开始

```python
import torch
import torch_npu          # 注册 NPU 设备
import torchcomms
import torchcomms_hccl    # 触发 HCCL 后端注册

# 创建 HCCL 通信器
device = torch.device("npu:0")
comm = torchcomms.new_comm("hccl", device, name="main_comm")

rank = comm.get_rank()
world_size = comm.get_size()

# 创建张量
tensor = torch.full((1024,), float(rank + 1), dtype=torch.float32, device=device)

# AllReduce
comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

print(f"Rank {rank}: result = {tensor[0].item()}")

comm.finalize()
```

### 多进程运行

```bash
# 单机多卡
torchrun --nproc_per_node=2 example.py

# 多机
torchrun --nnodes=2 --nproc_per_node=8 \
  --rdzv-endpoint="<master>:<port>" example.py
```

## 项目结构

```
torchcomms-hccl/
├── csrc/                          # C++ 源码
│   ├── include/comms/torchcomms/  # vendored torchcomms 头文件
│   ├── HcclApi.hpp                # HCCL API 抽象层
│   ├── TorchCommHCCL.hpp          # 后端实现
│   ├── TorchCommHCCL.cpp
│   ├── TorchWorkHCCL.hpp          # 异步工作句柄
│   ├── TorchWorkHCCL.cpp
│   └── module.cpp                 # pybind11 绑定 + 注册
├── torchcomms_hccl/               # Python 包
│   └── __init__.py
├── tests/                         # 测试
├── CMakeLists.txt                 # CMake 构建
├── setup.py                       # Python 打包
├── pyproject.toml                 # 项目配置 (pylint, entry_points)
├── .clang-format                  # C++ 代码风格
├── .pylintrc                      # Python 代码检查
└── .github/workflows/             # CI/CD
    ├── ci.yml                     # lint + build
    └── release.yml                # 发布到 PyPI
```

## 架构

```
用户代码
  ↓ torchcomms.new_comm("hccl", device)
TorchCommFactory (torchcomms 核心)
  ↓ register_backend("hccl", factory)
TorchCommHCCL (本项目)
  ↓ HcclAllReduce()
HCCL 库 (libhccl.so)
  ↓
NPU 硬件

依赖关系:
  torchcomms  ← 提供后端框架 (TorchCommBackend, TorchCommFactory)
  torch_npu   ← 提供 NPU 设备注册 (at::Device("npu"))
  CANN        ← 提供 HCCL + ACL 运行时
```

## 支持的集合操作

| 操作 | 状态 |
|------|------|
| `all_reduce` | ✅ 已实现 |
| `all_gather` | 🚧 开发中 |
| `broadcast` | 🚧 开发中 |
| `reduce_scatter` | 🚧 开发中 |
| `send` / `recv` | 🚧 开发中 |
| `barrier` | 🚧 开发中 |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 格式化 C++
find csrc -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# 检查 Python
pylint torchcomms_hccl/

# 运行测试 (需要 NPU 环境)
pytest tests/ -v
```

## License

MIT
