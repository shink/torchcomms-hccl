# torchcomms-hccl

CANN HCCL backend for [torchcomms](https://github.com/meta-pytorch/torchcomms).

将华为昇腾 HCCL (Huawei Collective Communication Library) 接入 torchcomms 框架，
通过 `torchcomms.new_comm("hccl", device)` 即可使用。

## 特性

- ✅ **AllReduce** 集合通信
- ✅ 通过 `entry_points` 自动注册到 torchcomms
- ✅ 依赖 `torch_npu` 提供 NPU 设备支持
- ✅ 编译期不依赖 CANN Toolkit (头文件来自 git submodule,链接用桩库)
- ✅ 可作为独立 wheel 包发布 (`pip install torchcomms-hccl`)
- 🚧 其他集合操作 (all_gather, broadcast, reduce_scatter 等) 逐步扩展中

## 依赖

### 编译期依赖

| 依赖 | 说明 |
|------|------|
| Python ≥ 3.10 | |
| PyTorch (CPU 版) ≥ 2.8 | 编译期只需 CPU 版,不需要 NPU 版 |
| torchcomms | Meta 的集合通信框架,提供 `libtorchcomms.so` |
| CMake ≥ 3.22 | 构建工具 |
| C++20 编译器 | GCC ≥ 11 或 Clang ≥ 12 |

CANN 头文件通过 git submodule 引入 (`third_party/hccl`、`third_party/hcomm`、`third_party/runtime`),
编译期不依赖已安装的 CANN Toolkit。`libhccl.so`、`libascendcl.so` 由 `third_party/stubs/` 下的桩库
满足链接期符号解析,运行期由动态链接器加载真实 CANN 库。

### 运行期依赖

| 依赖 | 说明 |
|------|------|
| CANN Toolkit | 提供 `libhccl.so` 和 `libascendcl.so` |
| torch_npu | 昇腾 NPU 设备注册层 |
| NPU 硬件 | 昇腾 AI 处理器 |

## 安装

### 从源码构建

```bash
# 克隆仓库 (含 submodule)
git clone --recursive <repo-url>
cd torchcomms-hccl

# 编译期只需 CPU 版 torch + torchcomms
pip install torch torchcomms
pip install --no-build-isolation -v .
```

如果已安装 CANN Toolkit,CMake 会自动检测并使用真实 CANN 头文件和库:

```bash
source /usr/local/Ascend/cann/set_env.sh   # 设置 ASCEND_HOME_PATH
pip install torch torch_npu torchcomms
pip install --no-build-isolation -v .
```

### 从 PyPI 安装

```bash
pip install torchcomms-hccl
```

## 快速开始

```bash
torchrun --nproc_per_node=2 examples/all_reduce.py
```

完整代码见 [`examples/all_reduce.py`](examples/all_reduce.py)。

### 多进程运行

```bash
# 单机多卡
torchrun --nproc_per_node=2 examples/all_reduce.py

# 多机
torchrun --nnodes=2 --nproc_per_node=8 \
  --rdzv-endpoint="<master>:<port>" examples/all_reduce.py
```

## 项目结构

```
torchcomms-hccl/
├── src/                           # C++ 源码
│   ├── HcclApi.hpp                # HCCL API 抽象层
│   ├── TorchCommHCCL.hpp          # 后端实现
│   ├── TorchCommHCCL.cpp
│   ├── TorchWorkHCCL.hpp          # 异步工作句柄
│   ├── TorchWorkHCCL.cpp
│   └── module.cpp                 # pybind11 绑定 + 注册
├── third_party/                   # git submodules + 桩库
│   ├── hccl/                      # submodule: CANN HCCL 源码
│   ├── hcomm/                     # submodule: CANN HCOMM 源码
│   ├── runtime/                   # submodule: CANN runtime 源码
│   ├── torchcomms/                # submodule: torchcomms 源码
│   └── stubs/                     # 编译期桩库 (无 CANN 时链接用)
│       ├── hccl_stub.c            #   libhccl.so 桩 (5 个函数)
│       └── acl_stub.c             #   libascendcl.so 桩 (4 个函数)
├── examples/                      # 示例代码
│   └── all_reduce.py              #   AllReduce 示例
├── torchcomms_hccl/               # Python 包
│   └── __init__.py
├── tests/                         # 测试
├── CMakeLists.txt                 # CMake 构建
├── setup.py                       # Python 打包
├── pyproject.toml                 # 项目配置
├── MANIFEST.in                    # sdist 打包清单
├── .clang-format                  # C++ 代码风格
├── .pylintrc                      # Python 代码检查
└── .github/workflows/             # CI/CD
    ├── lint.yml                   # clang-format + pylint
    ├── build.yml                  # 编译验证
    └── release.yml                # 发布到 PyPI
```

## 架构

```
用户代码
  ↓ torchcomms.new_comm("hccl", device)
TorchCommFactory (torchcomms 核心)
  ↓ register_backend("hccl", factory, ABI_VERSION)
TorchCommHCCL (本项目)
  ↓ HcclAllReduce()
HCCL 库 (libhccl.so)
  ↓
NPU 硬件

编译期:
  third_party/hccl/include/         → <hccl.h>
  third_party/hcomm/include/        → <hccl/hccl_types.h>, <hccl/hccl_comm.h> 等
  third_party/runtime/include/      → <acl/acl.h>, <acl/acl_rt.h> 等
  third_party/torchcomms/comms/     → <comms/torchcomms/*.hpp>
  third_party/stubs/*.c             → libhccl.so, libascendcl.so (桩库,仅链接期)

运行期:
  libhccl.so     → CANN 安装目录/lib64/libhccl.so
  libascendcl.so → CANN 安装目录/lib64/libascendcl.so
  libtorchcomms.so → pip 安装的 torchcomms 包
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
find src -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# 检查 Python
pylint torchcomms_hccl/

# 运行测试 (需要 NPU 环境)
pytest tests/ -v
```

## License

MIT
