# torchcomms-hccl: CANN HCCL backend for torchcomms
#
# 导入此包时会加载 _comms_hccl C++ 扩展,完成 HCCL 后端向
# TorchCommFactory 的注册。之后即可通过 torchcomms.new_comm("hccl", device)
# 创建 HCCL 通信器。

import ctypes
import os

import torch  # noqa: F401  # torchcomms 依赖 torch,需先加载
import torchcomms  # noqa: F401  # 确保依赖已安装

# 加载 libtorchcomms.so(RTLD_LOCAL),使 _comms_hccl 能链接到 TorchCommFactory 符号
_libtorchcomms = os.path.join(
    os.path.dirname(torchcomms.__file__), "libtorchcomms.so"
)
if os.path.exists(_libtorchcomms):
    ctypes.CDLL(_libtorchcomms, mode=ctypes.RTLD_LOCAL)

# 导入 C++ 扩展,触发静态注册器注册 "hccl" 后端
from . import _comms_hccl  # noqa: F401

__version__ = "0.1.0"

__all__ = ["__version__"]
