# Copyright (c) CANN HCCL Development Team
#
# torchcomms-hccl: CANN HCCL backend for torchcomms
#
# import torchcomms_hccl 时会加载 _comms_hccl C++ 扩展，完成 HCCL 后端向
# TorchCommFactory 的注册。之后即可通过 torchcomms.new_comm("hccl", device)
# 创建 HCCL 通信域。

import ctypes
import os
from importlib.metadata import version

import torch  # noqa: F401
import torchcomms  # noqa: F401

# 加载 libtorchcomms.so(RTLD_LOCAL)，使 _comms_hccl 能链接到 TorchCommFactory 符号
_libtorchcomms = os.path.join(
    os.path.dirname(torchcomms.__file__), "libtorchcomms.so"
)
if os.path.exists(_libtorchcomms):
    ctypes.CDLL(_libtorchcomms, mode=ctypes.RTLD_LOCAL)

# 导入 C++ 扩展，触发静态注册器注册 "hccl" 后端
from . import _comms_hccl  # noqa: F401

__version__ = version("torchcomms-hccl")

__all__ = ["__version__"]
