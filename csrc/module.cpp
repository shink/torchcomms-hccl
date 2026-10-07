// Copyright (c) torchcomms-hccl contributors
//
// Python 绑定模块:加载时自动注册 HCCL 后端到 TorchCommFactory

#include <comms/torchcomms/TorchCommFactory.hpp>
#include <pybind11/pybind11.h>

#include "TorchCommHCCL.hpp"

#include <memory>

namespace {

/// 静态注册器:模块加载时将 HCCL 后端注册到 TorchCommFactory
/// 通过 entry_points("torchcomms.backends") 发现并 import 此模块
class HCCLBackendRegister {
 public:
  HCCLBackendRegister() {
    torch::comms::TorchCommFactory::get().register_backend("hccl", []() {
      return std::make_shared<torch::comms::TorchCommHCCL>();
    });
  }
};

// 静态变量在 .so 加载时构造,完成注册
static HCCLBackendRegister s_register;

}  // namespace

// pybind11 模块定义
// 模块本身不导出任何符号,仅作为 entry_point 加载的载体
PYBIND11_MODULE(_comms_hccl, m) {
  m.doc() = "torchcomms HCCL backend extension";
  // 注册在静态初始化阶段已完成,此处无需额外操作
}
