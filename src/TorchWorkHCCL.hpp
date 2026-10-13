// Copyright (c) torchcomms-hccl contributors
//
// 异步工作句柄:HCCL 后端的 TorchWork 实现

#pragma once

#include <ATen/core/Tensor.h>
#include <acl/acl.h>
#include <comms/torchcomms/TorchWork.hpp>

#include <chrono>
#include <vector>

namespace torch::comms {

/// HCCL 后端的异步工作句柄
/// 通过同步 NPU 流来完成 wait 语义。
class TorchWorkHCCL : public TorchWork {
 public:
  /// 构造函数
  /// \param stream HCCL 操作所在的 NPU 流
  /// \param timeout 操作超时时间(暂未使用,预留)
  /// \param inputTensors 异步操作时持有的输入张量引用,防止提前释放
  TorchWorkHCCL(aclrtStream stream, std::chrono::milliseconds timeout,
                std::vector<at::Tensor> inputTensors = {});

  ~TorchWorkHCCL() override = default;

  /// 阻塞当前流直到操作完成
  void wait() override;

  /// 阻塞 CPU 线程直到操作完成
  void waitBlocking() override;

 private:
  aclrtStream stream_;
  std::chrono::milliseconds timeout_;
  std::vector<at::Tensor> inputTensors_;  // 持有引用,防止异步期间张量被释放
};

}  // namespace torch::comms
