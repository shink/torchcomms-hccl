// Copyright (c) torchcomms-hccl contributors
//
// 异步工作句柄实现

#include "TorchWorkHCCL.hpp"

#include <c10/util/Logging.h>

namespace torch::comms {

TorchWorkHCCL::TorchWorkHCCL(aclrtStream stream,
                             std::chrono::milliseconds timeout,
                             std::vector<at::Tensor> inputTensors)
    : stream_(stream),
      timeout_(timeout),
      inputTensors_(std::move(inputTensors)) {
  // 构造时标记为进行中,等待 wait() 后转为完成
  setStatus(WorkStatus::INPROGRESS);
}

void TorchWorkHCCL::wait() {
  runWaitPreHooks();
  if (stream_ != nullptr) {
    auto err = aclrtSynchronizeStream(stream_);
    if (err != ACL_SUCCESS) {
      LOG(ERROR) << "HCCL: aclrtSynchronizeStream failed, error=" << err;
      setStatus(WorkStatus::ERROR);
      runWaitPostHooks();
      throw std::runtime_error("HCCL stream synchronization failed");
    }
  }
  setStatus(WorkStatus::COMPLETED);
  runWaitPostHooks();
}

void TorchWorkHCCL::waitBlocking() {
  // 对于 HCCL,wait() 本身已是主机阻塞语义
  wait();
}

}  // namespace torch::comms
