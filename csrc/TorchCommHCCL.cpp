// Copyright (c) torchcomms-hccl contributors
//
// HCCL 后端实现

#include "TorchCommHCCL.hpp"

#include <c10/util/Logging.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "HcclApi.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace torch::comms {

// ============================================================
// 构造与析构
// ============================================================

TorchCommHCCL::TorchCommHCCL()
    : hccl_api_(std::make_shared<DefaultHcclApi>()) {}

TorchCommHCCL::~TorchCommHCCL() {
  finalize();
}

// ============================================================
// 异常辅助
// ============================================================

[[noreturn]] void TorchCommHCCL::throwNotSupported(const char* opName) {
  throw std::runtime_error(std::string("HCCL backend does not support '") +
                           opName +
                           "' yet. Only all_reduce is currently implemented.");
}

// ============================================================
// 生命周期: init / finalize
// ============================================================

void TorchCommHCCL::init(at::Device device, const std::string& name,
                         const CommOptions& options) {
  device_ = device;
  name_ = name;
  options_ = options;

  // 1. 获取 rank 和 world_size
  auto [r, s] = getRankAndSize(options);
  rank_ = r;
  comm_size_ = s;

  // 2. 设置 NPU 设备
  auto aclErr = aclrtSetDevice(device_.index());
  TORCH_CHECK(aclErr == ACL_SUCCESS,
              "HCCL init: aclrtSetDevice failed, error=", aclErr);

  // 3. 创建内部 NPU 流(用于异步操作)
  aclErr = aclrtCreateStream(&internal_stream_);
  TORCH_CHECK(aclErr == ACL_SUCCESS,
              "HCCL init: aclrtCreateStream failed, error=", aclErr);

  // 4. 通过 Store 交换 RootInfo 并初始化通信器
  //    若 options.store 为空,自动创建 TCPStore
  c10::intrusive_ptr<c10d::Store> store = options_.store;
  if (!store) {
    // 从环境变量创建 TCPStore
    const char* masterAddr = std::getenv("MASTER_ADDR");
    const char* masterPort = std::getenv("MASTER_PORT");
    TORCH_CHECK(masterAddr != nullptr && masterPort != nullptr,
                "HCCL init: MASTER_ADDR and MASTER_PORT must be set, "
                "or a store must be provided via CommOptions.store");
    store = c10::make_intrusive<c10d::TCPStore>(
        masterAddr, std::stoi(masterPort), comm_size_, rank_ == 0);
  }

  // 使用 PrefixStore 隔离不同通信器的 key
  c10d::PrefixStore prefixStore(name_, *store);

  HcclRootInfo rootInfo{};
  std::string key = "hccl_root_info";
  exchangeRootInfo(prefixStore, key, rootInfo, options_.timeout);

  // 5. 创建 HCCL 通信器
  auto hcclErr =
      hccl_api_->commInitRootInfo(static_cast<uint32_t>(comm_size_), &rootInfo,
                                  static_cast<uint32_t>(rank_), &hccl_comm_);
  TORCH_CHECK(hcclErr == HCCL_SUCCESS,
              "HCCL init: HcclCommInitRootInfo failed: ",
              hccl_api_->getErrorString(hcclErr));

  LOG(INFO) << "HCCL backend initialized: rank=" << rank_
            << " size=" << comm_size_ << " name=" << name_;
}

void TorchCommHCCL::finalize() {
  if (finalized_.exchange(true)) {
    return;  // 已 finalize,避免重复调用
  }

  if (hccl_comm_ != nullptr) {
    hccl_api_->commDestroy(hccl_comm_);
    hccl_comm_ = nullptr;
  }

  if (internal_stream_ != nullptr) {
    aclrtDestroyStream(internal_stream_);
    internal_stream_ = nullptr;
  }

  LOG(INFO) << "HCCL backend finalized: name=" << name_;
}

// ============================================================
// rank / world_size 获取
// ============================================================

std::pair<int, int> TorchCommHCCL::getRankAndSize(const CommOptions& options) {
  // 优先从环境变量获取(torchrun / torch.distributed.launch 标准变量)
  const char* rankEnv = std::getenv("RANK");
  const char* worldSizeEnv = std::getenv("WORLD_SIZE");

  if (rankEnv != nullptr && worldSizeEnv != nullptr) {
    return {std::stoi(rankEnv), std::stoi(worldSizeEnv)};
  }

  // TODO: 从 Store 中查询(需要 Store 支持 add/get 原子计数)
  // 当前简化实现:默认单进程
  return {0, 1};
}

// ============================================================
// RootInfo 交换(Bootstrap)
// ============================================================

void TorchCommHCCL::exchangeRootInfo(c10d::Store& store, const std::string& key,
                                     HcclRootInfo& rootInfo,
                                     std::chrono::milliseconds timeout) {
  if (rank_ == 0) {
    // rank 0 生成 RootInfo
    auto err = hccl_api_->getRootInfo(&rootInfo);
    TORCH_CHECK(err == HCCL_SUCCESS, "HCCL: HcclGetRootInfo failed: ",
                hccl_api_->getErrorString(err));

    // 序列化并写入 Store
    std::vector<uint8_t> buf(
        reinterpret_cast<uint8_t*>(&rootInfo),
        reinterpret_cast<uint8_t*>(&rootInfo) + sizeof(rootInfo));
    store.set(key, buf);
  } else {
    // 其他 rank 等待并读取 RootInfo
    store.wait({key}, timeout);
    auto val = store.get(key);
    TORCH_CHECK(val.size() == sizeof(rootInfo),
                "HCCL: RootInfo size mismatch from store: got ", val.size(),
                " expected ", sizeof(rootInfo));
    std::memcpy(&rootInfo, val.data(), sizeof(rootInfo));
  }
}

// ============================================================
// 类型映射
// ============================================================

HcclDataType TorchCommHCCL::getHcclDataType(at::ScalarType type) {
  switch (type) {
    case at::kFloat:
      return HCCL_DATA_TYPE_FP32;
    case at::kHalf:
      return HCCL_DATA_TYPE_FP16;
    case at::kBFloat16:
      return HCCL_DATA_TYPE_BFP16;
    case at::kDouble:
      return HCCL_DATA_TYPE_FP64;
    case at::kLong:
      return HCCL_DATA_TYPE_INT64;
    case at::kInt:
      return HCCL_DATA_TYPE_INT32;
    case at::kShort:
      return HCCL_DATA_TYPE_INT16;
    case at::kChar:
      return HCCL_DATA_TYPE_INT8;
    case at::kByte:
      return HCCL_DATA_TYPE_UINT8;
    default:
      TORCH_CHECK(false, "HCCL: unsupported dtype: ", type);
  }
}

HcclReduceOp TorchCommHCCL::getHcclReduceOp(const ReduceOp& op) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:
      return HCCL_REDUCE_SUM;
    case ReduceOp::RedOpType::PRODUCT:
      return HCCL_REDUCE_PROD;
    case ReduceOp::RedOpType::MAX:
      return HCCL_REDUCE_MAX;
    case ReduceOp::RedOpType::MIN:
      return HCCL_REDUCE_MIN;
    default:
      TORCH_CHECK(false,
                  "HCCL: unsupported reduce op. "
                  "HCCL only supports SUM, PRODUCT, MAX, MIN.");
  }
}

// ============================================================
// 流选择
// ============================================================

aclrtStream TorchCommHCCL::getOperationStream(bool async_op) {
  if (async_op) {
    return internal_stream_;
  }
  // 同步模式:使用当前 NPU 流,使操作与用户的计算流同步
  aclrtStream current = nullptr;
  // aclrtGetCurrentStream 在部分 CANN 版本不可用,回退到内部流 + 同步等待
  // 此处使用内部流,wait() 会调用 aclrtSynchronizeStream 保证同步语义
  return internal_stream_;
}

// ============================================================
// AllReduce:当前唯一实现的集合操作
// ============================================================

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_reduce(
    at::Tensor& tensor, const ReduceOp& op, bool async_op,
    const AllReduceOptions& options) {
  // 前置校验
  TORCH_CHECK(hccl_comm_ != nullptr, "HCCL: communicator not initialized");
  TORCH_CHECK(tensor.is_contiguous(),
              "HCCL all_reduce: tensor must be contiguous");
  TORCH_CHECK(tensor.device().type() == device_.type(),
              "HCCL all_reduce: tensor device mismatch");

  // 类型映射
  auto hcclDtype = getHcclDataType(tensor.scalar_type());
  auto hcclRedOp = getHcclReduceOp(op);

  // 获取操作流
  aclrtStream stream = getOperationStream(async_op);

  // 调用 HCCL AllReduce(原地操作:sendbuf == recvbuf)
  auto err = hccl_api_->allReduce(tensor.data_ptr(), tensor.data_ptr(),
                                  static_cast<uint64_t>(tensor.numel()),
                                  hcclDtype, hcclRedOp, hccl_comm_, stream);
  TORCH_CHECK(err == HCCL_SUCCESS,
              "HCCL all_reduce failed: ", hccl_api_->getErrorString(err));

  // 创建 Work 句柄
  // 异步操作时持有张量引用,防止提前释放
  auto timeout = options.timeout;
  if (timeout == std::chrono::milliseconds(0)) {
    timeout = options_.timeout;
  }

  auto work = c10::make_intrusive<TorchWorkHCCL>(
      stream, timeout,
      async_op ? std::vector<at::Tensor>{tensor} : std::vector<at::Tensor>{});

  // 同步模式:立即等待完成
  if (!async_op) {
    work->wait();
  }

  return work;
}

// ============================================================
// split:暂不支持(需要 HCCL commSplit,后续扩展)
// ============================================================

std::shared_ptr<TorchCommBackend> TorchCommHCCL::split(
    const std::vector<int>& ranks, const std::string& name,
    const CommOptions& options) {
  throw std::runtime_error("HCCL backend: split is not yet supported");
}

}  // namespace torch::comms
