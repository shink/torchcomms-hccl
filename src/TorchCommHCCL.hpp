// Copyright (c) CANN HCCL Development Team
//
// HCCL 后端：实现 TorchCommBackend 接口
// 当前仅支持 all_reduce 操作，其他集合操作后续扩展。

#pragma once

#include <ATen/core/Tensor.h>
#include <acl/acl.h>
#include <c10/core/Device.h>
#include <comms/torchcomms/TorchCommBackend.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <hccl.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "HcclApi.hpp"
#include "TorchWorkHCCL.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>

namespace torch::comms {

/**
 * @brief CANN HCCL 后端实现
 * 通过继承 TorchCommBackend，将 HCCL 库接入 torchcomms 框架。
 * 用户通过 torchcomms.new_comm("hccl", device) 创建实例。
 */
class TorchCommHCCL : public TorchCommBackend {
 public:
  static constexpr std::string_view kBackendName = "hccl";

  TorchCommHCCL();
  ~TorchCommHCCL() override;

  // --- 生命周期管理 ---
  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;

  // --- 基本信息查询 ---
  int getRank() const override {
    return rank_;
  }
  int getSize() const override {
    return comm_size_;
  }
  std::string_view getBackendName() const override {
    return kBackendName;
  }
  std::string_view getCommName() const override {
    return name_;
  }

  // --- AllReduce(当前唯一实现的集合操作) ---
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override;

  // --- 以下操作暂未实现，抛出异常 ---
  // 后续版本逐步扩展
  c10::intrusive_ptr<TorchWork>
  send(const at::Tensor&, int, bool, const SendOptions&) override {
    throwNotSupported("send");
  }
  c10::intrusive_ptr<TorchWork> recv(at::Tensor&, int, bool, const RecvOptions&)
      override {
    throwNotSupported("recv");
  }
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>&,
      bool,
      const BatchP2POptions&) override {
    throwNotSupported("batch_op_issue");
  }
  c10::intrusive_ptr<TorchWork>
  broadcast(at::Tensor&, int, bool, const BroadcastOptions&) override {
    throwNotSupported("broadcast");
  }
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor&,
      int,
      const ReduceOp&,
      bool,
      const ReduceOptions&) override {
    throwNotSupported("reduce");
  }
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>&,
      const at::Tensor&,
      bool,
      const AllGatherOptions&) override {
    throwNotSupported("all_gather");
  }
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>&,
      const at::Tensor&,
      bool,
      const AllGatherOptions&) override {
    throwNotSupported("all_gather_v");
  }
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor&,
      const at::Tensor&,
      bool,
      const AllGatherSingleOptions&) override {
    throwNotSupported("all_gather_single");
  }
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor&,
      const std::vector<at::Tensor>&,
      const ReduceOp&,
      bool,
      const ReduceScatterOptions&) override {
    throwNotSupported("reduce_scatter");
  }
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor&,
      const std::vector<at::Tensor>&,
      const ReduceOp&,
      bool,
      const ReduceScatterOptions&) override {
    throwNotSupported("reduce_scatter_v");
  }
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor&,
      const at::Tensor&,
      const ReduceOp&,
      bool,
      const ReduceScatterSingleOptions&) override {
    throwNotSupported("reduce_scatter_single");
  }
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor&,
      const at::Tensor&,
      bool,
      const AllToAllSingleOptions&) override {
    throwNotSupported("all_to_all_single");
  }
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor&,
      const at::Tensor&,
      const std::vector<uint64_t>&,
      const std::vector<uint64_t>&,
      bool,
      const AllToAllvSingleOptions&) override {
    throwNotSupported("all_to_all_v_single");
  }
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>&,
      const std::vector<at::Tensor>&,
      bool,
      const AllToAllOptions&) override {
    throwNotSupported("all_to_all");
  }
  c10::intrusive_ptr<TorchWork> barrier(bool, const BarrierOptions&) override {
    throwNotSupported("barrier");
  }
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor&,
      const std::vector<at::Tensor>&,
      int,
      bool,
      const ScatterOptions&) override {
    throwNotSupported("scatter");
  }
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>&,
      const at::Tensor&,
      int,
      bool,
      const GatherOptions&) override {
    throwNotSupported("gather");
  }

  // --- 通信域管理 ---
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // --- 配置查询 ---
  const CommOptions& getOptions() const override {
    return options_;
  }
  const at::Device& getDevice() const override {
    return device_;
  }

  // --- 可选能力(当前均不支持) ---
  bool supportsReconfigure() const override {
    return false;
  }
  bool isAbortSupported() const override {
    return false;
  }

 private:
  /**
   * @brief 抛出"操作未支持"异常
   *
   * @param opName 操作名称
   * @throws std::runtime_error 总是抛出
   */
  [[noreturn]] static void throwNotSupported(const char* opName);

  /**
   * @brief 从环境变量或 Store 获取 rank 和 world_size
   *
   * @param options 通信选项
   * @return (rank, world_size)
   */
  std::pair<int, int> getRankAndSize(const CommOptions& options);

  /**
   * @brief 通过 Store 交换 HcclRootInfo(rank 0 生成，广播给其他 rank)
   *
   * @param store 分布式 Store
   * @param key Store 键
   * @param rootInfo 输入输出的 RootInfo
   * @param timeout 等待超时
   */
  void exchangeRootInfo(
      c10d::Store& store,
      const std::string& key,
      HcclRootInfo& rootInfo,
      std::chrono::milliseconds timeout);

  /**
   * @brief PyTorch ScalarType 映射为 HCCL DataType
   *
   * @param type PyTorch 标量类型
   * @return HCCL 数据类型
   */
  static HcclDataType getHcclDataType(at::ScalarType type);

  /**
   * @brief torchcomms ReduceOp 映射为 HCCL ReduceOp
   *
   * @param op torchcomms 归约操作
   * @return HCCL 归约操作
   */
  static HcclReduceOp getHcclReduceOp(const ReduceOp& op);

  /**
   * @brief 获取操作流：异步用内部流，同步用当前流
   *
   * @param async_op 是否异步
   * @return NPU 流
   */
  aclrtStream getOperationStream(bool async_op);

  // --- 成员变量 ---
  HcclComm hccl_comm_{nullptr};
  at::Device device_{c10::kPrivateUse1, 0};
  int rank_{0};
  int comm_size_{1};
  std::string name_;
  CommOptions options_;
  aclrtStream internal_stream_{nullptr};
  std::shared_ptr<HcclApi> hccl_api_;
  std::atomic<bool> finalized_{false};
};

} // namespace torch::comms
