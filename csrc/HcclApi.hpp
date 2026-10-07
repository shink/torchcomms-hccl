// Copyright (c) torchcomms-hccl contributors
//
// HCCL API 抽象层:封装 HCCL C API,支持依赖注入与单元测试 mock

#pragma once

#include <acl/acl.h>
#include <hccl.h>

#include <memory>
#include <string>

namespace torch::comms {

// HCCL 操作结果类型别名
using HcclResult = hcclResult;

/// HCCL API 抽象接口
/// 设计目的与 torchcomms 的 NcclApi 一致:将 HCCL C API 调用抽象为可注入接口,
/// 使后端逻辑可在无 NPU 环境下通过 mock 进行单元测试。
class HcclApi {
 public:
  virtual ~HcclApi() = default;

  // --- Bootstrap ---
  /// rank 0 生成 RootInfo,用于广播给其他 rank
  virtual HcclResult getRootInfo(HcclRootInfo* rootInfo) = 0;
  /// 根据 RootInfo 初始化通信器
  virtual HcclResult commInitRootInfo(uint32_t nRanks,
                                      const HcclRootInfo* rootInfo,
                                      uint32_t rank, HcclComm* comm) = 0;
  /// 销毁通信器
  virtual HcclResult commDestroy(HcclComm comm) = 0;
  /// 查询异步错误状态
  virtual HcclResult getCommAsyncError(HcclComm comm,
                                       HcclResult* asyncError) = 0;

  // --- 集合通信 ---
  /// AllReduce:对 sendbuff/recvbuff 执行归约
  virtual HcclResult allReduce(void* sendbuff, void* recvbuff, uint64_t count,
                               HcclDataType dataType, HcclReduceOp op,
                               HcclComm comm, aclrtStream stream) = 0;

  // --- 错误处理 ---
  virtual const char* getErrorString(HcclResult result) = 0;
};

/// 默认实现:直接调用 HCCL 库函数
class DefaultHcclApi : public HcclApi {
 public:
  HcclResult getRootInfo(HcclRootInfo* rootInfo) override {
    return HcclGetRootInfo(rootInfo);
  }

  HcclResult commInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo,
                              uint32_t rank, HcclComm* comm) override {
    return HcclCommInitRootInfo(nRanks, rootInfo, rank, comm);
  }

  HcclResult commDestroy(HcclComm comm) override {
    return HcclCommDestroy(comm);
  }

  HcclResult getCommAsyncError(HcclComm comm, HcclResult* asyncError) override {
    return HcclGetCommAsyncError(comm, asyncError);
  }

  HcclResult allReduce(void* sendbuff, void* recvbuff, uint64_t count,
                       HcclDataType dataType, HcclReduceOp op, HcclComm comm,
                       aclrtStream stream) override {
    return HcclAllReduce(sendbuff, recvbuff, count, dataType, op, comm, stream);
  }

  const char* getErrorString(HcclResult result) override {
    // HCCL 未提供统一的 getErrorString,这里返回固定提示
    static thread_local char buf[64];
    snprintf(buf, sizeof(buf), "HCCL error code: %d", static_cast<int>(result));
    return buf;
  }
};

}  // namespace torch::comms
