// Copyright (c) torchcomms-hccl contributors
//
// HCCL stub library: provides link-time symbols for libhccl.so so the
// extension can be built without a CANN toolkit installation. At runtime
// the dynamic linker loads the real libhccl.so from the installed CANN.
//
// Only the 5 HCCL functions actually called by this project are stubbed.

#include <stddef.h>
#include <stdint.h>

// --- Type stubs (must match hccl_types.h / hccl_comm.h) ---
// 扮桩类型定义,必须与 CANN 安装包中的 hccl_types.h 一致

typedef int HcclResult;
typedef void *HcclComm;

typedef enum {
    HCCL_REDUCE_SUM = 0,
    HCCL_REDUCE_PROD = 1,
    HCCL_REDUCE_MAX = 2,
    HCCL_REDUCE_MIN = 3,
} HcclReduceOp;

typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,
    HCCL_DATA_TYPE_INT16 = 1,
    HCCL_DATA_TYPE_INT32 = 2,
    HCCL_DATA_TYPE_FP16 = 3,
    HCCL_DATA_TYPE_FP32 = 4,
    HCCL_DATA_TYPE_INT64 = 5,
    HCCL_DATA_TYPE_UINT64 = 6,
    HCCL_DATA_TYPE_UINT8 = 7,
    HCCL_DATA_TYPE_UINT16 = 8,
    HCCL_DATA_TYPE_UINT32 = 9,
    HCCL_DATA_TYPE_FP64 = 10,
    HCCL_DATA_TYPE_BFP16 = 11,
} HcclDataType;

// HCCL_ROOT_INFO_BYTES = 4108 (must match CANN exactly)
// 必须与 CANN 安装包中 hccl_types.h 的 HCCL_ROOT_INFO_BYTES 完全一致
#define HCCL_ROOT_INFO_BYTES 4108
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

typedef void *aclrtStream;

#define HCCL_SUCCESS 0

// --- Stub implementations (return error, never called at runtime) ---
// 桩实现:返回错误码,运行时不会被调用

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo) {
  (void)rootInfo;
  return 1; /* HCCL_E_PARA */
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks,
                                const HcclRootInfo *rootInfo,
                                uint32_t rank, HcclComm *comm) {
  (void)nRanks;
  (void)rootInfo;
  (void)rank;
  (void)comm;
  return 1;
}

HcclResult HcclCommDestroy(HcclComm comm) {
  (void)comm;
  return 1;
}

HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult *asyncError) {
  (void)comm;
  (void)asyncError;
  return 1;
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count,
                         HcclDataType dataType, HcclReduceOp op,
                         HcclComm comm, aclrtStream stream) {
  (void)sendBuf;
  (void)recvBuf;
  (void)count;
  (void)dataType;
  (void)op;
  (void)comm;
  (void)stream;
  return 1;
}
