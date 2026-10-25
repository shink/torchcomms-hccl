// Copyright (c) CANN HCCL Development Team
//
// libhcomm.so/libhccl.so 桩实现，避免编译时依赖 CANN Toolkit 包

#include <stddef.h>
#include <stdint.h>

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

#define HCCL_ROOT_INFO_BYTES 4108
typedef struct HcclRootInfoDef {
    char internal[HCCL_ROOT_INFO_BYTES];
} HcclRootInfo;

typedef void *aclrtStream;

#define HCCL_SUCCESS 0

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
