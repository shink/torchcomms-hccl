// Copyright (c) CANN HCCL Development Team
//
// libaclrt.so 桩实现，避免编译时依赖 CANN Toolkit 包

#include <stddef.h>
#include <stdint.h>

typedef void *aclrtStream;
typedef int aclError;

#define ACL_SUCCESS 0

aclError aclrtSetDevice(int32_t deviceId) {
  (void)deviceId;
  return 100000; /* ACL_ERROR_INVALID_PARAM */
}

aclError aclrtCreateStream(aclrtStream *stream) {
  (void)stream;
  return 100000;
}

aclError aclrtDestroyStream(aclrtStream stream) {
  (void)stream;
  return 100000;
}

aclError aclrtSynchronizeStream(aclrtStream stream) {
  (void)stream;
  return 100000;
}
