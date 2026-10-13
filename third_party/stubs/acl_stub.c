// Copyright (c) torchcomms-hccl contributors
//
// ACL stub library: provides link-time symbols for libascendcl.so so the
// extension can be built without a CANN toolkit installation. At runtime
// the dynamic linker loads the real libascendcl.so from the installed CANN.
//
// Only the 4 ACL functions actually called by this project are stubbed.

#include <stddef.h>
#include <stdint.h>

// --- Type stubs (must match acl_base_rt.h / acl_rt.h) ---
// 扮桩类型定义,必须与 CANN 安装包中的 acl_base_rt.h 一致

typedef void *aclrtStream;
typedef int aclError;

#define ACL_SUCCESS 0

// --- Stub implementations (return error, never called at runtime) ---
// 桩实现:返回错误码,运行时不会被调用

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
