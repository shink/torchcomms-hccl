cmake_minimum_required(VERSION 3.22)
project(torchcomms_hccl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ====================================================================
# 路径常量
# ====================================================================

set(CSRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)
set(STUBS_DIR ${THIRD_PARTY_DIR}/stubs)

# ====================================================================
# 查找 Python
# ====================================================================

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ====================================================================
# 查找 PyTorch (从已安装的 torch 包获取 include/lib 路径)
# ====================================================================

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import os,sys,io;old=sys.stdout;sys.stdout=io.StringIO();import torch;sys.stdout=old;print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_INCLUDE_DIR "${TORCH_PKG_DIR}/include")
set(TORCH_LIB_DIR "${TORCH_PKG_DIR}/lib")
message(STATUS "PyTorch package: ${TORCH_PKG_DIR}")

# ====================================================================
# 查找 torchcomms（头文件从 submodule 获取，库文件从已安装的包获取）
# ====================================================================

# 用 importlib.util.find_spec 定位 torchcomms 包，而不是直接 import:
# import torchcomms 会触发其加载 libtorchcomms.so，而后者依赖运行期的
# CUDA/CANN 库，在无 CANN/无 CUDA 的构建容器中无法加载。这里只需要
# libtorchcomms.so 的路径用于链接，并不需要真正加载它。
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import importlib.util,os,sys; s=importlib.util.find_spec('torchcomms'); (s is None) and sys.exit(1); print(os.path.dirname(s.origin))"
    OUTPUT_VARIABLE TORCHCOMMS_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCHCOMMS_FIND_RESULT
)
if(NOT TORCHCOMMS_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "torchcomms package not found. Please: pip install torchcomms")
endif()
set(TORCHCOMMS_LIB "${TORCHCOMMS_PKG_DIR}/libtorchcomms.so")
if(NOT EXISTS "${TORCHCOMMS_LIB}")
    message(FATAL_ERROR "libtorchcomms.so not found at: ${TORCHCOMMS_LIB}")
endif()
message(STATUS "torchcomms lib: ${TORCHCOMMS_LIB}")

# torchcomms 头文件来自 git submodule
set(TORCHCOMMS_INCLUDE_DIR "${THIRD_PARTY_DIR}/torchcomms")
message(STATUS "torchcomms headers (submodule): ${TORCHCOMMS_INCLUDE_DIR}")

# ====================================================================
# 查找 torch_npu (从已安装的 torch_npu 包获取 include 路径)
# ====================================================================

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import os,sys,io;old=sys.stdout;sys.stdout=io.StringIO();import torch_npu;sys.stdout=old;print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_FIND_RESULT
)
if(TORCH_NPU_FIND_RESULT EQUAL 0)
    set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_PKG_DIR}/include")
    message(STATUS "torch_npu include: ${TORCH_NPU_INCLUDE_DIR}")
else()
    message(WARNING "torch_npu not found; NPU device headers will be unavailable")
    set(TORCH_NPU_INCLUDE_DIR "")
endif()

# ====================================================================
# 获取 pybind11 头文件路径 (从 PyTorch 内置的 pybind11)
# ====================================================================

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import os,sys,io;old=sys.stdout;sys.stdout=io.StringIO();import torch;p=os.path.join(os.path.dirname(torch.__file__),'include','pybind11');sys.stdout=old;print(p if os.path.exists(os.path.join(p,'pybind11.h')) else '')"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ====================================================================
# CANN: 优先使用已安装的 CANN Toolkit，否则使用 submodule + stub 库
# ====================================================================

set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(ASCEND_HOME_PATH AND EXISTS "${ASCEND_HOME_PATH}/include/hccl/hccl.h")
    # --- 方式 1: 使用已安装的 CANN Toolkit ---
    # Mode 1: Use installed CANN toolkit (full headers + real .so)
    set(CANN_INCLUDE_DIR "${ASCEND_HOME_PATH}/include")
    set(CANN_LIB_DIR "${ASCEND_HOME_PATH}/lib64")
    set(HCCL_LIB "hccl")
    set(ACL_LIB "ascendcl")
    message(STATUS "CANN (installed): include=${CANN_INCLUDE_DIR} lib=${CANN_LIB_DIR}")
else()
    # --- 方式 2: 使用 submodule 头文件 + stub 库 ---
    # Mode 2: Use submodule headers + stub libraries (link-only)
    # 编译期用 submodule 头文件，链接期用 stub .so (运行期 dlopen 真实 CANN)
    set(CANN_INCLUDE_DIR
        "${THIRD_PARTY_DIR}/hccl/include"        # <hccl.h>, <hccl_mc2.h>
        "${THIRD_PARTY_DIR}/hcomm/include"        # <hccl/hccl_types.h>, <hccl/hccl_comm.h> 等
        "${THIRD_PARTY_DIR}/runtime/include/external"  # <acl/acl.h>, <acl/acl_rt.h> 等
        "${THIRD_PARTY_DIR}/hcomm/test/stub/depends/include"  # securec.h 等 hcomm 内部依赖
    )
    message(STATUS "CANN (submodule + stubs): headers from ${THIRD_PARTY_DIR}")

    # 构建 stub 共享库 (仅用于链接期符号解析，不打包进 wheel)
    add_library(hccl_stub SHARED ${STUBS_DIR}/hccl_stub.c)
    set_target_properties(hccl_stub PROPERTIES
        OUTPUT_NAME "hccl"
        PREFIX "lib"
    )

    add_library(acl_stub SHARED ${STUBS_DIR}/acl_stub.c)
    set_target_properties(acl_stub PROPERTIES
        OUTPUT_NAME "ascendcl"
        PREFIX "lib"
    )

    # stub 库输出到 build 目录，链接时使用完整路径
    set(HCCL_LIB "hccl_stub")
    set(ACL_LIB "acl_stub")
endif()

# ====================================================================
# 编译后端扩展模块
# ====================================================================

# 获取 Python ABI 后缀
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(torchcomms_hccl MODULE
    ${CSRC_DIR}/TorchCommHCCL.cpp
    ${CSRC_DIR}/TorchWorkHCCL.cpp
    ${CSRC_DIR}/module.cpp
)

set_target_properties(torchcomms_hccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_hccl"
    SUFFIX "${PYTHON_EXT_SUFFIX}"
)

target_include_directories(torchcomms_hccl PRIVATE
    ${CSRC_DIR}
    ${TORCHCOMMS_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIR}/torch/csrc/api/include
    ${CANN_INCLUDE_DIR}
    ${TORCH_NPU_INCLUDE_DIR}
)

if(PYBIND11_INCLUDE_DIR)
    target_include_directories(torchcomms_hccl PRIVATE ${PYBIND11_INCLUDE_DIR})
endif()

target_link_directories(torchcomms_hccl PRIVATE
    ${TORCH_LIB_DIR}
)

target_link_libraries(torchcomms_hccl PRIVATE
    ${TORCHCOMMS_LIB}
    torch
    ${HCCL_LIB}
    ${ACL_LIB}
)

# 安装
install(TARGETS torchcomms_hccl LIBRARY DESTINATION .)
