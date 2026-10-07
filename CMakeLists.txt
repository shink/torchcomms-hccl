cmake_minimum_required(VERSION 3.22)
project(torchcomms_hccl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ====================================================================
# 查找依赖
# ====================================================================

# --- Python ---
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# --- PyTorch ---
# 从已安装的 torch 包获取 include 和 lib 路径
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_INCLUDE_DIR "${TORCH_PKG_DIR}/include")
set(TORCH_LIB_DIR "${TORCH_PKG_DIR}/lib")
message(STATUS "PyTorch package: ${TORCH_PKG_DIR}")

# --- torchcomms ---
# 从已安装的 torchcomms 包获取 libtorchcomms.so 路径
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import torchcomms; import os; print(os.path.dirname(torchcomms.__file__))"
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

# --- CANN (HCCL + ACL) ---
# 通过 ASCEND_HOME_PATH 环境变量查找 CANN 工具链
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "/usr/local/Ascend/cann")
    message(STATUS "ASCEND_HOME_PATH not set, using default: ${ASCEND_HOME_PATH}")
endif()
set(CANN_INCLUDE_DIR "${ASCEND_HOME_PATH}/include")
set(CANN_LIB_DIR "${ASCEND_HOME_PATH}/lib64")
message(STATUS "CANN include: ${CANN_INCLUDE_DIR}")
message(STATUS "CANN lib: ${CANN_LIB_DIR}")

# 检查 HCCL 头文件存在
if(NOT EXISTS "${CANN_INCLUDE_DIR}/hccl.h")
    message(FATAL_ERROR
        "hccl.h not found at ${CANN_INCLUDE_DIR}. "
        "Please install CANN toolkit and set ASCEND_HOME_PATH.")
endif()

# --- torch_npu ---
# 从 torch_npu 包获取 include 路径(NPU 设备相关头文件)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
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
# 获取 pybind11 头文件路径(从 PyTorch 内置的 pybind11)
# ====================================================================
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
    "import torch; import os; p=os.path.join(os.path.dirname(torch.__file__),'include','pybind11'); print(p if os.path.exists(os.path.join(p,'pybind11.h')) else '')"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# ====================================================================
# 编译后端扩展模块
# ====================================================================
set(CSRC_DIR ${CMAKE_SOURCE_DIR}/csrc)

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
    ${CSRC_DIR}/include
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
    ${CANN_LIB_DIR}
)

target_link_libraries(torchcomms_hccl PRIVATE
    ${TORCHCOMMS_LIB}
    torch
    hccl
    ascendcl
)

# 安装
install(TARGETS torchcomms_hccl LIBRARY DESTINATION .)
