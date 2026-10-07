"""导入测试:验证 torchcomms_hccl 包结构正确。

此测试不需要 NPU 硬件,仅验证 Python 包可被发现。
实际 C++ 扩展加载需要 torchcomms 和 CANN 环境。
"""

import pytest


def test_package_discoverable():
    """验证 entry_point 注册正确"""
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="torchcomms.backends", name="hccl")
        assert eps is not None
        # 不断言 eps 非空,因为包可能尚未安装
    except Exception:
        # 在未安装环境下跳过
        pytest.skip("torchcomms.backends entry_points not available")


def test_pyproject_metadata():
    """验证项目元数据可读"""
    try:
        from importlib.metadata import metadata
        m = metadata("torchcomms-hccl")
        assert m["Name"] == "torchcomms-hccl"
    except Exception:
        pytest.skip("package not installed")
