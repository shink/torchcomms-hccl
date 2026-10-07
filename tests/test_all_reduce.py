"""AllReduce 集成测试:需要 NPU 硬件 + CANN 环境。

运行方式:
    torchrun --nproc_per_node=2 tests/test_all_reduce.py
"""

import os

import pytest


def _check_npu_available():
    """检查 NPU 环境是否可用"""
    try:
        import torch
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_all_reduce_basic():
    """测试基本 AllReduce 操作"""
    import torch
    import torch_npu  # noqa: F401
    import torchcomms
    import torchcomms_hccl  # noqa: F401  # 触发注册

    device = torch.device("npu:0")
    comm = torchcomms.new_comm("hccl", device, name="test")

    rank = comm.get_rank()
    world_size = comm.get_size()

    # 创建 rank 特征张量
    tensor = torch.full(
        (1024,), float(rank + 1), dtype=torch.float32, device=device
    )

    # AllReduce SUM
    comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

    # 验证结果: sum(1..world_size) = world_size * (world_size + 1) / 2
    expected = world_size * (world_size + 1) / 2
    assert tensor[0].item() == pytest.approx(expected)

    comm.finalize()


@pytest.mark.skipif(not _check_npu_available(), reason="NPU not available")
def test_all_reduce_async():
    """测试异步 AllReduce 操作"""
    import torch
    import torch_npu  # noqa: F401
    import torchcomms
    import torchcomms_hccl  # noqa: F401

    device = torch.device("npu:0")
    comm = torchcomms.new_comm("hccl", device, name="test_async")

    rank = comm.get_rank()

    tensor = torch.ones((512,), dtype=torch.float32, device=device) * rank

    # 异步 AllReduce
    work = comm.all_reduce(
        tensor, torchcomms.ReduceOp.SUM, async_op=True
    )
    work.wait()

    # 验证
    world_size = comm.get_size()
    expected_sum = world_size * (world_size - 1) / 2
    assert tensor[0].item() == pytest.approx(expected_sum)

    comm.finalize()


if __name__ == "__main__":
    # 直接运行:需要设置 RANK 和 WORLD_SIZE 环境变量
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    test_all_reduce_basic()
    print("test_all_reduce_basic passed")
