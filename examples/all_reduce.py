"""AllReduce 示例 / AllReduce example

启动方式 / Run:

    torchrun --nproc_per_node=2 examples/all_reduce.py
"""

import torch
import torch_npu  # noqa: F401  注册 NPU 设备
import torchcomms
import torchcomms_hccl  # noqa: F401  注册 HCCL 后端

device = torch.device("npu")
comm = torchcomms.new_comm("hccl", device, name="demo")

rank = comm.get_rank()
world_size = comm.get_size()

tensor = torch.full((4,), float(rank + 1), dtype=torch.float32, device=device)
print(f"Before  rank={rank}: {tensor.tolist()}")

work = comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=True)
work.wait()

print(f"After   rank={rank}: {tensor.tolist()}")
print(f"Expected: [{float(sum(r + 1 for r in range(world_size)))}]")

comm.finalize()
