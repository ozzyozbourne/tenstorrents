#!/usr/bin/env python3
"""Chapter A09 "From Python": one 128x128 tensor placed in DRAM (interleaved), then
height-sharded and width-sharded in L1 over cores (0,0)-(3,0), with a ReLU on each.
Prints each tensor's buffer address, its memory layout, and whether the ReLU matches torch.

Run (workbench env + tt-metal python_env active):  python $W/py/a09_shard.py
"""
import torch, ttnn
torch.manual_seed(0)
device = ttnn.open_device(device_id=0)
x = torch.randn((128, 128), dtype=torch.bfloat16)          # 4x4 tiles
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
cfgs = {
    "DRAM interleaved": ttnn.DRAM_MEMORY_CONFIG,
    "L1 height-sharded": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                          ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)),
    "L1 width-sharded": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                          ttnn.ShardSpec(grid, [128, 32], ttnn.ShardOrientation.ROW_MAJOR)),
}
keep = []                                                  # keep all three tensors allocated
for name, mc in cfgs.items():
    t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    keep.append(t)
    y = ttnn.to_torch(ttnn.relu(t))
    print(f"{name:18s} addr=0x{t.buffer_address():x} layout={t.memory_config().memory_layout} "
          f"match={torch.equal(y, torch.relu(x))}")
print("shard spec:", keep[1].memory_config().shard_spec)
ttnn.close_device(device)
