#!/usr/bin/env python3
"""Chapter A18: two invalid shard specs. 128 rows height-sharded over 3 cores (rejected by the Python helper),
and a hand-built ShardSpec with 16-row shards for a TILE tensor (rejected by a TT_FATAL in TensorLayout).
"""
import torch, ttnn
device = ttnn.open_device(device_id=0)
a = torch.randn(128, 128, dtype=torch.bfloat16)
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
# 1. 128 rows over 3 cores
try:
    mc = ttnn.create_sharded_memory_config((128, 128), core_grid=ttnn.CoreGrid(y=3, x=1), strategy=ttnn.ShardStrategy.HEIGHT)
except Exception as e:
    print("3-core height shard helper FAILED:", e)
# 2. a hand-written shard shape that is not a multiple of 32 in TILE layout
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))})
    mc2 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, [16, 128], ttnn.ShardOrientation.ROW_MAJOR))
    s2 = ttnn.to_memory_config(ta, mc2); print("16-row shards: ran", s2.memory_config().shard_spec)
except Exception as e:
    msg = str(e)
    info = msg.split("info:")[1].split("backtrace")[0].strip() if "info:" in msg else msg.splitlines()[0]
    print("16-row shards FAILED: TT_FATAL:", info)
ttnn.close_device(device)
