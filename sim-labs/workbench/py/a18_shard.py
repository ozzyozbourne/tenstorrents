#!/usr/bin/env python3
"""Chapter A18: a 128x128 tensor in DRAM and L1 interleaved, then height-, width- and block-sharded in L1 with
ttnn.create_sharded_memory_config, added in place and checked. Prints the shard grids and shapes.

Seeded (1). Addresses and the compute grid differ between WH and BH; shard specs and PCCs do not.
"""
import torch, ttnn
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
torch.manual_seed(1)
device = ttnn.open_device(device_id=0)
print("compute grid:", device.compute_with_storage_grid_size())
a = torch.randn(128, 128, dtype=torch.bfloat16); b = torch.randn(128, 128, dtype=torch.bfloat16)
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)          # DRAM interleaved
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
print("interleaved DRAM: pages", ta.buffer_num_pages(), "addr", hex(ta.buffer_address()))
l1 = ttnn.to_memory_config(ta, ttnn.L1_MEMORY_CONFIG)
print("interleaved L1  : addr", hex(l1.buffer_address()), l1.memory_config().buffer_type)

cases = {
  "HEIGHT": (ttnn.CoreGrid(y=4, x=1), ttnn.ShardStrategy.HEIGHT),
  "WIDTH":  (ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
  "BLOCK":  (ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK),
}
for name, (grid, strat) in cases.items():
    try:
        mc = ttnn.create_sharded_memory_config((128, 128), core_grid=grid, strategy=strat,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR)
        sa = ttnn.to_memory_config(ta, mc); sb = ttnn.to_memory_config(tb, mc)
        ss = sa.memory_config().shard_spec
        sc = ttnn.add(sa, sb, memory_config=mc)
        c = ttnn.to_torch(sc)
        print(f"{name:6s} layout {sc.memory_config().memory_layout}  grid {ss.grid}  shard {ss.shape}  "
              f"PCC {pcc(c, a+b):.6f}")
    except Exception as e:
        print(name, "FAILED:", str(e).splitlines()[0:3])
ttnn.close_device(device)
