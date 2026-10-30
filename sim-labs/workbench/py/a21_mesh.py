# usage: python a21_mesh.py add | gather | fabric-gather
import sys, torch, ttnn
mode = sys.argv[1]
if mode == "fabric-gather":
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)   # must precede open_mesh_device
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 1))
print("mesh:", mesh.shape, "devices:", mesh.get_num_devices())
torch.manual_seed(0)
a = torch.rand((1, 1, 64, 32), dtype=torch.bfloat16)
b = torch.rand((1, 1, 64, 32), dtype=torch.bfloat16)
shard = ttnn.ShardTensorToMesh(mesh, dim=2)               # rows 0-31 -> device 0, rows 32-63 -> device 1
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=shard)
print("per-device shape:", ta.shape)
tc = ttnn.add(ta, tb)                                      # one call, runs on both chips
c = ttnn.to_torch(tc, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2))
ref = a + b
pcc = torch.corrcoef(torch.stack([c.flatten().float(), ref.flatten().float()]))[0, 1].item()
print("composed shape:", tuple(c.shape), f"PCC {pcc:.6f}", "max|diff|", (c.float() - ref.float()).abs().max().item())
if mode != "add":
    g = ttnn.all_gather(tc, dim=2)                         # every device should end with all 64 rows
    print("all_gather per-device shape:", g.shape)
ttnn.close_mesh_device(mesh)
