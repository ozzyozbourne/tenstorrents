# Lab B10, step 4c: ttnn.all_gather across the 2-chip wh_x2 mesh (needs TT-Fabric, so it also needs the
# B10 fabric-router overlay in TT_METAL_KERNEL_PATH). Each chip starts with a different 32x32 tile; after the
# all-gather each chip should hold both, concatenated along the last dim.
import sys
import torch
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 1))
print("mesh devices:", mesh.get_num_devices(), "shape:", mesh.shape)

# Global tensor [1,1,32,64]: columns 0..31 hold 1.0 + small ramp, columns 32..63 hold 2.0 + ramp.
x = torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16)
x[..., :32] = 1.0
x[..., 32:] = 2.0
x += (torch.arange(32 * 64).reshape(1, 1, 32, 64) % 7).to(torch.bfloat16) / 8
# Shard along dim 3: chip 0 gets columns 0..31, chip 1 gets columns 32..63.
t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3))
for i, s in enumerate(ttnn.get_device_tensors(t)):
    print(f"before: chip {i} holds shape {tuple(s.shape)}, [0,0,0,:3] = {ttnn.to_torch(s)[0,0,0,:3].tolist()}")
g = ttnn.all_gather(t, dim=3)
ok = True
for i, s in enumerate(ttnn.get_device_tensors(g)):
    y = ttnn.to_torch(s)
    same = torch.equal(y, x)
    ok &= same
    print(f"after:  chip {i} holds shape {tuple(y.shape)}, equals the full input: {same}")
print("PASS" if ok else "FAIL")
ttnn.close_mesh_device(mesh)
sys.exit(0 if ok else 1)
