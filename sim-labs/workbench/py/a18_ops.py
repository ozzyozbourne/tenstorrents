#!/usr/bin/env python3
"""Chapter A18: three ops (add, matmul, softmax) on device tensors, each checked against torch with PCC.

Seeded (42), so the numbers repeat run to run; WH and BH print the same values.
"""
import torch, ttnn
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
torch.manual_seed(42)
device = ttnn.open_device(device_id=0)

# 1. elementwise add
a = torch.randn(64, 64, dtype=torch.bfloat16); b = torch.randn(64, 64, dtype=torch.bfloat16)
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
tc = ttnn.add(ta, tb)
c = ttnn.to_torch(tc); ref = a + b
print(f"add     64x64     PCC {pcc(c, ref):.6f}  max|err| {(c.float()-ref.float()).abs().max():.4f}  out {tc.memory_config().buffer_type}")

# 2. matmul
x = torch.randn(64, 128, dtype=torch.bfloat16); w = torch.randn(128, 96, dtype=torch.bfloat16)
ref = x.float() @ w.float()
tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device)
ty = ttnn.matmul(tx, tw)
y = ttnn.to_torch(ty)
print(f"matmul  64x128@128x96 PCC {pcc(y, ref):.6f}  mean|err| {(y.float()-ref).abs().mean():.4f}  shape {ty.shape}")

# 3. softmax
s = torch.randn(32, 64, dtype=torch.bfloat16)
ts = ttnn.softmax(ttnn.from_torch(s, layout=ttnn.TILE_LAYOUT, device=device), dim=-1)
sr = torch.softmax(s.float(), dim=-1)
print(f"softmax 32x64     PCC {pcc(ttnn.to_torch(ts), sr):.6f}  row sums {ttnn.to_torch(ts).float().sum(-1)[:3].tolist()}")
ttnn.close_device(device)
