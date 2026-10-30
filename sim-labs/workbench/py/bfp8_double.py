import torch, ttnn
row = [1.0, 0.50390625, 0.5078125, 1.9921875, 0.30078125, 0.2001953125, -0.75, 0.0078125] + [0.0]*8
x = torch.zeros((32, 32), dtype=torch.bfloat16); x[0, :16] = torch.tensor(row, dtype=torch.bfloat16)
device = ttnn.open_device(device_id=0)
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
z = ttnn.from_torch(torch.zeros_like(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
a = ttnn.to_torch(ttnn.typecast(t, ttnn.bfloat8_b))
b = ttnn.to_torch(ttnn.add(t, z, dtype=ttnn.bfloat8_b))
for i in range(8): print(f"{x[0,i].item():>14.10g} typecast={a[0,i].item():<10g} add(x,0)={b[0,i].item():g}")
ttnn.close_device(device)
