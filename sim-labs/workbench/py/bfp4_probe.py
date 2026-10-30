import torch, ttnn
# One 32x32 tile; row 0, cols 0-15 is face 0 row 0 = one BFP8 block (16 values, one shared exponent).
row = [1.0, 1.0078125, 0.5078125, 0.0078125, 0.01, 0.005, -0.75, 0.3,
       0.1, -0.1, 0.0625, 0.03, 1.9921875, -1.5, 0.2, 0.001]
x = torch.zeros((32, 32), dtype=torch.bfloat16)
x[0, :16] = torch.tensor(row, dtype=torch.bfloat16)
device = ttnn.open_device(device_id=0)
host_q = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))   # host packer (C++)
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
dev_q = ttnn.to_torch(ttnn.typecast(t, ttnn.bfloat4_b))                                      # device packer (ttsim)
print(f"{'input(bf16)':>14} {'host bfp4_b':>12} {'device bfp4_b':>14}")
for i in range(16):
    a, h, d = x[0, i].item(), host_q[0, i].item(), dev_q[0, i].item()
    print(f"{a:14.9g} {h:12.9g} {d:14.9g}{'   <-- differ' if h != d else ''}")
ttnn.close_device(device)
