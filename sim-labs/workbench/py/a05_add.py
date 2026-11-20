"""A05: a 32x32 bfloat16 ttnn.add compared with PyTorch's own bfloat16 add.

The device result is close to torch but not bit-identical: the device adds in the FPU
and its packer rounds the result to bfloat16 its own way. The script counts how many
of the 1,024 elements differ, the largest difference, and the PCC against the fp32 sum.

Run (workbench env + tt-metal python_env active):  python $W/py/a05_add.py
"""
import torch, ttnn

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
a = torch.randn(32, 32).bfloat16()
b = torch.randn(32, 32).bfloat16()
ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.to_torch(ttnn.add(ta, tb)).float()
ttnn.close_device(device)

ref_bf16 = (a + b).float()                  # torch adds two bf16 tensors, result rounded to bf16
ref_fp32 = a.float() + b.float()            # the exact sum of the same bf16 inputs
diff = (out - ref_bf16).abs()
pcc = torch.corrcoef(torch.stack([out.flatten(), ref_fp32.flatten()]))[0, 1].item()
print("device result[0,:4]:", out[0, :4].tolist())
print("torch  result[0,:4]:", ref_bf16[0, :4].tolist())
print("max abs diff vs torch bf16 add:", diff.max().item())
print(f"elements differing from torch bf16 add: {int((diff > 0).sum())}/{out.numel()}")
print("max abs diff vs exact fp32 sum:", (out - ref_fp32).abs().max().item())
print(f"PCC vs exact fp32 sum: {pcc:.6f}")
# how big is each difference, in units of the result's bf16 spacing (ulp)?
ulp = torch.pow(2.0, torch.floor(torch.log2(ref_bf16.abs().clamp_min(1e-30))) - 7)
print("largest difference in ulps:", (diff / ulp).max().item())
print("PASS" if pcc > 0.9999 else "FAIL")
