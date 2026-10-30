# B11: does ttsim reproduce tt-metal's warning that, on Wormhole, HiFi4 + fp32 accumulation can be worse than HiFi3?
import torch, ttnn
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
a = torch.randn(128, 1024).bfloat16(); b = torch.randn(1024, 128).bfloat16()
ref = a.double() @ b.double()
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=dev); tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev)
outs = {}
for fid in ("HiFi2", "HiFi3", "HiFi4"):
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, packer_l1_acc=False)
    o = ttnn.to_torch(ttnn.matmul(ta, tb, dtype=ttnn.float32, compute_kernel_config=ck)).double()
    outs[fid] = o
    print(f"{fid}: mean|err| {(o - ref).abs().mean().item():.3e}  max|err| {(o - ref).abs().max().item():.3e}")
print("HiFi3 == HiFi4 bit for bit:", torch.equal(outs["HiFi3"], outs["HiFi4"]),
      " elements that differ:", int((outs["HiFi3"] != outs["HiFi4"]).sum()), "of", outs["HiFi3"].numel())
import os; torch.save(outs, os.environ["B11_SAVE"]) if "B11_SAVE" in os.environ else None
ttnn.close_device(dev)
