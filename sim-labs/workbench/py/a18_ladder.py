#!/usr/bin/env python3
"""Chapter A18: the dtype x fidelity ladder. A 256x128 random weight stored as bfloat16 / bfloat8_b / bfloat4_b,
first round-tripped on the host, then used in a 64x256 @ 256x128 matmul at HiFi4, HiFi2 and LoFi.

Errors are against an fp32 torch matmul. Seeded (42), so the numbers repeat run to run.
"""
import torch, ttnn
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
torch.manual_seed(42)
device = ttnn.open_device(device_id=0)
x = torch.randn(64, 256); w = torch.randn(256, 128)
ref = x @ w                                                # fp32 reference
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

print("-- weight storage only (host round trip) --")
for dt in [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b]:
    wq = ttnn.to_torch(ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()
    print(f"{str(dt):22s} PCC {pcc(wq, w):.6f}  mean|err| {(wq-w).abs().mean():.5f}")

print("-- matmul 64x256 @ 256x128, x in bf16 --")
for dt in [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b]:
    for fid in [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi]:
        tw = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                               fp32_dest_acc_en=False, packer_l1_acc=False)
        y = ttnn.to_torch(ttnn.matmul(tx, tw, compute_kernel_config=cfg)).float()
        print(f"w={str(dt):20s} {str(fid):22s} PCC {pcc(y, ref):.6f}  mean|err| {(y-ref).abs().mean():.4f}  page {tw.buffer_page_size()} B")
ttnn.close_device(device)
