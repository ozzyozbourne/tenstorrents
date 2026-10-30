#!/usr/bin/env python3
"""Chapter A18: matmul error vs math fidelity with a bf16 Dst, against the fp32 product of the same bf16 inputs,
then the same with fp32_dest_acc_en=True. Under ttsim v1.15.0 the first FP32-Dst matmul aborts the process with
"UnsupportedFunctionality: tensix_unpacr: cfg_context_id=1" (expected).

Seeded (42). Prints are flushed so the lines before the abort survive a pipe.
"""
import torch, ttnn
torch.manual_seed(42)
device = ttnn.open_device(device_id=0)
x = torch.randn(64, 256); w = torch.randn(256, 128); ref = x.to(torch.bfloat16).float() @ w.to(torch.bfloat16).float()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
print("reference = fp32 matmul of the bf16-rounded inputs", flush=True)
for fp32 in [False, True]:
    for fid in [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4]:
        cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                               fp32_dest_acc_en=fp32, packer_l1_acc=False)
        y = ttnn.to_torch(ttnn.matmul(tx, tw, compute_kernel_config=cfg)).float()
        print(f"fp32_dest_acc_en={fp32!s:5} {str(fid):20s} mean|err| {(y-ref).abs().mean():.5f}  max|err| {(y-ref).abs().max():.4f}", flush=True)
ttnn.close_device(device)
