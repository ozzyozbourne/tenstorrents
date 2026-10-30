#!/usr/bin/env python3
"""Chapter A18: a 128x256 @ 256x128 bf16 matmul run six ways (auto config, compute kernel configs, a
MatmulMultiCoreReuseMultiCastProgramConfig on 2x2 cores) to expose the silent LoFi default, then a two-layer
torch MLP ported to ttnn.linear with bfloat8_b weights.

Seeded (7), so the numbers repeat run to run.
"""
import torch, ttnn
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
torch.manual_seed(7)
device = ttnn.open_device(device_id=0)
M, K, N = 128, 256, 128                      # Mt=4, Kt=8, Nt=4 tiles
x = torch.randn(M, K); w = torch.randn(K, N); ref = x @ w
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(2, 2),   # 2x2 cores
    in0_block_w=2,                           # K is consumed 2 tiles at a time (divides Kt=8)
    out_subblock_h=1, out_subblock_w=2,      # 1x2 tiles of Dst per sub-block
    per_core_M=2, per_core_N=2,              # each core owns a 2x2-tile output block
    transpose_mcast=False, fused_activation=None)
hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                         math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
hifi2 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                         math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False)
hifi2acc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                         math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
runs = [("default (auto config)", dict()),
        ("auto + HiFi2", dict(compute_kernel_config=hifi2)),
        ("auto + HiFi2 + l1_acc", dict(compute_kernel_config=hifi2acc)),
        ("program_config + HiFi2", dict(program_config=pc, compute_kernel_config=hifi2)),
        ("program_config only", dict(program_config=pc)),
        ("program_config + HiFi4", dict(program_config=pc, compute_kernel_config=hifi4))]
for name, kw in runs:
    y = ttnn.to_torch(ttnn.matmul(tx, tw, **kw)).float()
    print(f"{name:24s} PCC {pcc(y, ref):.6f}  mean|err| {(y-ref).abs().mean():.4f}")

# a torch module, mapped by hand
mlp = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).eval()
with torch.no_grad():
    want = mlp(x)
W1 = ttnn.from_torch(mlp[0].weight.T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
b1 = ttnn.from_torch(mlp[0].bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
W2 = ttnn.from_torch(mlp[2].weight.T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
b2 = ttnn.from_torch(mlp[2].bias.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
h = ttnn.linear(tx, W1, bias=b1, activation="gelu")
out = ttnn.linear(h, W2, bias=b2)
got = ttnn.to_torch(out).float()
print(f"MLP 128x256 -> 512 -> 256 (bfp8 weights)  PCC {pcc(got, want):.6f}")
ttnn.close_device(device)
