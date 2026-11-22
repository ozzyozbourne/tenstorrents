#!/usr/bin/env python3
"""Chapter A19: run exactly one TT-NN op, so that an empty JIT cache afterwards holds only that op's kernels.

  export TT_METAL_CACHE=$LAB_HOME/jit_add && rm -rf $TT_METAL_CACHE
  python -u $W/py/a19_oneop.py add        # or: matmul, addf32

add    = ttnn.add on 32x32 bfloat16;  matmul = ttnn.matmul on 64x64 bfloat16;
addf32 = ttnn.add on 32x32 float32 (SFPU kernel; ttsim v1.15.0 stops with UndefinedBehavior on Wormhole).
The PCC is against the torch result on the fp32 inputs. Seeded (0).
"""
import sys, torch, ttnn
op = sys.argv[1]
torch.manual_seed(0)
device = ttnn.open_device(device_id=0)
def pcc(a, b): return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()
if op == "add":
    a = torch.randn(32, 32); b = torch.randn(32, 32)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"add PCC {pcc(ttnn.to_torch(ttnn.add(ta, tb)), a + b):.5f}", flush=True)
elif op == "matmul":
    a = torch.randn(64, 64); b = torch.randn(64, 64)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"matmul PCC {pcc(ttnn.to_torch(ttnn.matmul(ta, tb)), a @ b):.5f}", flush=True)
elif op == "addf32":
    a = torch.randn(32, 32); b = torch.randn(32, 32)
    ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    print("calling float32 add", flush=True)
    print(f"add f32 PCC {pcc(ttnn.to_torch(ttnn.add(ta, tb)), a + b):.5f}", flush=True)
print("program cache entries:", device.num_program_cache_entries(), flush=True)
ttnn.close_device(device)
