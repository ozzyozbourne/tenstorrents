#!/usr/bin/env python3
"""Chapter A18: call ttnn.add three times and watch the program cache, then list the kernel binaries the JIT built.

Run it with an empty private kernel cache so the first call has to compile:
  export TT_METAL_CACHE=$LAB_HOME/jit && rm -rf $TT_METAL_CACHE
tt-metal logs "JIT cache stats: 0/13 hits" when the device closes. Wall times are the simulator's, on this CPU.
"""
import glob, os, time, torch, ttnn
torch.manual_seed(0)
device = ttnn.open_device(device_id=0)
a = torch.randn(64, 64, dtype=torch.bfloat16); b = torch.randn(64, 64, dtype=torch.bfloat16)
ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
for i in range(3):
    t0 = time.time(); tc = ttnn.add(ta, tb); ttnn.synchronize_device(device)
    print(f"ttnn.add call {i+1}: {time.time()-t0:.2f} s wall   program cache entries: {device.num_program_cache_entries()}", flush=True)
ttnn.close_device(device)
cache = os.environ.get("TT_METAL_CACHE")
if cache:   # kernel binaries: <cache>/<build key>/kernels/<kernel name>/<hash>/<risc>/
    for p in sorted(glob.glob(os.path.join(cache, "*", "kernels", "*", "*", "*", ""))):
        print("/".join(p.rstrip("/").split("/")[-3:]))
