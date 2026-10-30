# Chapter A19: per-call time of ttnn.add with the program cache on, off, then on again.
import time, torch, ttnn
dev = ttnn.open_device(device_id=0)
def T(): return ttnn.from_torch(torch.rand((32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
a, b = T(), T()
ttnn.add(a, b); ttnn.synchronize_device(dev)          # warm-up: JIT binaries now on disk
def bench(n=100):
    t0 = time.perf_counter()
    for _ in range(n):
        c = ttnn.add(a, b); ttnn.deallocate(c)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n * 1e3
print(f"cache ON : {bench():6.2f} ms per ttnn.add   entries={dev.num_program_cache_entries()}")
dev.disable_and_clear_program_cache()
print(f"cache OFF: {bench():6.2f} ms per ttnn.add   entries={dev.num_program_cache_entries()}")
dev.enable_program_cache()
print(f"cache ON : {bench():6.2f} ms per ttnn.add   entries={dev.num_program_cache_entries()}")
ttnn.close_device(dev)
