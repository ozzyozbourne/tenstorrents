# Chapter A19: watch the TT-NN program cache grow (or not) as ttnn.add/multiply/matmul are called
# with the same and different shapes, dtypes and memory configs. Run after `source env.sh wh`.
import time, torch, ttnn
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
print("program cache enabled at open:", dev.num_program_cache_entries(), "entries")
def T(shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.rand(shape, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
def run(label, fn):
    t0 = time.time(); out = fn(); ttnn.synchronize_device(dev); dt = time.time() - t0
    print(f"{label:<34} entries={dev.num_program_cache_entries():<2} wall={dt:5.2f}s  out@0x{out.buffer_address():x}")
    return out
a, b = T((32, 32)), T((32, 32))
run("add 32x32   (1st call)", lambda: ttnn.add(a, b))
run("add 32x32   (2nd call, same shape)", lambda: ttnn.add(a, b))
c, d = T((64, 128)), T((64, 128))
run("add 64x128  (new shape)", lambda: ttnn.add(c, d))
run("mul 32x32   (new op type)", lambda: ttnn.multiply(a, b))
e, f = T((32, 32), ttnn.bfloat8_b), T((32, 32), ttnn.bfloat8_b)
run("add 32x32 bfp8 (new dtype)", lambda: ttnn.add(e, f))
run("add 32x32 out in L1 (new mem cfg)", lambda: ttnn.add(a, b, memory_config=ttnn.L1_MEMORY_CONFIG))
m1, m2 = T((32, 64)), T((64, 32))
run("matmul 32x64 @ 64x32 (1st)", lambda: ttnn.matmul(m1, m2))
run("matmul 32x64 @ 64x32 (2nd)", lambda: ttnn.matmul(m1, m2))
m3, m4 = T((64, 64)), T((64, 64))
run("matmul 64x64 @ 64x64 (new shape)", lambda: ttnn.matmul(m3, m4))
x = run("add 64x128  again", lambda: ttnn.add(c, d))
print("check add 64x128 vs torch, max abs diff:",
      (ttnn.to_torch(x).float() - (ttnn.to_torch(c).float() + ttnn.to_torch(d).float())).abs().max().item())
ttnn.close_device(dev)
