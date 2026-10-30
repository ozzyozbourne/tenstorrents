# Chapter A22: capture the decode MLP (w1, w3, silu*mul, w2) as a trace once, then replay it for new tokens.
# Trace needs FAST dispatch, so run this one WITHOUT TT_METAL_SLOW_DISPATCH_MODE:
#   unset TT_METAL_SLOW_DISPATCH_MODE; python a22_trace_decode_mlp.py
# Pattern from models/tt_transformers/tt/generator.py:2488-2512 (capture) and 2611-2670 (replay):
# allocate inputs once, run once to compile, capture, then for each step copy new host data into the SAME
# device input buffer and execute_trace.
import math, time, torch, ttnn

DIM, FFN = 384, 768
torch.manual_seed(0)
w1, w3, w2 = (torch.randn(a, b) / math.sqrt(a) for a, b in ((DIM, FFN), (DIM, FFN), (FFN, DIM)))
dev = ttnn.open_device(device_id=0, trace_region_size=1 << 20)
grid = ttnn.CoreGrid(y=2, x=6)
def wsh(n): return ttnn.create_sharded_memory_config((32, n // grid.num_cores), grid, ttnn.ShardStrategy.WIDTH,
                                                     ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dev.dram_grid_size().x - 1, 0))})
def dram_sharded(w, dt):
    k, n = w.shape
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                           ttnn.ShardSpec(dram_grid, (k, n // dev.dram_grid_size().x), ttnn.ShardOrientation.ROW_MAJOR))
    return ttnn.from_torch(w[None, None], dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
def pc(k, n): return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    in0_block_w=k // 32 // grid.num_cores, per_core_M=1, per_core_N=n // 32 // grid.num_cores, fused_activation=None)
ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                      fp32_dest_acc_en=False, packer_l1_acc=True)
t_w1, t_w3, t_w2 = dram_sharded(w1, ttnn.bfloat4_b), dram_sharded(w3, ttnn.bfloat4_b), dram_sharded(w2, ttnn.bfloat8_b)

def mlp(x):  # x: [1,1,32,DIM] width-sharded in L1 (decode layout)
    a = ttnn.linear(x, t_w1, program_config=pc(DIM, FFN), memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=ck)
    b = ttnn.linear(x, t_w3, program_config=pc(DIM, FFN), memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=ck)
    m = ttnn.mul(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat8_b, memory_config=a.memory_config())
    m = ttnn.to_memory_config(m, wsh(FFN))
    return ttnn.linear(m, t_w2, program_config=pc(FFN, DIM), memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=ck)
def ref(x): return (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2
def pcc(a, b): return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()
def host(x): return ttnn.from_torch(x[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

x_dev = ttnn.to_device(host(torch.randn(32, DIM)), dev, memory_config=wsh(DIM))   # the persistent input buffer
t0 = time.time(); mlp(x_dev); ttnn.synchronize_device(dev)                         # 1. compile run
print(f"compile run: {time.time() - t0:.1f} s, program cache entries = {dev.num_program_cache_entries()}")
tid = ttnn.begin_trace_capture(dev, cq_id=0)                                       # 2. capture
y_dev = mlp(x_dev)
ttnn.end_trace_capture(dev, tid, cq_id=0)
print("trace captured: 5 ops (w1, w3, fused silu*mul, reshard, w2)")
for step in range(3):                                                              # 3. replay per token
    x = torch.randn(32, DIM)
    ttnn.copy_host_to_device_tensor(host(x), x_dev)                                # new data, same address
    t0 = time.time()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = time.time() - t0
    print(f"replay {step}: PCC vs torch = {pcc(ttnn.to_torch(y_dev)[0, 0], ref(x)):.4f}  ({dt:.2f} s simulator wall time)")
eager = ttnn.to_torch(mlp(x_dev))                                                 # same input, no trace
print(f"last replay bit-identical to an eager run on the same input: {torch.equal(eager, ttnn.to_torch(y_dev))}")
ttnn.release_trace(dev, tid)
ttnn.close_device(dev)
