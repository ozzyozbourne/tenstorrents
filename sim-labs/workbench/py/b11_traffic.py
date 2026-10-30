"""B11: one TT-NN matmul (or only its setup) per process, for counting NoC traffic in an instrumented ttsim.

    python b11_traffic.py <dram|l1|height|block> [prep]

x [256, 256] bf16 is placed with the given memory config, w [256, 256] bf16 stays DRAM-interleaved, and the
output uses the same memory config as x. With "prep" the script uploads the tensors but skips the matmul,
so (counts with the op) - (counts with prep) is the op's own traffic. Run it under a ttsim built with
../patches/a20-counters.patch; the counters are printed when the simulator exits.
"""
import sys, torch, ttnn

mem = sys.argv[1]
prep = len(sys.argv) > 2 and sys.argv[2] == "prep"
M = K = N = 256
torch.manual_seed(0)
x, w = torch.randn(M, K), torch.randn(K, N) / K ** 0.5
dev = ttnn.open_device(device_id=0)
if mem == "dram":
    mc = ttnn.DRAM_MEMORY_CONFIG
elif mem == "l1":
    mc = ttnn.L1_MEMORY_CONFIG
elif mem == "height":                                # 8 cores, 32 rows (8 tiles along K) each
    mc = ttnn.create_sharded_memory_config((M, K), core_grid=ttnn.CoreGrid(y=8, x=1),
                                           strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
elif mem == "block":                                 # 4x4 cores, 64x64 (2x2 tiles) each
    mc = ttnn.create_sharded_memory_config((M, K), core_grid=ttnn.CoreGrid(y=4, x=4),
                                           strategy=ttnn.ShardStrategy.BLOCK, orientation=ttnn.ShardOrientation.ROW_MAJOR)
else:
    raise SystemExit(__doc__)
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
if not prep:
    out = ttnn.matmul(tx, tw, memory_config=mc)
    got = ttnn.to_torch(out).float()
    ref = x.bfloat16().float() @ w.bfloat16().float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    print(f"B11TRAFFIC mem={mem} out={out.memory_config().memory_layout} PCC={pcc:.6f}", flush=True)
else:
    print(f"B11TRAFFIC mem={mem} prep only", flush=True)
ttnn.close_device(dev)
