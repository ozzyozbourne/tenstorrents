# A20: run ONE ttnn op (or only its setup) so that an instrumented ttsim can count its traffic.
#   python a20_count.py <case> [prep]
# With "prep" the script does all the setup (device tensors, resharding) but skips the op,
# so (counters with op) - (counters with prep) = the op's own traffic.
import sys, torch, ttnn

case = sys.argv[1]
prep_only = len(sys.argv) > 2 and sys.argv[2] == "prep"

def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                            fp32_dest_acc_en=False, packer_l1_acc=False)

FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2,
       "hifi3": ttnn.MathFidelity.HiFi3, "hifi4": ttnn.MathFidelity.HiFi4}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

torch.manual_seed(0)
device = ttnn.open_device(device_id=0)
M, K, N = 128, 512, 512                       # Mt=4, Kt=16, Nt=16 tiles
x = torch.randn(M, K); w = torch.randn(K, N)
ref = x @ w
out = None

if case.startswith("mm_"):                     # mm_<wdtype>_<fidelity>, default program config
    _, wdt, fid = case.split("_")
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=DT[wdt], layout=ttnn.TILE_LAYOUT, device=device)
    if not prep_only:
        out = ttnn.matmul(tx, tw, compute_kernel_config=ckc(FID[fid]))
elif case.startswith("blk_m"):                 # blk_m<pm>: no multicast; each core owns per_core_M rows x all N columns
    pm = int(case[5:])
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 8), in0_block_w=2,
                                                out_subblock_h=1, out_subblock_w=8,
                                                per_core_M=pm, per_core_N=N // 32)
    print(f"A20CFG per_core_M={pm} per_core_N={N // 32} cores={(M // 32) // pm}")
    if not prep_only:
        out = ttnn.matmul(tx, tw, program_config=pc, compute_kernel_config=ckc(FID["hifi2"]))
elif case in ("fuse_no", "fuse_yes"):          # matmul + relu as two ops, or relu fused into the matmul
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    ref = torch.relu(ref)
    if not prep_only:
        if case == "fuse_no":
            out = ttnn.relu(ttnn.matmul(tx, tw, compute_kernel_config=ckc(FID["hifi2"])))
        else:
            out = ttnn.linear(tx, tw, activation="relu", compute_kernel_config=ckc(FID["hifi2"]))
elif case in ("grid_noact", "grid_relu"):    # same, but with a user core_grid: now the activation is fused
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    act = "relu" if case == "grid_relu" else None
    if act: ref = torch.relu(ref)
    if not prep_only:
        out = ttnn.linear(tx, tw, activation=act, core_grid=ttnn.CoreGrid(y=4, x=8),
                          compute_kernel_config=ckc(FID["hifi2"]))
elif case in ("add_dram", "add_l1shard"):      # an elementwise add of two 512x512 bf16 tensors
    a = torch.randn(512, 512); b = torch.randn(512, 512); ref = a + b
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mc = None
    if case == "add_l1shard":
        mc = ttnn.create_sharded_memory_config((512, 512), core_grid=ttnn.CoreGrid(y=8, x=2),
                                               strategy=ttnn.ShardStrategy.HEIGHT,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR)
        ta = ttnn.to_memory_config(ta, mc); tb = ttnn.to_memory_config(tb, mc)
    if not prep_only:
        out = ttnn.add(ta, tb, memory_config=mc) if mc else ttnn.add(ta, tb)
else:
    raise SystemExit("unknown case " + case)

if out is not None:
    got = ttnn.to_torch(out).float()
    print(f"A20RESULT case={case} shape={tuple(got.shape)} dtype={out.dtype} PCC={pcc(got, ref):.5f}")
else:
    print(f"A20RESULT case={case} prep only")
ttnn.close_device(device)
