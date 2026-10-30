# B11 setup probes: one case per process, because a ttsim error ends the process.
import sys, torch, ttnn
case = sys.argv[1]
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()
if case == "sum_fp32":
    x = torch.randn(32, 32)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    print("sum", ttnn.to_torch(ttnn.sum(t)).item(), "torch", x.sum().item())
elif case == "mm_fp32acc":
    a = torch.randn(64, 64); b = torch.randn(64, 64)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    print("PCC", pcc(ttnn.to_torch(ttnn.matmul(ta, tb, compute_kernel_config=ck)).float(), a @ b))
elif case == "add_fp32":
    a = torch.randn(32, 32); b = torch.randn(32, 32)
    ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(ttnn.add(ta, tb))
    print("add_fp32 max|err|", (o - (a + b)).abs().max().item(), "bit-identical", torch.equal(o, a + b))
elif case == "trace":
    a = torch.randn(32, 32)
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.relu(ta)                       # compile outside the trace
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = ttnn.relu(ta)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    print("trace replay ok, PCC", pcc(ttnn.to_torch(out).float(), torch.relu(a)))
    ttnn.release_trace(dev, tid)
print("PROBE DONE", case)
ttnn.close_device(dev)
