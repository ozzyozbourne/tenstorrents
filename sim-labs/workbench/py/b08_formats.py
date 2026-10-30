# Lab B08: round-trip the same FP32 values through float32, bfloat16, bfloat8_b and bfloat4_b,
# once with the host converter (ttnn.from_torch) and once with the device packer (ttnn.typecast
# on the simulated device), and show where precision goes.
# B08_SRC=fp32 (default) puts an FP32 tile on the device; B08_SRC=bf16 puts the BF16-rounded tile
# there instead (on Wormhole ttsim, typecast from FP32 stops with UndefinedBehavior; see lab B08).
import os, torch, ttnn
SRC = ttnn.bfloat16 if os.environ.get("B08_SRC", "fp32") == "bf16" else ttnn.float32

FMTS = [("float32", ttnn.float32), ("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b), ("bfloat4_b", ttnn.bfloat4_b)]
device = ttnn.open_device(device_id=0)

def host_rt(x, dt):    # host packer (C++), then back to float32
    return ttnn.to_torch(ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT)).float()

def dev_rt(x, dt):     # FP32 (or BF16) tile on device, device packer writes dt, read back
    t = ttnn.from_torch(x, dtype=SRC, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(t if dt == SRC else ttnn.typecast(t, dt)).float()

# Part 1: one block = face 0, row 0, columns 0-15 (16 values sharing one exponent in BFP formats).
block = [3.0, 1.2345678, 0.7, -0.4, 0.3, 0.15, -0.1, 0.06, 0.03, 0.012, 0.007, -0.003, 0.0015, 1e-4, 2.5, -1.75]
x = torch.zeros((32, 32), dtype=torch.float32)
x[0, :16] = torch.tensor(block)
res = {n: (host_rt(x, d), dev_rt(x, d)) for n, d in FMTS}
print("Part 1: one 16-value block, host | device")
print(f"{'input fp32':>14} " + " ".join(f"{n:>21}" for n, _ in FMTS))
for i in range(16):
    cells = []
    for n, _ in FMTS:
        h, d = res[n][0][0, i].item(), res[n][1][0, i].item()
        cells.append(f"{h:10.6g}|{d:<10.6g}")
    print(f"{x[0, i].item():14.8g} " + " ".join(cells))

# Part 2: a random N(0,1) tile, then the same tile with one outlier (64.0) at [0,0].
def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
g = torch.Generator().manual_seed(0)
r = torch.randn((32, 32), generator=g)
for label, t in [("random N(0,1)", r), ("same + outlier 64 at [0,0]", r.clone().index_put_((torch.tensor([0]), torch.tensor([0])), torch.tensor(64.0)))]:
    print(f"\nPart 2: {label}")
    print(f"{'format':>10} {'path':>6} {'max_abs':>10} {'mean_rel':>10} {'PCC':>9} {'blk0 mean_rel':>14} {'blk1 mean_rel':>14}")
    for n, d in FMTS:
        for path, q in (("host", host_rt(t, d)), ("device", dev_rt(t, d))):
            rel = ((q - t).abs() / t.abs().clamp_min(1e-30))
            print(f"{n:>10} {path:>6} {(q - t).abs().max().item():10.3g} {rel.mean().item():10.3g} {pcc(q, t):9.6f}"
                  f" {rel[0, 1:16].mean().item():14.3g} {rel[0, 16:32].mean().item():14.3g}")
ttnn.close_device(device)
