# B11 section 5: compare two tensor dictionaries saved by b11_s3_hifi.py with B11_SAVE=<file>.pt.
#   python b11_s3_compare.py wh.pt bh.pt
# Prints, per fidelity, whether the two runs agree bit for bit and how many elements differ.
# Needs only torch (no device), so it runs under either simulator's environment.
import sys, torch
if len(sys.argv) != 3:
    raise SystemExit("usage: python b11_s3_compare.py <wormhole.pt> <blackhole.pt>")
wh, bh = torch.load(sys.argv[1]), torch.load(sys.argv[2])
for fid in sorted(wh):
    print(f"{fid} WH==BH bitwise:", torch.equal(wh[fid], bh[fid]), int((wh[fid] != bh[fid]).sum()))
