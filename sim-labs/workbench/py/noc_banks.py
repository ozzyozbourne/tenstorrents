#!/usr/bin/env python3
"""Summarise a B05 NoC log (ttsim built with patches/b05-noclog.patch) by Wormhole DRAM bank.

    python3 -I noc_banks.py $LAB_HOME/loopback.log [soc_descriptor.yaml]

The bank table (which DRAM tile and which 1 GiB half form each bank) is read from the
soc descriptor's dram / dram_views sections, the same source tt-metal's allocator uses.
It then checks every transfer against A09's interleaving rule:
page p lives in bank p % 12 at row p // 12, i.e. at base + (p // 12) * page_size.
"""
import os, re, sys
from collections import defaultdict

log = sys.argv[1]
soc = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.environ.get("TT_METAL_HOME", ""), "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml")

# --- bank table from the soc descriptor -----------------------------------------------------------
text = open(soc).read()
dram_block = re.search(r"^dram:\s*\[(.*?)^\s*\]", text, re.S | re.M).group(1)
controllers = [re.findall(r"(\d+)-(\d+)", row) for row in re.findall(r"\[([^\]]*)\]", dram_block)]
views = re.findall(r"channel:\s*(\d+),.*?worker_endpoint:\s*\[(\d+),.*?address_offset:\s*(\d+)", text, re.S)
banks = {}  # (x, y, offset) -> bank id
for bank, (ch, ep, off) in enumerate(views):
    x, y = controllers[int(ch)][int(ep)]
    banks[(int(x), int(y), int(off))] = bank
GIB = 1 << 30

# --- parse the log ---------------------------------------------------------------------------------
pat = re.compile(r"NOCLOG (READ|WRITE)\s+noc=(\d) core=\S+ buf=\d \S+\s+src=(\w)\((\d+),(\d+)\):0x([0-9a-f]+) "
                 r"dst=(\w)\((\d+),(\d+)\):0x([0-9a-f]+) len=(\d+)")
rows = []
for line in open(log):
    m = pat.search(line)
    if not m:
        continue
    kind = m.group(1)
    # the DRAM end is the source of a read and the destination of a write
    t, x, y, a = (m.group(3, 4, 5, 6) if kind == "READ" else m.group(7, 8, 9, 10))
    if t != "D":
        continue
    a = int(a, 16)
    off = GIB if a >= GIB else 0
    rows.append((kind, int(x), int(y), off, a - off, int(m.group(11))))

stats = defaultdict(lambda: {"READ": [0, 0], "WRITE": [0, 0]})
order = {"READ": [], "WRITE": []}
for kind, x, y, off, local, n in rows:
    b = banks[(x, y, off)]
    stats[b][kind][0] += 1
    stats[b][kind][1] += n
    order[kind].append((b, local))

print(f"{len(rows)} DRAM transfers: {len(order['READ'])} reads, {len(order['WRITE'])} writes\n")
print("bank  tile    half   reads  bytes_rd  writes  bytes_wr")
for b in sorted(stats):
    (x, y, off), = [k for k, v in banks.items() if v == b]
    r, w = stats[b]["READ"], stats[b]["WRITE"]
    print(f"{b:4d}  ({x},{y:<2d})  {'+1GiB' if off else '+0   '}  {r[0]:5d}  {r[1]:8d}  {w[0]:6d}  {w[1]:8d}")

nb = len(banks)
for kind in ("READ", "WRITE"):
    seq = [b for b, _ in order[kind]]
    print(f"\n{kind.lower()} bank order: {' '.join(map(str, seq[:2 * nb]))} ...")
    base = min(a for _, a in order[kind])
    page = rows[0][5]
    ok = all(b == p % nb and a == base + (p // nb) * page for p, (b, a) in enumerate(order[kind]))
    print(f"  page p -> bank p % {nb}, address base + (p // {nb}) * {page}  (base 0x{base:x}): "
          f"{'holds for all ' + str(len(seq)) + ' pages' if ok else 'DOES NOT HOLD'}")
