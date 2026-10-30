#!/usr/bin/env python3
"""Chapter A14: pick tile 0's 48 journey events out of the trace written by ttsim + sim-labs/patches/a14-journey.patch.

Each event is the first line (or first k lines) of the trace that matches one pattern, so the selection does not
depend on clock values. The output lists the events in trace order; the line numbers, clocks and lines are the
"n", "clk" and "raw" fields of the figure in chapters/a14-one-tiles-journey.html, and "rnd" is the issue round of the
nearest preceding instruction line (X/P/M/E/D). The a0/b0/d0 datums change from run to run because eltwise_binary
fills input A from std::random_device; everything else should match the page line for line.

Usage:
  a14_journey.py TRACE          one event per line: <line> <round> <trace line>
  a14_journey.py TRACE --json   the same as a JSON list of {n, clk, rnd, raw}
"""
import json
import re
import sys

# (pattern, how many matches to take). All tile-0 events of the journey, from host write to host read.
PATTERNS = [
    (r"^HOSTW \d+ D\(0,11\):0x2d4c40 ", 1),        # host writes page 0 of A
    (r"^HOSTW \d+ D\(0,11\):0x2d7c40 ", 1),        # host writes page 0 of B
    (r"^X \d+ \d+ T1 \w+ \d+ \d+ SEMWAIT", 1),     # MATH: tile_regs_acquire
    (r"^M \d+ \d+ T1 ", 1),                        # MATH MOP ingested
    (r"^D \d+ \d+ T1 ", 1),                        # MATH MOP expanded
    (r"^CBR \d+ BRISC cb=0 acked=0", 1),           # reader: reserve_back(c_0)
    (r"^NOC \d+ READ ", 2),                        # reader: two NoC reads
    (r"^CBW \d+ BRISC cb=0 received=1", 1),        # reader: push_back(c_0)
    (r"^CBR \d+ TRISC0 cb=0 received=1", 1),       # UNPACK: cb_wait_front(c_0)
    (r"^CBR \d+ TRISC2 cb=16 acked=", 1),          # PACK: cb_reserve_back(c_16), running ahead
    (r"^CBW \d+ BRISC cb=1 received=1", 1),        # reader: push_back(c_1)
    (r"^CBR \d+ TRISC0 cb=1 received=1", 1),       # UNPACK: cb_wait_front(c_1)
    (r"^M \d+ \d+ T0 ", 1),                        # UNPACK MOP ingested
    (r"^UNP ", 8),                                 # 8 x UNPACR
    (r"^ELW ", 8),                                 # 8 x ELWADD
    (r"^X \d+ \d+ T1 37cc0003 ", 4),               # 4 x SETRWC(CLR_AB) executed
    (r"^CBW \d+ TENSIX cb=0 acked=1", 1),          # cb_pop_front(c_0) via STOREREG
    (r"^X \d+ \d+ T1 \w+ \d+ \d+ SEMPOST", 1),     # tile_regs_commit
    (r"^X \d+ \d+ T2 \w+ \d+ \d+ SEMWAIT", 1),     # tile_regs_wait
    (r"^PCK .* datums=256 ", 4),                   # the pack MOP's PACR, 4 packers
    (r"^CBW \d+ TENSIX cb=1 acked=1", 1),          # cb_pop_front(c_1)
    (r"^CBW \d+ TENSIX cb=16 received=1", 1),      # cb_push_back(c_16)
    (r"^X \d+ \d+ T2 \w+ \d+ \d+ SEMGET", 1),      # tile_regs_release
    (r"^CBR \d+ NCRISC cb=16 received=1", 1),      # writer: wait_front(c_16)
    (r"^NOC \d+ WRITE ", 1),                       # writer: NoC write
    (r"^CBW \d+ NCRISC cb=16 acked=1", 1),         # writer: pop_front(c_16)
    (r"^HOSTR \d+ D\(0,11\):0x2dac40 ", 1),        # host reads page 0 of the output
]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    lines = open(sys.argv[1]).read().split("\n")
    picked = set()
    for pat, k in PATTERNS:
        rx = re.compile(pat)
        hits = [i for i, l in enumerate(lines) if rx.search(l)][:k]
        if len(hits) < k:
            sys.exit(f"pattern {pat!r}: found {len(hits)} of {k} lines")
        picked.update(hits)
    events, rnd = [], None
    for i, l in enumerate(lines):
        f = l.split()
        if f and f[0] in ("X", "P", "M", "E", "D"):
            rnd = int(f[2])
        if i in picked:
            events.append({"n": i + 1, "clk": int(f[1]), "rnd": rnd, "raw": l})
    if "--json" in sys.argv:
        print(json.dumps(events, indent=0))
    else:
        for e in events:
            print(f'{e["n"]:6} {e["rnd"]:6}  {e["raw"]}')
        print(f"{len(events)} events, {sum(e['clk'] == 4523 for e in events)} at clock 4523", file=sys.stderr)


if __name__ == "__main__":
    main()
