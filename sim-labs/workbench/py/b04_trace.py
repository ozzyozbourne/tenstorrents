#!/usr/bin/env python3
"""Lab B04: turn the raw trace written by the B04-instrumented ttsim into readable Tensix instruction streams.

The instrumented simulator (sim-labs/patches/b04-*.patch) writes one line per event for one tile. <round> counts
the issue rounds of that tile: ttsim clocks a busy tile several rounds per clock (libttsim.cpp clock_current_chip).
  X <clock> <round> T<pipe> <inst> <failed tries> <stalled clocks>  an instruction left the FIFO and executed (b04-trace)
  P <clock> <round> T<pipe> <inst> <bypass>        a RISC-V pushed an instruction                   (b04-mop)
  M <clock> <round> T<pipe> <inst> <cfg0..cfg8>    the MOP expander ingested a MOP                  (b04-mop)
  E <clock> <round> T<pipe> <inst>                 the MOP expander emitted an instruction          (b04-mop)
  D <clock> <round> T<pipe> <mop> <emitted>        the MOP expansion finished                       (b04-mop)
  R <clock> <round> T<pipe> <inst>                 the replay expander saw a REPLAY instruction     (b04-mop)
  Y <clock> <round> T<pipe> <inst>                 the replay expander played back an instruction   (b04-mop)
Opcode names and field positions come from ttsim's data/<chip>/tensix_isa.json, the file ttsim generates its decoder from.

Usage:
  b04_trace.py stream TRACE [--pipe N] [--kinds XPMEDRY] [--from R] [--to R] [--limit N]
  b04_trace.py stalls TRACE [--min N]
  b04_trace.py mops   TRACE
  b04_trace.py hist   TRACE [TRACE2 ...]        executed-opcode counts per thread, side by side
  b04_trace.py js     TRACE [--from R] [--to R]  compact JSON [round, pipe, name, stalled rounds] of X events
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

SEM_NAMES = ["FPU_SFPU", "MATH_PACK", "UNPACK_TO_DEST", "UNPACK_OPERAND_SYNC",
             "PACK_DONE", "UNPACK_SYNC", "UNPACK_MATH_DONE", "MATH_DONE"]  # ckernel_structs.h:15-24
# STALLWAIT condition bits C0..C14 (tt-isa-documentation STALLWAIT.md "Condition mask"), short names
COND_NAMES = ["THCON", "UNPACK0", "UNPACK1", "PACK0", "PACK1", "PACK2", "PACK3", "MATH",
              "SRCA_UNPACKER_OWNS", "SRCB_UNPACKER_OWNS", "SRCA_FPU_OWNS", "SRCB_FPU_OWNS", "MOVER", "TRISC_CFG", "SFPU"]


def load_isa(path=None):
    if path is None:
        home = os.environ.get("TTSIM_HOME", os.path.expanduser("~/ttsim"))
        path = os.path.join(home, "data", "wh", "tensix_isa.json")
    with open(path) as f:
        isa = json.load(f)
    by_op = {}
    for name, info in isa.items():
        by_op[info["opcode"]] = (name, info.get("args", {}))
    return by_op


def field(inst, spec):
    hi, lo = (int(x) for x in spec.split(":"))
    return (inst >> lo) & ((2 << (hi - lo)) - 1)


def describe(by_op, inst):
    """Opcode name plus a few decoded fields that matter in this lab."""
    op = inst >> 24
    name, args = by_op.get(op, ("op%02x" % op, {}))
    if name in ("MOP", "REPLAY"):  # tensix_isa.json has placeholder field specs for these four; ttsim decodes them by hand
        if name == "MOP":  # tensix.cpp mop_ingest: zlo 15:0, loop_count 22:16, mop_type 23
            return name, "type=%d loop_count=%d zmask_lo=0x%x" % ((inst >> 23) & 1, (inst >> 16) & 127, inst & 0xFFFF)
        # tensix.cpp replay_expander: load_mode 0, execute_while_loading 1, len 13:4, start_idx 23:14
        return name, "%s start=%d len=%d%s" % ("load" if inst & 1 else "play", (inst >> 14) & 0x3FF, (inst >> 4) & 0x3FF,
                                              " +exec" if inst & 2 else "")
    f = {k: field(inst, v) for k, v in args.items()}
    if name in ("SEMINIT", "SEMPOST", "SEMGET", "SEMWAIT"):
        sems = [SEM_NAMES[i] for i in range(8) if (f.get("sem_sel", 0) >> i) & 1]
        extra = ""
        if name == "SEMWAIT":
            extra = {1: " while==0", 2: " while==max"}.get(f["wait_sem_cond"], " cond=%d" % f["wait_sem_cond"])
        if name == "SEMINIT":
            extra = " init=%d max=%d" % (f.get("init_value", 0), f.get("max_value", 0))
        return name, "/".join(sems) + extra
    if name == "STALLWAIT":
        conds = [COND_NAMES[i] for i in range(15) if (f["wait_res"] >> i) & 1]
        return name, "until " + ("+".join(conds) or "-") + " block=0x%03x" % f["stall_res"]
    if name == "UNPACR":
        return name, "Src%s" % ("B" if f["unpack_block_selection"] else "A")
    if name in ("ELWADD", "MVMUL", "ELWMUL", "ELWSUB"):
        return name, "dst=%d addr_mode=%d clear_dvalid=%d" % (f["dst"], f["addr_mode"], f["clear_dvalid"])
    return name, ""


def read(path):
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5 and p[3].startswith("T"):
                yield p[0], int(p[1]), int(p[2]), int(p[3][1:]), p[4:]


def cmd_stream(a, by_op):
    n = 0
    for kind, clk, rnd, pipe, rest in read(a.trace):
        if kind not in a.kinds or (a.pipe is not None and pipe != a.pipe):
            continue
        if rnd < a.frm or rnd > a.to:  # --from/--to select issue rounds
            continue
        if kind == "D":
            print("%-2s %7d %6d T%d   ...MOP done: %s instructions emitted" % (kind, clk, rnd, pipe, rest[1]))
        else:
            inst = int(rest[0], 16)
            name, desc = describe(by_op, inst)
            tail = ""
            if kind == "X" and int(rest[1]):
                tail = "   <- stalled %s rounds, %s clocks" % (rest[1], rest[2])
            if kind == "P" and rest[1] == "1":
                tail = "   (bypasses MOP expander)"
            print("%-2s %7d %6d T%d %08x %-10s %s%s" % (kind, clk, rnd, pipe, inst, name, desc, tail))
        n += 1
        if a.limit and n >= a.limit:
            break


def cmd_stalls(a, by_op):
    tot = defaultdict(lambda: [0, 0, 0])  # (pipe, name) -> [times stalled, stalled rounds, max rounds]
    for kind, clk, rnd, pipe, rest in read(a.trace):
        if kind == "X" and int(rest[1]):
            name, desc = describe(by_op, int(rest[0], 16))
            key = (pipe, name + " " + desc.split(" block")[0])
            c = int(rest[1])
            t = tot[key]
            t[0] += 1; t[1] += c; t[2] = max(t[2], c)
    print("thread  instruction                                   stalled  rounds  max")
    for (pipe, nm), (k, c, m) in sorted(tot.items(), key=lambda kv: (kv[0][0], -kv[1][1])):
        if c >= a.min:
            print("T%d      %-45s %7d %7d %4d" % (pipe, nm[:45], k, c, m))


def cmd_mops(a, by_op):
    cur = {}
    for kind, clk, rnd, pipe, rest in read(a.trace):
        if kind == "M":
            cur[pipe] = Counter()
            name, desc = describe(by_op, int(rest[0], 16))
            c = rest[1:]
            nm = [describe(by_op, int(x, 16))[0] for x in c[2:9]]  # cfg[2..8]
            if (int(rest[0], 16) >> 23) & 1:  # mop_type 1, the double loop (tensix.cpp mop_step_inst)
                print("T%d clock %d round %d: MOP %s  %s  outer=%s inner=%s" % (pipe, clk, rnd, rest[0], desc, c[0], c[1]))
                print("      start=%s end0=%s end1=%s loop=%s loop1=%s last0=%s last1=%s" % tuple(nm))
            else:  # mop_type 0, the unpack template: per iteration, zmask bit set -> cfg[7](+cfg[8]) else cfg[3](+cfg[4..6])(+cfg[2])
                print("T%d clock %d round %d: MOP %s  %s  flags=%s" % (pipe, clk, rnd, rest[0], desc, c[1]))
                print("      A=%s B=%s,%s,%s tail=%s skipA=%s skipB=%s" % (nm[1], nm[2], nm[3], nm[4], nm[0], nm[5], nm[6]))
        elif kind == "E" and pipe in cur:
            cur[pipe][describe(by_op, int(rest[0], 16))[0]] += 1
        elif kind == "D" and pipe in cur:
            print("      -> %s emitted: %s" % (rest[1], ", ".join("%s x%d" % kv for kv in cur[pipe].items())))
            del cur[pipe]


def cmd_hist(a, by_op):
    hs = []
    for t in a.trace:
        h = Counter()
        for kind, clk, rnd, pipe, rest in read(t):
            if kind == "X":
                h[(pipe, describe(by_op, int(rest[0], 16))[0])] += 1
        hs.append(h)
    keys = sorted(set().union(*hs), key=lambda k: (k[0], -sum(h[k] for h in hs), k[1]))
    print("thread opcode        " + "".join("%12s" % os.path.basename(t)[:12] for t in a.trace))
    for k in keys:
        print("T%d     %-14s" % k + "".join("%12d" % h[k] for h in hs))
    print("total                " + "".join("%12d" % sum(h.values()) for h in hs))


def cmd_js(a, by_op):
    ev = []
    for kind, clk, rnd, pipe, rest in read(a.trace):
        if kind == "X" and a.frm <= rnd <= a.to:
            name, desc = describe(by_op, int(rest[0], 16))
            ev.append([rnd, pipe, name, int(rest[1])])
    print(json.dumps(ev, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stream", "stalls", "mops", "hist", "js"])
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--isa", help="path to tensix_isa.json (default $TTSIM_HOME/data/wh/tensix_isa.json)")
    ap.add_argument("--pipe", type=int)
    ap.add_argument("--kinds", default="X")
    ap.add_argument("--from", dest="frm", type=int, default=0, help="first issue round")
    ap.add_argument("--to", type=int, default=1 << 62)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min", type=int, default=1)
    a = ap.parse_args()
    by_op = load_isa(a.isa)
    if a.cmd != "hist":
        a.trace = a.trace[0]
    {"stream": cmd_stream, "stalls": cmd_stalls, "mops": cmd_mops, "hist": cmd_hist, "js": cmd_js}[a.cmd](a, by_op)


if __name__ == "__main__":
    sys.exit(main())
