#!/usr/bin/env python3
"""Lab B07: reproduce ttsim's MVMUL results bit for bit from a dump.

The dump comes from a ttsim built with sim-labs/patches/b07-mvmul-dump.patch and run with
B07_OUT=<file> (see the lab page). Each record holds one MVMUL on one Tensix tile:

    M <index> <clock> <phase> <srca_fmt> <srcb_fmt> <dst32b> <exp8b> <srca_row> <srcb_row> <dst_row>
    A ... 16 lines: SrcA[srca_row + i][0..15], raw 32-bit cells (the low 13 bits are always zero)
    B ...  8 lines: SrcB[srcb_row + r][0..15]
    D ...  8 lines: Dst rows dst_row..dst_row+7 before the MVMUL, in Dst's storage encoding
    R ...  8 lines: the same Dst rows after the MVMUL
    T <total MVMULs seen on the tile>   (last line, written when the process exits)

Usage:
    python3 b07_mvmul.py DUMP                 # check every MVMUL, independently and chained
    python3 b07_mvmul.py DUMP --isa           # also score the ISA spec's approximate float model
    python3 b07_mvmul.py DUMP --trace M R C   # walk through one Dst element of MVMUL #M bit by bit
    python3 b07_mvmul.py DUMP --json OUT      # tile-level data for the lab page's heatmap

Only the BF16-style path is modelled (SrcA format BF16/FP32/BFP*, Dst BF16 or FP32), on Wormhole.
"""
import json
import struct
import sys

# ---------------------------------------------------------------- dump parsing

def parse(path):
    recs, total, cur = [], None, None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == 'M':
                v = [int(x) for x in t[1:]]
                cur = dict(idx=v[0], clock=v[1], phase=v[2], fa=v[3], fb=v[4], d32=bool(v[5]), e8=bool(v[6]),
                           ar=v[7], br=v[8], dr=v[9], A=[], B=[], D=[], R=[])
                recs.append(cur)
            elif t[0] in 'ABDR':
                cur[t[0]].append([int(x, 16) for x in t[1:]])
            elif t[0] == 'T':
                total = int(t[1])
    return recs, total

# ---------------------------------------------------------------- formats

def src_truncate(v, fmt):
    """Wormhole drops Src bits according to the operand's own format (ISA TruncateSrc)."""
    if fmt in (1, 2, 10, 14):      # fp16, bfp8a, fp8, int8: high 3 exponent bits
        return v & 0x8FFFE000
    if fmt in (3, 11):             # bfp4a, bfp2a
        return v & 0x8FFF0000
    if fmt != 4:                   # anything but tf32: low 3 mantissa bits
        return v & 0xFFFF0000
    return v

def dst_decode_bf16(x):            # Dst stores BF16 as sign | mantissa(7) << 8 | exponent(8)
    return (x & 0x8000) | ((x & 255) << 7) | ((x >> 8) & 127)

def dst_encode_bf16(x):
    return (x & 0x8000) | ((x & 127) << 8) | ((x >> 7) & 255)

def dst_read(raw, d32):            # -> natural FP32 bits
    if d32:
        return (dst_decode_bf16(raw >> 16) << 16) | (raw & 0xFFFF)
    return dst_decode_bf16(raw) << 16

def dst_write(bits, d32):          # natural FP32 bits -> Dst storage
    if d32:
        return (dst_encode_bf16(bits >> 16) << 16) | (bits & 0xFFFF)
    return dst_encode_bf16(bits >> 16)

def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]

def bits32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

# ---------------------------------------------------------------- the bit-exact model

def phase_pieces(a, b, phase):
    """Split the 11-bit significands (implicit 1 + 10 bits) into the slice each phase multiplies.
    Returns (man_a 5 bits, man_b 7 bits, exponent adjustment for the slices' weights)."""
    man_a = (a & 0x3FF) | 0x400
    man_b = (b & 0x3FF) | 0x400
    adj = 0
    if phase & 1:
        man_a = (man_a >> 1) & 31   # SrcA bits 5..1: the next five bits after the top five
        adj -= 5
    else:
        man_a >>= 6                 # SrcA bits 10..6: implicit 1 + top four mantissa bits
    if phase & 2:
        man_b = (man_b & 15) << 3   # SrcB bits 3..0: the low four bits, placed at the top of 7
        adj -= 7
    else:
        man_b >>= 4                 # SrcB bits 10..4: implicit 1 + top six mantissa bits
    return man_a, man_b, adj

def products(avals, bvals, phase, fp32_acc, e8):
    """The 16 partial products of one dot product: (sign, exponent, integer mantissa)."""
    base = -127 if e8 else ((127 - 30) if fp32_acc else -15)
    out = []
    for a, b in zip(avals, bvals):
        if (a & 0x3FFFF) < 0x400 or (b & 0x3FFFF) < 0x400:   # exponent 0: zero (denormals flushed)
            out.append((0, max(base - 5 * (phase & 1) - 7 * ((phase >> 1) & 1), 0), 0))
            continue
        ma, mb, adj = phase_pieces(a, b, phase)
        out.append(((a >> 18) ^ (b >> 18), ((a >> 10) & 255) + ((b >> 10) & 255) + base + adj, ma * mb))
    return out

def group_sums(prods):
    """Two groups of 8: align each product to the group's largest exponent, rounding to nearest
    (ties away from zero, on magnitudes), and add as integers."""
    res = []
    for g in range(2):
        grp = prods[8 * g: 8 * g + 8]
        e = max(p[1] for p in grp)
        if e <= 0:
            res.append((0, 0, 0))
            continue
        s = 0
        for sign, exp, man in grp:
            d = min(e - exp, 30)
            m = ((man << 1) + (1 << d)) >> (d + 1)
            s += -m if sign else m
        res.append((int(s < 0), e, abs(s) << 13))   # << 13: the 2.10 product now sits at 2.23, like Dst's 1.23
    return res

def align_term(man, sign, exp_diff, lsb):
    shift = exp_diff + lsb
    if shift == 0:
        return man
    if shift >= 31:
        return 0
    return ((man + (1 << (shift - 1)) - sign) >> shift) << lsb

def accumulate(sops, dst_bits, fp32_acc, e8):
    """Add the two group sums to the Dst value and round to Dst's format (ttsim's
    fpu_accum_normalize_encode). Returns natural FP32 bits."""
    sign_d, exp_d = dst_bits >> 31, (dst_bits >> 23) & 255
    man_d = ((dst_bits & 0x7FFFFF) | 0x800000) if exp_d else 0
    (s0, e0, m0), (s1, e1, m1) = sops
    exp = max(e0, e1, exp_d)
    if exp <= 0:
        return 0
    lsb = 0 if fp32_acc else 13    # a BF16/FP16 Dst rounds each term to 11 bits while aligning
    m0 = align_term(m0, s0, exp - e0, lsb)
    m1 = align_term(m1, s1, exp - e1, lsb)
    man_d = align_term(man_d, 0, exp - exp_d, lsb)
    sign, man = sign_d, man_d
    man += m0 if s0 == sign else -m0
    man += m1 if s1 == sign else -m1
    if man < 0:
        man, sign = -man, sign ^ 1
    if man == 0:
        return 0
    shift = (32 - man.bit_length()) - 8          # clz32 - 8: shift that normalises to 24 bits
    if sign and man == 1:                        # Wormhole renormalisation quirk for -1 (TT_ARCH_VERSION 0)
        shift -= 27
    exp -= shift
    if not fp32_acc:
        shift -= 16 if e8 else 13                # keep 8 (BF16) or 11 (FP16) significant bits
    if shift < 0:
        shift = -shift
        man = (man + (1 << (shift - 1))) >> shift  # round half up
    else:
        man <<= shift
    if not fp32_acc:
        man <<= 16 if e8 else 13
    if man & 0x1000000:
        exp += 1
    man &= 0x7FFFFF
    if exp <= 0:
        return 0
    if fp32_acc or e8:
        return (sign << 31) | (255 << 23) if exp >= 255 else (sign << 31) | (exp << 23) | man
    return (sign << 31) | (31 << 23) | (0x3FF << 13) if exp >= 32 else (sign << 31) | (exp << 23) | man

def operands(rec, r, c):
    """SrcA column c and SrcB row r, truncated and in the 19-bit view (sign 18, exp 17..10, man 9..0)."""
    a = [src_truncate(rec['A'][i][c], rec['fa']) >> 13 for i in range(16)]
    b = [src_truncate(rec['B'][r][i], rec['fb']) >> 13 for i in range(16)]
    return a, b

def mvmul_element(rec, r, c, dst_raw):
    if not rec['e8'] and not rec['d32']:
        raise NotImplementedError('FP16 Dst path is not modelled')
    a, b = operands(rec, r, c)
    sops = group_sums(products(a, b, rec['phase'], rec['d32'], rec['e8']))
    return dst_write(accumulate(sops, dst_read(dst_raw, rec['d32']), rec['d32'], rec['e8']), rec['d32'])

# ---------------------------------------------------------------- the ISA's float "rough guide"

def rf(x):                         # round a double to the nearest float32
    return struct.unpack('<f', struct.pack('<f', x))[0]

def isa_element(rec, r, c, dst_raw):
    """MVMUL.md's functional model taken literally: float32 products of the fidelity slices,
    summed in order, added to Dst, narrowed to BF16 by truncation (WriteDstBF16)."""
    a, b = operands(rec, r, c)
    ph = rec['phase']
    x = 0.0
    for ai, bi in zip(a, b):
        fa, fb = f32(ai << 13), f32(bi << 13)
        if (ph & 1) == 0:
            sa = f32(bits32(fa) & 0xfff80000)
        else:
            sa = rf(fa - f32(bits32(fa) & 0xfff83fff))
        if (ph & 2) == 0:
            sb = f32(bits32(fb) & 0xfffe0000)
        else:
            sb = rf(fb - f32(bits32(fb) & 0xfffe1fff))
        x = rf(x + rf(sb * sa))
    x = rf(x + f32(dst_read(dst_raw, rec['d32'])))
    return dst_write(bits32(x) if rec['d32'] else bits32(x) & 0xFFFF0000, rec['d32'])

# ---------------------------------------------------------------- reports

def check(recs, model):
    """Every MVMUL from its own dumped inputs: compare all 8x16 results."""
    per_phase = {}
    first_bad = None
    for rec in recs:
        ok = tot = 0
        for r in range(8):
            for c in range(16):
                got = model(rec, r, c, rec['D'][r][c])
                tot += 1
                if got == rec['R'][r][c]:
                    ok += 1
                elif first_bad is None:
                    first_bad = (rec['idx'], r, c, got, rec['R'][r][c])
        p = per_phase.setdefault(rec['phase'], [0, 0, 0])
        p[0] += 1; p[1] += ok; p[2] += tot
    return per_phase, first_bad

def check_chained(recs):
    """Start from the first Dst value each row shows, then apply only our own model: the dumped
    D rows of later MVMULs are never used. Compare every dumped R row with our running Dst."""
    dst, ok, tot, hidden = {}, 0, 0, 0
    for rec in recs:
        for r in range(8):
            row = rec['dr'] + r
            if row not in dst:
                dst[row] = list(rec['D'][r])
            elif dst[row] != rec['D'][r]:
                hidden += 1          # something else wrote this Dst row in between
            new = [mvmul_element(rec, r, c, dst[row][c]) for c in range(16)]
            ok += sum(1 for c in range(16) if new[c] == rec['R'][r][c])
            tot += 16
            dst[row] = new
    return ok, tot, hidden

def phase_effect(recs):
    """How many Dst elements each phase actually changed."""
    eff = {}
    for rec in recs:
        e = eff.setdefault(rec['phase'], [0, 0])
        for r in range(8):
            for c in range(16):
                e[0] += rec['R'][r][c] != rec['D'][r][c]
                e[1] += 1
    return eff

def tile_layout(rec_list, phases):
    """Rebuild in0 (from SrcB), in1 (from SrcA) and C (from Dst) as 32x32 matrices. Faces are 16x16
    blocks in the order f0 f1 / f2 f3; rows 16f..16f+15 of SrcA, SrcB and Dst hold face f."""
    def put(m, row64, col16, v):
        f = row64 // 16
        m[(f // 2) * 16 + row64 % 16][(f % 2) * 16 + col16] = v
    in0 = [[0.0] * 32 for _ in range(32)]
    in1 = [[0.0] * 32 for _ in range(32)]
    for rec in rec_list:
        if rec['phase'] != 0:
            continue
        for i in range(16):
            for c in range(16):
                put(in1, rec['ar'] + i, c, f32(src_truncate(rec['A'][i][c], rec['fa'])))
        for r in range(8):
            for i in range(16):
                put(in0, rec['br'] + r, i, f32(src_truncate(rec['B'][r][i], rec['fb'])))
    return in0, in1, put

def matmul(x, y):
    return [[sum(x[i][k] * y[k][j] for k in range(32)) for j in range(32)] for i in range(32)]

def tile_products(recs):
    """Split the dump into tile products: each is 16 MVMULs x number of phases."""
    nph = len({r['phase'] for r in recs})
    n = 16 * nph
    return [recs[i:i + n] for i in range(0, len(recs) - len(recs) % n, n)], nph

def dst_tile(chunk, put):
    c = [[0.0] * 32 for _ in range(32)]
    raw = [[0] * 32 for _ in range(32)]
    for rec in chunk:                                # later MVMULs overwrite earlier ones
        for r in range(8):
            for col in range(16):
                put(c, rec['dr'] + r, col, f32(dst_read(rec['R'][r][col], rec['d32'])))
                put(raw, rec['dr'] + r, col, dst_read(rec['R'][r][col], rec['d32']) >> 16)
    return c, raw

def trace(recs, m, r, c):
    rec = next(x for x in recs if x['idx'] == m)
    a, b = operands(rec, r, c)
    ph = rec['phase']
    print(f"MVMUL #{m}: phase {ph}, SrcA rows {rec['ar']}..{rec['ar'] + 15} col {c}, SrcB row {rec['br'] + r}, Dst row {rec['dr'] + r} col {c}")
    prods = products(a, b, ph, rec['d32'], rec['e8'])
    print(" k  SrcB (=in0)          SrcA (=in1)          slice_b  slice_a  product  exp")
    for k in range(16):
        ma, mb, adj = phase_pieces(a[k], b[k], ph)
        print(f"{k:2d}  {b[k]:05x} {f32(b[k] << 13):<12.8g} {a[k]:05x} {f32(a[k] << 13):<12.8g} "
              f"{mb:3d}={mb:07b} {ma:2d}={ma:05b} {prods[k][2]:5d}  {prods[k][1]}")
    sops = group_sums(prods)
    for g, (s, e, mn) in enumerate(sops):
        print(f"group {g}: sign {s} exp {e} sum {mn >> 13} (x 2^{e - 127 - 10} = {(-1) ** s * (mn >> 13) * 2.0 ** (e - 137):.9g})")
    d_in = dst_read(rec['D'][r][c], rec['d32'])
    out = accumulate(sops, d_in, rec['d32'], rec['e8'])
    exact = sum(f32(bb << 13) * f32(aa << 13) for aa, bb in zip(a, b))
    print(f"Dst before {f32(d_in):.9g} (raw {rec['D'][r][c]:04x})  ->  model {f32(out):.9g} "
          f"(raw {dst_write(out, rec['d32']):04x}), ttsim raw {rec['R'][r][c]:04x}")
    print(f"exact (all bits) dot product of this slab: {exact:.9g}")

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    recs, total = parse(args[0])
    if '--trace' in args:
        i = args.index('--trace')
        trace(recs, *(int(x) for x in args[i + 1:i + 4])); return
    phases = sorted({r['phase'] for r in recs})
    r0 = recs[0]
    print(f"{args[0]}: {len(recs)} MVMULs dumped (tile total {total}), phases used {phases}, "
          f"SrcA fmt {r0['fa']} SrcB fmt {r0['fb']} Dst {'FP32' if r0['d32'] else 'BF16'}")
    per_phase, bad = check(recs, mvmul_element)
    ok = sum(v[1] for v in per_phase.values()); tot = sum(v[2] for v in per_phase.values())
    for ph in sorted(per_phase):
        n, o, t = per_phase[ph]
        print(f"  phase {ph}: {n:4d} MVMULs, {o}/{t} elements match")
    print(f"independent: {ok}/{tot} elements match" + (f"  first mismatch {bad}" if bad else ""))
    cok, ctot, hidden = check_chained(recs)
    print(f"chained:     {cok}/{ctot} elements match ({hidden} Dst rows changed by something other than these MVMULs)")
    eff = phase_effect(recs)
    print("elements changed by each phase: " + ", ".join(f"phase {p}: {e[0]}/{e[1]}" for p, e in sorted(eff.items())))
    if '--isa' in args:
        ip, _ = check(recs, isa_element)
        iok = sum(v[1] for v in ip.values())
        print(f"ISA float model (MVMUL.md, taken literally): {iok}/{tot} elements match ({100.0 * iok / tot:.1f}%)")
    if '--json' in args:
        out = args[args.index('--json') + 1]
        chunks, nph = tile_products(recs)
        in0, in1, put = tile_layout(chunks[0], nph)
        ref1 = matmul(in0, in1)
        c1, raw1 = dst_tile(chunks[0], put)
        refK = [[0.0] * 32 for _ in range(32)]
        for ch in chunks:
            x, y, _ = tile_layout(ch, nph)
            p = matmul(x, y)
            refK = [[refK[i][j] + p[i][j] for j in range(32)] for i in range(32)]
        cK, rawK = dst_tile(chunks[-1], put)
        json.dump(dict(phases=nph, k_tiles=len(chunks), in0=in0, in1=in1, ref1=ref1, dst1=raw1, refK=refK, dstK=rawK),
                  open(out, 'w'))
        e1 = sum(abs(c1[i][j] - ref1[i][j]) for i in range(32) for j in range(32)) / 1024
        eK = sum(abs(cK[i][j] - refK[i][j]) for i in range(32) for j in range(32)) / 1024
        print(f"tile (0,0): after 1 tile product mean |Dst - exact| = {e1:.6f}; after {len(chunks)} = {eK:.6f}; wrote {out}")

if __name__ == '__main__':
    main()
