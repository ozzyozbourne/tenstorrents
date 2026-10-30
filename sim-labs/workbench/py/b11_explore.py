"""B11 op explorer: run one TT-NN op over a grid of shapes, dtypes, layouts and memory configs under ttsim,
compare every result with torch (PCC and max |error|), and record which kernels each case JIT-compiled.

    python b11_explore.py --op add --shapes 64x128 --dtypes bf16,bfp8,fp32 --layouts tile,rm \
                          --mems dram,l1,height --jobs 3 --out $LAB_HOME/explore_add.json

Each case runs in its own process because ttsim ends the whole process on an unsupported instruction.
Each process gets a private TT_METAL_CACHE that starts with only the precompiled header and firmware
(hard links into a seed cache), so the kernels/ directory afterwards lists exactly the kernels that case
needed. Ops: add, relu, exp, matmul (x @ w with w a DRAM-interleaved TILE tensor of shape [W, W]).
"""
import argparse, concurrent.futures, glob, itertools, json, os, re, shutil, subprocess, sys, time

HERE = os.path.abspath(__file__)


def child(case):
    import torch, ttnn
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}[case["dtype"]]
    lay = {"tile": ttnn.TILE_LAYOUT, "rm": ttnn.ROW_MAJOR_LAYOUT}[case["layout"]]
    h, w = case["shape"]
    torch.manual_seed(0)
    x = torch.randn(h, w)
    y = torch.randn(h, w)
    dev = ttnn.open_device(device_id=0)
    if case["mem"] == "dram":
        mc = ttnn.DRAM_MEMORY_CONFIG
    elif case["mem"] == "l1":
        mc = ttnn.L1_MEMORY_CONFIG
    else:                                           # height sharded: one 32-row slab (or more) per core
        cores = max(1, min(8, h // 32))
        mc = ttnn.create_sharded_memory_config((h, w), core_grid=ttnn.CoreGrid(y=cores, x=1),
                                               strategy=ttnn.ShardStrategy.HEIGHT,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR)
    tx = ttnn.from_torch(x, dtype=dt, layout=lay, device=dev, memory_config=mc)
    if case["op"] == "add":
        ty = ttnn.from_torch(y, dtype=dt, layout=lay, device=dev, memory_config=mc)
        out, ref = ttnn.add(tx, ty, memory_config=mc), x + y
    elif case["op"] == "relu":
        out, ref = ttnn.relu(tx, memory_config=mc), torch.relu(x)
    elif case["op"] == "exp":
        out, ref = ttnn.exp(tx, memory_config=mc), torch.exp(x)
    elif case["op"] == "matmul":
        wt = torch.randn(w, w) / w ** 0.5
        tw = ttnn.from_torch(wt, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        out, ref = ttnn.matmul(tx, tw, memory_config=mc), x @ wt
    else:
        raise SystemExit("unknown op " + case["op"])
    got = ttnn.to_torch(out).float().reshape(ref.shape)
    a, b = got.flatten().double(), ref.flatten().double()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    res = {"pcc": round(pcc, 6), "max_err": float((got - ref).abs().max()),
           "out_layout": str(out.layout).split(".")[-1], "out_mem": str(out.memory_config().memory_layout).split(".")[-1]}
    ttnn.close_device(dev)
    print("B11RESULT " + json.dumps(res), flush=True)


def seed_cache(root, env):
    """Warm a cache once (firmware + precompiled headers), to hard-link into every case's cache."""
    seed = os.path.join(root, "seed")
    if not glob.glob(os.path.join(seed, "tt-metal-cache*", "firmware")):
        case = {"op": "relu", "shape": [32, 32], "dtype": "bf16", "layout": "tile", "mem": "dram"}
        subprocess.run([sys.executable, HERE, "--child", json.dumps(case)], env=dict(env, TT_METAL_CACHE=seed),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return seed


def run_case(case, root, seed, env):
    name = "%s_%dx%d_%s_%s_%s" % (case["op"], *case["shape"], case["dtype"], case["layout"], case["mem"])
    cache = os.path.join(root, name)
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache)
    for d in glob.glob(os.path.join(seed, "tt-metal-cache*")):  # keep pch/ and firmware/, drop kernels/
        for sub in ("pch", "firmware"):
            if os.path.isdir(os.path.join(d, sub)):
                os.makedirs(os.path.join(cache, os.path.basename(d)), exist_ok=True)
                subprocess.run(["cp", "-al", os.path.join(d, sub), os.path.join(cache, os.path.basename(d), sub)],
                               check=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, HERE, "--child", json.dumps(case)], env=dict(env, TT_METAL_CACHE=cache),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    row = dict(case, wall=round(time.time() - t0, 1))
    m = re.search(r"^B11RESULT (.*)$", p.stdout, re.M)
    sim = re.findall(r"^\[\d+\] ERROR: (.*)$", p.stdout, re.M)
    if m:
        row.update(json.loads(m.group(1)), status="ok")
    elif sim:
        row.update(status="abort", why=sim[-1])
    else:
        err = re.findall(r"^(?:RuntimeError|TypeError|ValueError): (.*)$", p.stdout, re.M)
        fatal = re.findall(r"(?:TT_FATAL|TT_THROW)[^:]*: (.*)", p.stdout)
        row.update(status="rejected", why=(fatal or err or ["exit %d" % p.returncode])[0][:160])
    row["kernels"] = sorted({os.path.basename(k) for k in glob.glob(os.path.join(cache, "tt-metal-cache*", "kernels", "*"))})
    shutil.rmtree(cache, ignore_errors=True)    # the kernel list is what we keep
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child")
    ap.add_argument("--op", default="add")
    ap.add_argument("--shapes", default="64x128")
    ap.add_argument("--dtypes", default="bf16,bfp8,fp32")
    ap.add_argument("--layouts", default="tile,rm")
    ap.add_argument("--mems", default="dram,l1,height")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child:
        return child(json.loads(a.child))
    root = os.path.join(os.environ.get("LAB_HOME", "."), "explore-cache")
    env = dict(os.environ, TT_METAL_INSPECTOR="0")
    seed = seed_cache(root, env)
    cases = [{"op": a.op, "shape": [int(v) for v in s.split("x")], "dtype": d, "layout": l, "mem": m}
             for s, d, l, m in itertools.product(a.shapes.split(","), a.dtypes.split(","),
                                                 a.layouts.split(","), a.mems.split(","))]
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(lambda c: run_case(c, root, seed, env), cases))
    print("%-10s %-5s %-4s %-6s %-8s %-9s %-8s  %s" % ("shape", "dtype", "lay", "mem", "status", "PCC", "max|err|", "kernels / reason"))
    for r in rows:
        shape = "%dx%d" % tuple(r["shape"])
        if r["status"] == "ok":
            print("%-10s %-5s %-4s %-6s %-8s %-9.6f %-8.4f  %s" % (shape, r["dtype"], r["layout"], r["mem"], "ok",
                                                                 r["pcc"], r["max_err"], " ".join(r["kernels"])))
        else:
            print("%-10s %-5s %-4s %-6s %-8s %-18s  %s" % (shape, r["dtype"], r["layout"], r["mem"], r["status"], "",
                                                          r["why"]))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
