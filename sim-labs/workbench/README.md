# ttsim workbench

Small scripts for running tt-metal on the ttsim simulator without touching the tt-metal tree.
Setup from scratch is in lab B00 (`../b00-workbench.html`). Tested against tt-metal `7aad8aac`
and ttsim `v1.15.0` (`b3620244`).

## Expected layout

```
~/tt-metal/                      built with build_metal.sh (build_Release/)   -> TT_METAL_HOME
~/ttsim/                         source checkout                             -> TTSIM_HOME
~/sim/wh/libttsim_wh.so          + soc_descriptor.yaml (wormhole_b0_80_arch.yaml)
~/sim/bh/libttsim_bh.so          + soc_descriptor.yaml (blackhole_140_arch.yaml)
```

Export `TT_METAL_HOME`, `TTSIM_HOME`, `LAB_HOME` or `SIM` before sourcing `env.sh` to override any of
these. If `~/tt-metal` or `~/ttsim` is missing, `env.sh` falls back to `~/workspace/refs/...` (the
series' own build machine).

## Scripts

### `source env.sh [wh|bh]`

Sets up one lab session, then changes into `$LAB_HOME/run`.

| Variable | Value |
|---|---|
| `TT_METAL_SIMULATOR` | `$SIM`, or `~/sim/$ARCH/libttsim_$ARCH.so`; warns if no `soc_descriptor.yaml` is beside it |
| `TT_METAL_SLOW_DISPATCH_MODE=1` | ttsim's recommended dispatch mode |
| `TT_METAL_DISABLE_SFPLOADMACRO=1` | SFPLOADMACRO is unsupported in ttsim |
| `TT_METAL_RUNTIME_ROOT` | `$TT_METAL_HOME` (firmware, headers, stock kernels) |
| `TT_METAL_KERNEL_PATH` | `$LAB_HOME/kernels`, the kernel overlay |
| `TT_METAL_LOGS_PATH`, `TT_METAL_PROFILER_DIR` | `$LAB_HOME/logs`, `$LAB_HOME/profiler` |
| `EX` | `$TT_METAL_HOME/build_Release/programming_examples` |

Use a separate `LAB_HOME` per lab (default `~/ttlab`), for example `export LAB_HOME=~/ttlab/b01`.

**Kernel overlay:** copy a kernel to `$LAB_HOME/kernels/<the relative path the host passes>`, edit
it, and rerun. No rebuild is needed. Kernels are searched in the cwd first, then
`TT_METAL_KERNEL_PATH`, then `/usr/share/tenstorrent/kernels`, then the runtime root
(`tt_metal/impl/kernels/kernel.cpp:48-95`). So run from `$LAB_HOME/run`, never from
`$TT_METAL_HOME`, or the stock kernel silently wins.

```bash
W=/path/to/tenstorrent-from-zero/sim-labs/workbench   # env.sh cds away, so use absolute paths
export LAB_HOME=~/ttlab/demo
source $W/env.sh wh
$EX/metal_example_add_2_integers_in_riscv     # Success: Result is 21, [4501] clocks
```

### `build_host.sh file.cpp`

Compiles and links one host program out of tree against `$TT_METAL_HOME/build_Release`, using the
in-tree examples' flags (about 6 s). The object and binary are written next to `file.cpp`. The flags
contain pinned `.cpmcache/` hashes and library versions, so regenerate them from
`build_Release/build.ninja` if you move the tt-metal pin. Needs `clang++-20`.

```bash
cp $W/host/mm_fid.cpp $LAB_HOME/ && $W/build_host.sh $LAB_HOME/mm_fid.cpp
MM_FIDELITY=0 MM_SEED=42 $LAB_HOME/mm_fid    # result_hash = 213e53fed10c0383
```

### `build_ttsim.sh <name> [wh|bh] [release|debug]`

On first use, copies `$TTSIM_HOME` to `$LAB_HOME/ttsim-<name>/` (edit `src/` there). Then it builds
that flavour (about 3.5 s), packages `libttsim.so` with the matching `soc_descriptor.yaml` in
`$LAB_HOME/sim-<name>-<arch>-<flavor>/`, and prints the `export TT_METAL_SIMULATOR=...` line to use.

## Examples and helpers

- `host/mm_fid.cpp`: single-core matmul with `MM_FIDELITY=0|2|3|4` (numeric: LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4; names silently mean LoFi) and `MM_SEED`;
  prints `mean_abs_err` and `result_hash`.
- `host/run_twice.cpp` (chapter A12): the stock `add_2_integers_in_compute` host program with the enqueue run twice on the same core;
  prints `run 1 finished`, `run 2 finished`, then the usual result check. Pair it with a kernel overlay.
- `host/dbrw.cpp`: two-device MeshBuffer write/read on a `MeshShape(2,1)`. Needs the `wh_x2` simulator and `TT_METAL_MOCK_CLUSTER_DESC_PATH=.../wormhole_N300.yaml`.
- `harness/h.c`: calls `libttsim.so` directly with no tt-metal:
  `gcc -O2 -o h harness/h.c -ldl && ./h ~/sim/wh/libttsim_wh.so` prints `cfg[0] = 0x401e1e52`.
- `py/bfp8_probe.py`, `py/bfp4_probe.py`, `py/bfp8_double.py` (chapter A05): quantize one BFP block
  on the host and on the simulated device and print both. Activate `$TT_METAL_HOME/python_env`,
  source `env.sh`, then run `python $W/py/bfp8_probe.py`.
- `py/a14_journey.py` (chapter A14): picks tile 0's 48 journey events out of a trace written by ttsim +
  `patches/a14-journey.patch`: `python3 $W/py/a14_journey.py $LAB_HOME/journey.trace [--json]`.
- `py/a18_*.py` (chapter A18): TT-NN from the user's side. `a18_pad` and `a18_devtilize` (padding and pages,
  host vs device tilize), `a18_ops` (add/matmul/softmax PCC), `a18_ladder` (dtype x fidelity), `a18_shard` and
  `a18_shard_edge` (interleaved, height/width/block sharding and two invalid specs), `a18_progcfg` (the silent LoFi
  default with a program config, plus a torch MLP port), `a18_fid` (fidelity vs Dst format; aborts by design at
  `fp32_dest_acc_en=True` under ttsim v1.15.0), `a18_addpath` (program cache and JIT; set an empty `TT_METAL_CACHE`).
  Activate `$TT_METAL_HOME/python_env`, source `env.sh`, export `TT_METAL_INSPECTOR=0`, then run `python $W/py/a18_ops.py`.
- `py/a20_count.py`, `py/a20_summary.py` (chapter A20): run one ttnn op (format, fidelity, block size, fusion,
  sharding cases) under a ttsim built with `../patches/a20-counters.patch`, then summarise its exact DRAM/NoC byte
  counts and Tensix instruction counts. Counts only; the simulator's clocks are not performance.
- `py/b08_formats.py` (lab B08): round-trips one tile through float32, bfloat16, bfloat8_b and bfloat4_b on the host
  and on the device. `B08_SRC=bf16` starts from a BF16 tile (needed on Wormhole, where FP32 typecast hits ttsim UB).
- `host/b08_sfpu.cpp` (lab B08): one SFPU function over evenly spaced inputs vs a double reference, chosen with
  `B08_FN=copy|map|piecewise|sin|exp|recip` and `B08_FMT`, `B08_DST32`, `B08_U2D`, `B08_APPROX`, `B08_EXPT`, `B08_FOLD`.
  Needs the overlay: `cp -r ../overlays/b08/. $LAB_HOME/kernels/`.
- `py/b04_trace.py` (lab B04): names and summarises the per-tile Tensix instruction trace written by a ttsim
  built with `../patches/b04-trace.patch` and `b04-mop.patch` (`stream`, `stalls`, `mops`, `hist`, `js`).
- `py/b07_mvmul.py` (lab B07): reproduces every Dst element of an MVMUL dump bit for bit (independent and chained
  checks, `--isa`, `--trace M ROW COL`, `--json OUT`). The dump comes from a ttsim built with
  `../patches/b07-mvmul-dump.patch` and run with `B07_OUT=file` (`B07_TILE`, `B07_SKIP`, `B07_COUNT`).
- `py/a19_cache_demo.py`, `py/a19_hit_vs_miss.py`, `py/a19_sqdiff_generic.py` (chapter A19): the TT-NN
  program cache growing per call, cache on vs off, and a custom `(a-b)^2` op run through `ttnn.generic_op`.
  Set `TT_METAL_CACHE=$LAB_HOME/jit` first so you can list the JIT-compiled kernels afterwards.

## Common errors

- `YAML::BadFile ... soc_descriptor.yaml`: there is no descriptor next to the `.so`.
- `TT_FATAL: Root Directory is not set.`: `TT_METAL_RUNTIME_ROOT` is unset (source `env.sh`).
- An edit has no effect: the cwd contains a `tt_metal/` tree; `cd $LAB_HOME/run`.
- The JIT cache is `~/.cache/tt-metal-cache/`. Kernel hash directories are keyed on the kernel path, args and config
  (not file contents); edits are detected through `.dephash` sidecar files, so it rarely needs clearing.
