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
MM_FIDELITY=LoFi MM_SEED=42 $LAB_HOME/mm_fid    # result_hash = 213e53fed10c0383
```

### `build_ttsim.sh <name> [wh|bh] [release|debug]`

On first use, copies `$TTSIM_HOME` to `$LAB_HOME/ttsim-<name>/` (edit `src/` there). Then it builds
that flavour (about 3.5 s), packages `libttsim.so` with the matching `soc_descriptor.yaml` in
`$LAB_HOME/sim-<name>-<arch>-<flavor>/`, and prints the `export TT_METAL_SIMULATOR=...` line to use.

## Examples and helpers

- `host/mm_fid.cpp`: single-core matmul with `MM_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4` and `MM_SEED`;
  prints `mean_abs_err` and `result_hash`.
- `host/dbrw.cpp`: two-device MeshBuffer write/read on a `MeshShape(2,1)`. Needs the `wh_x2` simulator and `TT_METAL_MOCK_CLUSTER_DESC_PATH=.../wormhole_N300.yaml`.
- `harness/h.c`: calls `libttsim.so` directly with no tt-metal:
  `gcc -O2 -o h harness/h.c -ldl && ./h ~/sim/wh/libttsim_wh.so` prints `cfg[0] = 0x401e1e52`.

## Common errors

- `YAML::BadFile ... soc_descriptor.yaml`: there is no descriptor next to the `.so`.
- `TT_FATAL: Root Directory is not set.`: `TT_METAL_RUNTIME_ROOT` is unset (source `env.sh`).
- An edit has no effect: the cwd contains a `tt_metal/` tree; `cd $LAB_HOME/run`.
- The JIT cache is `~/.cache/tt-metal-cache/`. Edited kernels get new hash directories, so it rarely
  needs clearing.
