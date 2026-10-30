# Feasibility findings: experiments actually run in the orb on 2026-10-30

## Workbench
`sim-labs/workbench/` has been verified end to end.

- **`env.sh [wh|bh]`** sets up a run:
  - `TT_METAL_SIMULATOR`, `SLOW_DISPATCH`, `DISABLE_SFPLOADMACRO`
  - `TT_METAL_RUNTIME_ROOT`, `TT_METAL_KERNEL_PATH=$LAB_HOME/kernels`, `LOGS_PATH`, `PROFILER_DIR`
  - It then changes into `$LAB_HOME/run`.
  - Set `LAB_HOME` (default `~/ttlab`) to a unique directory per lab when several labs run at once.
- **`build_host.sh file.cpp`** builds a host program out of tree against `build_Release` in about 6 s. Example: `host/mm_fid.cpp` is a single-core matmul that takes `MM_FIDELITY=LoFi|HiFi2|HiFi3|HiFi4` and `MM_SEED`, and prints `mean_abs_err` and `result_hash`.
- **`build_ttsim.sh <name> [wh|bh] [release|debug]`** copies ttsim into `$LAB_HOME/ttsim-<name>`, builds it (about 3.5 s), packages the `.so` next to `soc_descriptor.yaml`, and prints the `export TT_METAL_SIMULATOR=...` line.
- **Kernel path search order** (`tt_metal/impl/kernels/kernel.cpp:50-95`): absolute path, then cwd, then `TT_METAL_KERNEL_PATH`, then the system kernel directory, then the runtime root. Running from `TT_METAL_HOME` therefore silently uses the stock kernels.
- **Editing a kernel and rerunning needs no rebuild.** The JIT cache keys on the source content.

## Verified results
| Test | Result |
|---|---|
| add_2_integers_in_riscv | "Success: Result is 21", `[4501]` clocks, deterministic across runs |
| Same test with the kernel's `+` changed to `*` | "Expected result of 21, got 98" |
| matmul_single_core, HiFi4, seed 42 | 330591 clocks, hash `ef7abcd145720383` |
| matmul_single_core, LoFi, seed 42 | 699743 clocks, `mean_abs_err` 2.216, PCC 0.98085, hash `213e53fed10c0383`, repeatable |

Unseeded errors by fidelity: HiFi2 1.511, HiFi3 1.505, HiFi4 1.505 (mean_abs_err).

- **Clock counts are not performance.** LoFi takes more clocks than HiFi4 in the simulator, the opposite of silicon. Clocks are driven by the host (one per UMD read). With DPRINT enabled, add_2 rises from 4501 to 6611 clocks. Fast dispatch gives 10946 clocks.
- **DPRINT works:** `TT_METAL_DPRINT_CORES=0,0` prints e.g. "0:0-0:BR: My logical coordinates are 0,0", plus lines from TR0, TR1 and TR2.
- **Watcher works** with `TT_METAL_WATCHER=1` and writes `$TT_METAL_LOGS_PATH/generated/watcher/watcher.log`. It makes the run about 40x slower (0.4 s to 16 s).
- **Misaligned NoC read** (`noc_async_read` to `src0_l1+4`):
  - Without Watcher, ttsim reports `[4343] ERROR: UndefinedBehavior: noc_cmd_ctrl: read: alignment of src_addr=0x2d4c40 and dst_addr=0x16dfe4 does not match`.
  - With Watcher, tt-metal catches it first: "Watcher detected NOC error ... (invalid address alignment in NOC transaction)".
- **Instrumented ttsim:** a `ttsim_printf` in `noc_cmd_ctrl` printed 100 NoC commands for loopback.
- **Device profiler:** `TT_METAL_DEVICE_PROFILER=1` produces a CSV with only the header. No data, cause unknown.
- **Debugger:** gdb is not installed. `lldb-20` on the debug `.so` with `breakpoint set -n ttsim_error` gives a backtrace into `noc_cmd_ctrl<'T'>` at tile.cpp:1610. `sudo apt install gdb` is possible.
- **Raw dlopen harness** (`workbench/harness/h.c`): reading PCI config returns 0x401e1e52 on WH and 0xb1401e52 on BH; `libttsim_clock(1000)` works.
- **Multichip:**
  - The stock distributed examples (2x4 mesh) FAIL.
  - `wh_x8` fails with "UndefinedBehavior: eth base fw reported error code 1".
  - `wh_x2` with `MeshShape(2,1)` and `TT_METAL_MOCK_CLUSTER_DESC_PATH=.../wormhole_N300.yaml` works: `host/dbrw.cpp` printed "devices=2 match=true" in 7.6 s.
  - CCLs and fabric are untested.
- **SFPU examples** (eltwise_sfpu, custom_sfpi_add, sfpu_eltwise_chain, hello_world_datatypes) pass on both WH and BH.
- **Trace** needs fast dispatch and is untested. Tracy is untested.

## Build agent's final report: ttnn and the toolchain
- Toolchain: clang 20.1.8, cmake 4.0.2, SFPI 7.92.0 (riscv-tt-elf-g++ 15.1.0). On Debian 12 the SFPI .deb fails; the SFPI tarball that CMake downloads works.
- **Python ttnn works on the simulator.** Activate it with `source $TT_METAL_HOME/python_env/bin/activate` (Python 3.11); the example script is `refs/sim-runs/ttnn/ttnn_add.py`.
  - A 32x32 bf16 `ttnn.add` passes on both WH (2.7 s) and BH (3.5 s).
  - Accuracy: PCC 0.99998, and 120 of 1024 elements are 1 bf16 ulp away from torch. So the result is NOT bit-identical to torch, and the page should not claim it is.
- **All single-chip examples pass on both WH and BH.** Wall-clock times, cold cache / warm cache:

  | Example | Cold | Warm |
  |---|---|---|
  | add_2_integers_in_riscv | 1.08 s | 0.36 s |
  | matmul_single_core | 8.9 s | 8.0 s |
  | matmul_multicore_reuse_mcast | 20 s | 18.7 s |
  | noc_tile_transfer | — | — |

  noc_tile_transfer prints "Result = 14 : Expected = 14"; it needs `TT_METAL_DPRINT_CORES='(0,0),(0,1)'`.
- **The distributed_* examples fail**: they need 8 devices. The `wh_x8` build fails at the Ethernet firmware stage, which matches the known issue that it needs the micah/native-sim branch.
- Logs live under `refs/sim-runs/`: `bh/`, `warm/`, `dprint/`, `ttnn/`, `multichip/`. The simulator binaries are in `~/sim/{wh,bh,wh_x2,wh_x8}`.
- The bracketed number in log lines is the simulated clock count. The seconds figure is host CPU time. The KHz figure is the effective simulation rate.
