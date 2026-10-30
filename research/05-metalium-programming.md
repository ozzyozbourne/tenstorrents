# Research: TT-Metalium programming model (tt-metal 7aad8aac)

## Facts that shape the series
- **Mesh API only.** The legacy `CreateDevice`/`EnqueueProgram` API is gone, so a single chip is a 1x1 mesh:
  ```cpp
  auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
  auto& cq = mesh_device->mesh_command_queue();
  distributed::MeshWorkload workload; auto range = distributed::MeshCoordinateRange(mesh_device->shape());
  Program program = CreateProgram();
  ... workload.add_program(range, std::move(program));
  distributed::EnqueueMeshWorkload(cq, workload, false); distributed::Finish(cq);
  ```
  Sources: add_2_integers_in_riscv.cpp:29-40, loopback.cpp:115-119, distributed.hpp:38-128.
- **Two data-movement kernel styles.** The classic free-function style (`noc_async_read_page`, `cb_reserve_back`) and the "Device 2.0" object style (`Noc noc; CircularBuffer cb(id); noc.async_read(...)`), which eltwise_binary/eltwise_sfpu use. Migration guide: docs/source/tt-metalium/tt_metal/apis/kernel_apis/data_movement/device_api_migration_guide.md
- **Compute kernels are `void kernel_main()`**, not `namespace NAMESPACE { void MAIN }`.
- **DPRINT uses fmt-style formatting**: `DPRINT("x={}\n", x)`, and each print must end in `\n`. Enable with `TT_METAL_DPRINT_CORES=0,0`.
- **LLK location.** LLK lives in `tt_metal/tt-llk/`. Metal 2.0 ProgramSpec is experimental.

## Host
- **Buffers.**
  - `DeviceLocalBufferConfig{.page_size, .buffer_type=BufferType::DRAM|L1}` + `ReplicatedBufferConfig{.size}` → `MeshBuffer::create(...)`
  - Page size is the unit of round-robin distribution across banks, usually one tile (2048 B for bf16).
  - Sharded buffers take `.sharding_args = BufferShardingArgs(ShardSpecBuffer(...), TensorMemoryLayout::HEIGHT_SHARDED)` (vecadd_sharding.cpp).
  - I/O: `EnqueueWriteMeshBuffer(cq, buf, vec, false)` and `EnqueueReadMeshBuffer(cq, vec, buf, true)`.
- **Kernels.**
  - `CreateKernel(program, path, core_spec, DataMovementConfig{.processor=DataMovementProcessor::RISCV_0, .noc=NOC::RISCV_0_default, .compile_args=...})`
  - Processors: RISCV_0 = BRISC, RISCV_1 = NCRISC. NOCs: NOC_0 = 0, NOC_1 = 1. Convention: reader on RISCV_0/NOC0, writer on RISCV_1/NOC1. The two choices are independent.
  - `ComputeConfig{.math_fidelity=HiFi4 default, .fp32_dest_acc_en, .dst_full_sync_en, .math_approx_mode, .compile_args, .defines}` (kernel_types.hpp:106-134)
- **Circular buffers.**
  - `CreateCircularBuffer(program, core, CircularBufferConfig(n*tile_bytes, {{cb_idx, tt::DataFormat::Float16_b}}).set_page_size(cb_idx, tile_bytes))`
  - Up to 32 CBs. Two tiles per CB gives double buffering.
  - `.set_globally_allocated_address(*l1_buf)` makes a sharded buffer act as a CB without copying.
- **Args.**
  - Runtime args: `SetRuntimeArgs` on the host, `get_arg_val<uint32_t>(i)` on the device.
  - Common runtime args: `SetCommonRuntimeArgs` / `get_common_arg_val`.
  - Compile-time args: `.compile_args` / `get_compile_time_arg_val(i)`, plus named variants. A new value means a new binary.
  - Rule of thumb: layouts and sizes are compile-time; addresses and counts are runtime.
- **Semaphores.** `CreateSemaphore(program, cores, init)` returns an id; the kernel calls `get_semaphore(id)`.
- **SPMD.** `split_work_to_cores(grid, n)` returns `{num_cores, all, g1, g2, n1, n2}`.
- **Coordinates.** Convert logical cores to NoC coordinates with `worker_core_from_logical_core`.

## Data movement API
- **Addresses.**
  - `get_noc_addr(x,y,addr)`
  - `get_noc_multicast_addr(x0,y0,x1,y1,addr)`
  - `get_noc_addr_from_bank_id<true>(bank, addr)`
- **Transfers.**
  - `noc_async_read(src_noc, dst_l1, size)` + `noc_async_read_barrier()`
  - `noc_async_write` + `noc_async_write_barrier`
  - `noc_async_writes_flushed()` means the request has left the core and the source may be reused.
- **Interleaved buffers.**
  - `InterleavedAddrGen<true>{.bank_base_address, .page_size}` (older style)
  - `TensorAccessor` (modern). Host: `TensorAccessorArgs(*buf).append_to(ct_args)`. Kernel:
    ```cpp
    constexpr auto a_args = TensorAccessorArgs<0>(); auto a = TensorAccessor(a_args, addr);
    auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    noc_async_read_page(i, a, l1)
    ```
    (loopback_dram_copy.cpp:24-28)
- **CB semantics** (dataflow_api.h):

  | Call | Effect |
  |---|---|
  | `cb_reserve_back(cb,n)` | Spins until free ≥ n. No pointer moves. |
  | `get_write_ptr` | Returns the write pointer. |
  | `cb_push_back` | received += n; write pointer advances with wrap; publishes the data. |
  | `cb_wait_front` | Spins until available ≥ n. |
  | `get_read_ptr` | Returns the read pointer. |
  | `cb_pop_front` | acked += n; read pointer advances with wrap. |

  - The tiles_received and tiles_acked counters live in per-CB registers that all 5 RISCs can see.
  - Writing data does not publish it; only push does.
  - Pushes over one wrap cycle must add up to exactly the CB size.
  - Repeated `wait_front` calls without a pop must ask for cumulative totals.
  - Before NoC-writing or pushing data the RISC-V stored itself, call `ckernel::load_blocking(ptr)` to ensure the store has landed.
  - In compute kernels, `wait_front`/`pop_front` run on UNPACK and `reserve_back`/`push_back` run on PACK.
- **Semaphores and multicast.**
  - Semaphores: `noc_semaphore_wait(ptr,v)`, `noc_semaphore_set`, `noc_semaphore_inc(noc_addr,1)`, `noc_semaphore_set_multicast`.
  - Multicast: `noc_async_write_multicast(src, mcast_addr, size, num_dests)`.
  - Ordering: writes are ordered only on the same NoC and the same virtual channel (ordering.rst).

## Compute
- **One source, three builds.** The JIT generates `chlkc_{unpack,math,pack}.cpp`: the same source with `#define TRISC_UNPACK`, `TRISC_MATH` or `TRISC_PACK` (jit_build/genfiles.cpp:78-97, 791-846). The macros `UNPACK(..)`, `MATH(..)` and `PACK(..)` expand on one TRISC only (common_globals.h). Example:
  ```cpp
  ALWI void add_tiles(icb0, icb1, itile0, itile1, idst) {
      UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
      MATH((llk_math_eltwise_binary<ELWADD, NONE, fp32, MathFidelity::LoFi, NONE>(icb0, icb1, idst, true)));
  }
  ```
  (hw/inc/api/compute/eltwise_binary.h:266-275)
- **Dst handshake.** `tile_regs_acquire` (MATH waits) → `commit` (MATH posts semaphore MATH_PACK) → `wait` (PACK SEMWAITs) → `release` (PACK frees; ZEROACC in full-sync mode). Dst capacity:

  | | 16-bit | 32-bit |
  |---|---|---|
  | Double-buffered (default) | 8 tiles | 4 tiles |
  | Full sync | 16 tiles | 8 tiles |

  Recommended loop: wait → acquire → math → commit → pop/reserve → wait → pack → release → push.
- **API calls.**
  - `compute_kernel_hw_startup(icb0, icb1, ocb)` runs once. Matmul uses `<SrcOrder::Reverse>` because in0 goes to SrcB and in1 to SrcA.
  - Init functions such as `add_init`, `matmul_init` and `exp_tile_init` reprogram the MOPs. A missing SFPU init gives silently wrong results.
  - FPU ops read CBs and write Dst: `add_tiles`, `matmul_tiles` (which accumulates). Add and sub always run at LoFi; only ELWMUL uses the fidelity setting.
  - SFPU ops need data in Dst: `copy_tile(cb,i,dst)`, then `exp_tile(dst)` and so on. The softplus chain in sfpu_eltwise_chain is exp, add_binary_tile, log.
  - `pack_tile(dst, ocb)`.

## LLK layers
1. Compute API (hw/inc/api/compute).
2. Metal LLK wrappers (hw/ckernels/<arch>/metal/llk_api).
3. tt-llk `_llk_*_` functions (tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib).

- **Encoding.** `TT_OP(opcode,params) = (opcode<<24)+params`; ELWADD is 0x28. `TTI_*` compiles to `__builtin_rvtt_ttinsn`.
- **MOPs.** Init builds `ckernel_template(outer, inner, ops)` and writes the MOP config. Each per-tile call is `TTI_MOP` plus `SETRWC`. This is why init calls matter.
- **Matmul** issues `TTI_MVMUL`. Unpack uses UNPACR and pack uses PACR.
- **SFPU** is written in SFPI (`sfpi::vFloat`, `dst_reg[i]`, `v_if`). The per-face function is called 4 times; on WH it loops i<8 over vectors of 32 lanes (custom_sfpi_add).

## Examples, easiest to hardest (tt_metal/programming_examples)
1. hello_world_datamovement_kernel
2. hello_world_compute_kernel (prints TR0/1/2)
3. hello_world_datatypes
4. add_2_integers_in_riscv
5. loopback
6. add_2_integers_in_compute
7. eltwise_binary
8. eltwise_sfpu
9. sfpu_eltwise_chain
10. custom_sfpi_add / smoothstep
11. contributed/vecadd
12. vecadd_multi_core
13. NoC_tile_transfer
14. shard_data_rm
15. vecadd_sharding
16. pad_multi_core
17. contributed/multicast
18. matmul_single_core (tilize_nfaces, mm.cpp)
19. matmul_multi_core
20. matmul_multicore_reuse (intermediate CB c_24)
21. matmul_multicore_reuse_mcast (4 roles, 4 semaphores)
22. long_matmul
23. distributed/1-4
24. profiler/*

## Dispatch
- **Fast dispatch** (default) runs cq_prefetch.cpp and cq_dispatch.cpp firmware on dispatch cores. `DispatchCoreConfig` defaults to WORKER cores (the guide says Ethernet).
- **Slow dispatch** (`TT_METAL_SLOW_DISPATCH_MODE=1`): the host does every step synchronously.
- **Queues** are in-order. Two CQs synchronize with `MeshEvent`.
- **Trace** (`begin_mesh_trace`) requires fast dispatch and a `trace_region_size`.

## Tools
- **DPRINT**: `TT_METAL_DPRINT_CORES`, `_RISCVS`, `_FILE`; `TileSlice` prints a tile.
- **Watcher** (`TT_METAL_WATCHER=<s>`): waypoints such as CRBW and NRBW, NoC sanitizing, asserts.
- **Device profiler** (`TT_METAL_DEVICE_PROFILER=1`): `DeviceZoneScopedN("x")`, output in `generated/profiler/.logs/profile_log_device.csv`. It cannot run together with DPRINT or Watcher.
- **Tracy**: `python -m tracy`.

## JIT build
- Kernels are compiled at runtime with SFPI `riscv-tt-elf-g++`. CPU flags: `-mcpu=tt-wh-tensix` for compute, `tt-wh` for data movement.
- Cache: `~/.cache/tt-metal-cache/<build_key>/kernels/<hash>`. Override with `TT_METAL_CACHE`; force recompilation with `TT_METAL_FORCE_JIT_COMPILE`.
- Compile-time args are passed as `-DKERNEL_COMPILE_TIME_ARGS`.

## Links
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/labs/matmul/lab1/lab1.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/sfpu/llk.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html
- https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_print.html
- https://github.com/tenstorrent/sfpi
- https://github.com/tenstorrent/tt-llk
