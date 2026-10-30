# Research: TT-NN, models and multichip (tt-metal 7aad8aac)

## Stack (top to bottom)
1. **Frameworks**
   - PyTorch and JAX use **TT-XLA** (a PJRT plugin; installs as `pjrt-plugin-tt`; emits StableHLO).
   - ONNX, TF and Paddle use **TT-Forge-ONNX**.
   - TT-Torch is deprecated.
2. **TT-MLIR.** Dialects: ttir, ttnn, ttkernel, ttmetal. Most lowering goes to TT-NN calls; the runtime is `tt::runtime::ttnn`.
3. **Models and training:** `models/` (tt_transformers, demos), the vLLM plugin, and tt-train (C++ autograd).
4. **TT-NN** (`ttnn/`): ops, Tensor, MeshDevice, CCLs, trace.
5. **TT-Metalium** (`tt_metal/`): host API, JIT build, fabric, distributed.
6. **TT-LLK** (`tt_metal/tt-llk`).
7. **Drivers:** UMD (submodule tt-umd), then KMD (`/dev/tenstorrent`).
8. **Hardware**, or ttsim.

Docs:
- https://docs.tenstorrent.com/forge/index.html
- https://docs.tenstorrent.com/tt-mlir/dialects-overview.html
- https://docs.tenstorrent.com/tt-xla/getting_started.html

## TT-NN basics
- **Opening a device.** `ttnn.open_device` is bound to `open_mesh_device`, so it returns a 1x1 mesh (ttnn/cpp/ttnn-nanobind/device.cpp:73-104). Arguments: `l1_small_size` (default 0; conv2d needs it) and `trace_region_size`.
- **Hello world:**
  ```python
  device = ttnn.open_device(device_id=0)
  a = ttnn.full(shape=(32,32), fill_value=1.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
  c = ttnn.add(a, b); ttnn.close_device(device)
  ```
  (ttnn/tutorials/basic_python/ttnn_add_tensors.py)
- **`from_torch`** defaults to ROW_MAJOR (ttnn/ttnn/operations/core.py:283). Dtypes: uint8, uint16, int32, uint32, float32, bfloat16, bfloat8_b, bfloat4_b.
- **Layout.** A tensor is a 2D buffer of pages, with outer dims folded into H.
  - Row-major: 1 page = 1 row.
  - Tile layout: 1 page = 1 tile.
  - Reference: tech_reports/tensor_layouts.
- **Memory.**
  - Banks: WH has 12 DRAM banks and BH has 8. L1 is 1 bank per core.
  - Allocation is lockstep: DRAM bottom-up, L1 top-down.
  - `ttnn.device.dump_device_memory_state(device, prefix=)` dumps the allocator state.
  - Ready-made configs: `ttnn.DRAM_MEMORY_CONFIG`, `L1_MEMORY_CONFIG`, `L1_{HEIGHT,WIDTH,BLOCK}_SHARDED_MEMORY_CONFIG`.
- **Matmul.**
  - Simple form: `ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=8, x=8))`.
  - Program configs (matmul_program_config_types.hpp): MultiCore, Reuse, ReuseMultiCast (2D), ReuseMultiCast1D, DRAMSharded.
  - 2D config fields: `compute_with_storage_grid_size`, `in0_block_w` (must divide Kt), `out_subblock_h/w` (h×w ≤ 8 tiles in BF16, 4 in FP32), `per_core_M/N`, `transpose_mcast` (required), `fused_activation`, `fuse_batch`.
  - The bindings use `noconvert`, so these must be real Python ints and bools.
- **Writing an op** (docs/source/ttnn/ttnn/adding_new_ttnn_operation.rst). Copy the example under `ttnn/cpp/ttnn/operations/examples/example/`. It has `operation_attributes_t`, `tensor_args_t`, `program_factory_t = variant<SingleCore,MultiCore>`, `create_descriptor` (the ProgramDescriptor pattern), `select_program_factory`, `validate`, and `compute_output_specs`. The single-core factory sets up 2-tile CBs `c_0`/`c_2`, a reader/writer pair, and the `eltwise_sfpu` compute kernel with `SFPU_OP_CHAIN_0` at HiFi4.
- **Kernel API migration.** Kernels are moving to "Metal 2.0" `DataflowBuffer`/`Noc` objects (`dfb.reserve_back`, `noc.async_read(s, dfb, ...)`). Pin examples to a commit.

## Sharding (tech_reports/tensor_sharding)
- **Height** sharding cuts horizontal bands; shard W must equal tensor W.
- **Width** sharding cuts vertical strips.
- **Block** sharding cuts a 2D grid; block (r,c) goes to core (x=c, y=r).
- **Orientation** ROW_MAJOR/COL_MAJOR sets the order in which shards are assigned to cores.
- **ND sharding** (experimental) assigns shards round-robin.
- **APIs:**
  - `TensorSpec(...).height_sharded(core_ranges)`
  - `ttnn.MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1, ShardSpec(grid, shard_shape, orientation))`
  - `ttnn.create_sharded_memory_config(shape, core_grid, strategy, use_height_and_width_as_shard_shape=)`
- **Why it matters:** keeping data in local L1 avoids DRAM round-trips, a form of fusion.
- **WH read bandwidth:** DRAM interleaved about 190 GB/s, DRAM sharded about 240 GB/s; L1 sharded is fastest.
- **Visualizer:** https://supermina999.github.io/sharding-visual/

## LLMs (tech_reports/LLMs/llms.md)
- **Prefill:** one user at a time; activations in DRAM; long sequences split into 2048-token chunks.
- **Decode:** batch of 32, one token each; activations width-sharded in L1. It is DRAM-bound on weight reads.
- **Paged KV cache:** `paged_fill_cache` and `paged_update_cache`.
- **Attention:** prefill uses `scaled_dot_product_attention` (FlashAttention-2). Decode uses `sdpa_decode`, with heads placed on the tile's y axis.
- **Tensor parallelism:**
  - Column-parallel layers need all-gathered inputs.
  - Row-parallel layers produce partial sums that need reduce-scatter.
  - The MLP pairs column (FF1/FF3) with row (FF2) so only one all-reduce is needed.
  - Galaxy uses 2D parallelism across both cluster axes.
- **Trace:**
  - API: `begin_trace_capture` / `end_trace_capture` / `execute_trace` / `release_trace`.
  - Needs `trace_region_size`. Captures ops only. Do a warm-up run first. No allocation inside a trace.
  - **Unsupported in slow dispatch** (tt_metal/distributed/sd_mesh_command_queue.cpp:487-493), so it does not work under the ttsim SD setup.
  - Current `generator.py` also traces prefill per sequence-length bucket.
- **Performance tips:**
  - Ladder: HiFi4 BF16 → HiFi2 BFP8 → LoFi BFP4.
  - Use subblocks of at least 2x1.
  - Use DRAM-sharded matmul for decode.
- **Accuracy targets (PCC):** 0.999 per submodule, 0.99 for the whole model.

## Multi-device
- **Mesh basics:** MeshDevice is SPMD. Shapes: N300 1x2, T3K 2x4, Galaxy 8x4.
- **Mapping tensors to devices:**
  ```python
  mapper = ttnn.ShardTensorToMesh(mesh, dim=3)
  ReplicateTensorToMesh(mesh)
  ConcatMeshToTensor(mesh, dim)
  ```
  `ttnn.visualize_mesh_device` and `visualize_tensor` show the result.
- **Fabric:** `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D|FABRIC_1D_RING|FABRIC_2D...)` must run BEFORE `open_mesh_device`.
- **CCLs:**
  - Current calls: `ttnn.all_gather(t, dim, cluster_axis=)` and `ttnn.reduce_scatter`. The `num_links` and `topology` arguments are deprecated and ignored.
  - Docs that pass `mesh_device=` are stale.
- **CCL bandwidth:**
  - Per link: WH 12.5 GB/s, BH 50 GB/s per direction.
  - Bytes on the wire: all-gather and reduce-scatter move (N-1)/N·T; all-reduce moves 2(N-1)/N·T.
- **TT-Fabric** (tech_reports/TT-Fabric):
  - Data plane: NoC plus Ethernet. Control plane: PCIe.
  - Links: TT-link adds a 16 B header and go-back-N retransmission.
  - Routing: source-routed within a mesh, up to 1024 meshes × 256 devices. WH has 4 routing planes, BH has 2. Dimension-order routing.
  - Galaxy fabric config: 1D ring on Galaxy, 2D torus XY for BH prefill.

## Profiling
- **Tracy:** build with `./build_metal.sh -p`, then run `python -m tracy -r -p -v -m pytest ...`. Output: `generated/profiler/reports/ops_perf_results*.csv` with columns such as DEVICE KERNEL DURATION, per-RISC times and CB waits.
- **Warm cache:** run twice, so the program cache is warm, and read the second run.
- **Reports:** `tt-perf-report`. Hardware perf counters: `--profiler-capture-perf-counters`.
- **Graph capture:** `ttnn.graph.begin_graph_capture(RunMode.NO_DISPATCH)` runs with no device needed. View the result in ttnn-visualizer.
- **Comparison mode:** `TTNN_CONFIG_OVERRIDES='{"enable_comparison_mode": true, ...}'`.

## Labs suited to the simulator (tiny shapes)
1. Add two tensors.
2. Padding: compare `shape` with `padded_shape`.
3. DRAM vs L1, and dump the memory state.
4. Height, width and block sharding of 128x128 on 4 cores.
5. Matmul with a 2D config, 64x64 on a 2x2 grid, LoFi vs HiFi4.
6. Dtype ladder: PCC for bfloat16 / bfp8 / bfp4.
7. Graph capture with NO_DISPATCH.
8. Comparison mode.
9. Read the example op.
10. A 1x2 mesh using `wh_x2` plus the N300 mock cluster descriptor.
11. all_gather.
12. Mini tensor-parallel MLP.
13. Trace (fast dispatch only, uncertain).
14. Tracy.

## Mismatches
- Stale CCL arguments in the docs.
- Prefill tracing: docs and code disagree.
- File names: `llama_attention.py` is now `attention.py`.
- Typos in doc snippets.
- WH bandwidth: 288 vs 336 GB/s.
