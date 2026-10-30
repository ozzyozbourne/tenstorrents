# Tenstorrent From Zero: series outline (v2)

The project has two interleaved series:

- **Series A** has 22 chapters that build understanding from first principles to expert level.
- **Series B** has 12 labs that teach the `ttsim` simulator and use it to verify Series A concepts by running real code.

Pinned sources: tt-metal `7aad8aac`, ttsim `b3620244` (v1.15.0), tt-isa-documentation `be833101`.

## Rules for every page
- Each page is one self-contained HTML file under `chapters/` (Series A) or `sim-labs/` (Series B), based on `assets/chapter-template.html`. Pages use `assets/style.css` and `assets/tt.js`, with no external JS or CSS.
- Explanations follow the 3Blue1Brown style:
  - Open with a concrete question.
  - Build intuition with a picture before formalising.
  - Name the concept only after the reader has felt why it is needed.
  - End with a recap.
- Each page has at least **two interactive animations** (canvas or SVG, with play/step controls and narration). One of them is the chapter's "aha" animation below.
- Code excerpts cite `repo/path:lines` at the pinned commits. Any output shown must be real output captured in this orb; nothing is invented.
- Each page has 3–6 self-check quiz questions with answers in `<details>`, and a "Further reading" section of verified links (papers, docs, talks).
- **Honesty banner on every lab:** ttsim is a functional, bit-exact model, not a timing model. Its clock counts are not silicon performance.
- Keep a glossary for words that mean several things: *tile* (a 32×32 data tile, a Tensix tile, or a ttsim NoC node), *core*, *page*, *bank*, and *semaphore* (the Tensix Sync-unit semaphore vs a Metalium L1 semaphore).

## Series A: Tenstorrent From Zero

| # | File | Title | Contents | Opening question / aha animation | Lab |
|---|---|---|---|---|---|
| 01 | a01-why-new-hardware | Why AI needs new hardware | Matmul is ~99.8% of FLOPs; arithmetic intensity; the roofline; energy cost of compute vs data movement (Horowitz); end of Dennard scaling; a 5-line `ttnn.add` teaser | "This chip does ~1000 trillion multiplies/s. Why does adding two tensors use 1% of it?" / slide N up the roofline; energy bars | — |
| 02 | a02-gpu-vs-tenstorrent | Two philosophies: GPU vs Tenstorrent | SIMT, warps, SMs, caches vs a grid of independent cores with software-managed SRAM; systolic arrays, TPU, dataflow chips; a preview of the 32×32 tile | "If you knew exactly which data each core needs next, would you still want a cache?" / the same blocked matmul on both machines with DRAM-trip counters | — |
| 03 | a03-chip-tour | A tour of the chip | Wormhole and Blackhole grids; Tensix, DRAM, Ethernet, PCIe, ARC and L2CPU tiles; harvesting; generations; physical vs logical coordinates | "Why does an 80-core chip sell as 72 or 64?" / clickable grid map; harvest a row | B02 |
| 04 | a04-inside-tensix | Inside a Tensix core | The 5 baby RISC-Vs and their roles; L1; the coprocessor (unpack, FPU, SFPU, pack, ThCon); pushing instructions; "RISC-V runs ahead"; the Sync unit (semaphores, mutexes) | "Why does one core need five CPUs, none of which does the math?" / TRISCs push into three queues and race ahead; toggle sync off and watch the result corrupt | B03, B04 |
| 05 | a05-tiles-and-formats | Tiles, faces and number formats | 32×32 tiles, 16×16 faces, padding; FP32/BF16/TF32/FP16; block floating point (BFP8/4/2); measuring correctness (PCC, tolerances) | "Can 16 numbers share one exponent and stay useful?" / BFP8 block with a draggable outlier | B08 |
| 06 | a06-matrix-engine | The matrix engine and math fidelity | SrcA/SrcB/Dst sizes; MVMUL = 8×16 @ 16×16; 5-bit × 7-bit multipliers; fidelity phases; deriving peak TFLOPS | "How do 5×7-bit multipliers compute a BF16 matmul?" / long-multiplication quadrants lit phase by phase | B07 |
| 07 | a07-vector-engine | The vector engine (SFPU) | 32 lanes, LRegs, Dst↔LReg, predication masks, instruction set; why the SFPU is 64× below the FPU; Wormhole vs Blackhole | "Why is exp() 64× slower than matmul on the same core?" / v_if lanes masked; both branches execute | B08 |
| 08 | a08-noc | The Network-on-Chip | Two opposite tori; dimension-order routing; flits and packets; unicast, multicast, atomics; virtual channels; ordering; coordinates | "Why build two networks running in opposite directions?" / NoC0 vs NoC1 hop counter; multicast rectangle | B05 |
| 09 | a09-memory | Memory: L1, DRAM, interleaving and sharding | No caches; L1 scratchpad and banking; DRAM banks; page interleaving; height/width/block sharding; allocator sidebar | "With no cache, who decides where each byte lives?" / one tensor redistributed across banks vs cores | B05, B11 |
| 10 | a10-first-program | Your first TT-Metalium program | MeshDevice, MeshBuffer, Program, CreateKernel, runtime vs compile-time args, enqueue/finish; `add_2_integers_in_riscv`; DPRINT as your first tool | "What is the least code that makes the chip compute 1+2?" / host↔device timeline; the compile bar vanishes on rerun | B01 |
| 11 | a11-dataflow-and-cbs | Data movement kernels and circular buffers | Reader/writer kernels; `noc_async_*` and barriers; TensorAccessor; exact CB semantics; double buffering; the two kernel-API dialects; Watcher | "How do two cores with no shared variables agree a tile is ready?" / CB ring with reserve/push/wait/pop pointers | B06 |
| 12 | a12-compute-kernels | Compute kernels: one source, three cores | UNPACK/MATH/PACK macros; the Dst handshake (acquire/commit/wait/release); FPU vs SFPU ops; `copy_tile`; init functions; SFPI | "How can one source file run on three CPUs at once?" / one source shown three times with greyed-out macros, turning into a pipeline | B04, B06 |
| 13 | a13-llk-mops-jit | Under the hood: LLK, MOPs and the JIT | Compute API → LLK → Tensix instructions; MOP and replay expanders; instruction encoding; JIT build; kernel cache; kernel path search | "`add_tiles` is one line. How many hardware instructions is it?" / zoom from add_tiles to an expanded instruction stream | B04 |
| 14 | a14-one-tiles-journey | One tile's journey | Synthesis chapter: follow one bf16 tile DRAM → NoC → CB → UNPACR → SrcA/B → ELWADD → Dst → semaphore → PACR → CB → NoC → DRAM, at chip, core and register zoom levels | "Where is this tile right now?" / scrubbable timeline across three zoom levels | B04–B06 |
| 15 | a15-matmul-one-to-many | Matmul on one core, then many | Tilize; matmul_tiles accumulation; in0→SrcB / in1→SrcA; blocking; subblocks; split_work_to_cores; data reuse | "One core takes T. Why don't 64 cores take T/64?" / blocked matmul over a grid with a DRAM-read counter | B07 |
| 16 | a16-multicast-sync | Multicast and cross-core synchronization | Metalium L1 semaphores vs Tensix semaphores; NoC ordering; `matmul_multicore_reuse_mcast` (4 roles, 4 semaphores) | "64 cores need the same row. Must it be read 64 times?" / unicast vs multicast with a semaphore count | B05 |
| 17 | a17-dispatch-host | Dispatch, command queues and the host | Fast vs slow dispatch; prefetcher and dispatcher firmware; dispatch cores; multiple CQs and events; trace; UMD/KMD/PCIe/TLBs | "What actually happens between EnqueueMeshWorkload and a kernel starting?" / command packets flowing host → prefetch → dispatch → workers | B01, B09 |
| 18 | a18-ttnn-user | TT-NN as a user | Tensors, from_torch, ROW_MAJOR vs TILE, padding, dtypes, memory configs, sharding configs, matmul program configs, PCC checks | "What does ttnn.from_torch do to your tensor?" / pad → tilize → page into banks | B11 |
| 19 | a19-op-anatomy-compilers | How an op is built, and the compiler stack | Device operation structure (ProgramDescriptor, program factory, kernels); program cache; TT-Forge / TT-MLIR / TT-XLA lowering | "What happens between ttnn.add(a,b) and the FPU?" / one call descending the stack, each layer emitting its artifact | B11 |
| 20 | a20-performance | Thinking like a performance engineer | Which wall: compute (fidelity, format), L1/NoC, DRAM, dispatch/host; the profiler, Tracy, the ops CSV; a checklist | "Which wall is your op hitting?" / multi-roofline with toggles | — |
| 21 | a21-many-chips | Many chips: Ethernet, meshes and TT-Fabric | Ethernet cores; n300, T3K, Galaxy; MeshDevice; mappers; CCLs (all-gather, reduce-scatter, all-reduce); tensor parallelism; TT-Fabric layers | "How do 32 chips with no shared memory act as one?" / ring all-gather; column→row parallel MLP | B10 |
| 22 | a22-llm-capstone | Capstone: running an LLM | A transformer primer; prefill vs decode; KV cache; decode is DRAM-bound; sharded and DRAM-sharded matmuls; precision ladder; trace; tt_transformers; a tour of tt-lab | "Why does generating one token take about as long as reading every weight once?" / decode roofline with a batch slider; growing KV cache | — |

## Series B: ttsim Labs

Every lab is run in this orb. Its page shows the exact commands, any patches, and the real output. Each lab lists the Series A prerequisites it builds on.

| # | File | Title | What you do | Validates |
|---|---|---|---|---|
| B00 | b00-workbench | Setting up the workbench | Install or build ttsim and tt-metal; `env.sh`, `build_host.sh` and `build_ttsim.sh`; kernel overlay via TT_METAL_KERNEL_PATH; the soc_descriptor.yaml gotcha | — |
| B01 | b01-meet-ttsim | Meet ttsim | Run the examples on WH and BH; determinism; what the clock count means and does not mean; what ttsim models (functional, bit-exact) vs not (timing, caches, contention); a tour of the source and the clock loop; first edit: change `+` to `*` in a kernel | A10, A17 |
| B02 | b02-map-and-boot | Map the chip and watch it boot | Instrument chip init and SOFT_RESET; see which cores leave reset and at which PCs; compare with the SoC descriptor | A03, A04 |
| B03 | b03-riscv-interpreter | The RISC-V interpreter | Decode table; the TTI escape for Tensix instructions; count instructions per core between kernel start and end | A04 |
| B04 | b04-tensix-instructions | Following Tensix instructions | Trace opcodes per pipe for add_2_integers_in_compute; MOP expansion; stalls on Src valid; map opcodes with tensix_isa.json | A04, A12, A13, A14 |
| B05 | b05-noc-and-dram | NoC transactions, DRAM and the alignment trap | Log NoC transfers for loopback and multicast; break alignment so both ttsim and Watcher catch it; why missing barriers hide in ttsim | A08, A09, A11, A16 |
| B06 | b06-cbs-and-semaphores | Circular buffers and semaphores | Watch CB pointers with DPRINT; trace SEMPOST/SEMGET for the Dst handshake; **write your own** reader/compute/writer kernel | A11, A12 |
| B07 | b07-matmul-fidelity | Matmul and math fidelity, bit for bit | Run `mm_fid` at each fidelity with a fixed seed; dump one MVMUL's SrcA/SrcB/Dst from instrumented ttsim; reproduce it in Python | A06, A15 |
| B08 | b08-formats-and-sfpu | Data formats and SFPU numerics | BF16/FP32/BFP8 behaviour; Dst encoding; **write a custom SFPI kernel**; WH vs BH numerics | A05, A07 |
| B09 | b09-breaking-things | Breaking things on purpose | The error taxonomy; trigger UB/NCB/Unsupported; lldb on a debug build; what ttsim cannot see; capstone: drive libttsim directly via dlopen | Simulator contract, A17 |
| B10 | b10-two-chips | Two chips | `wh_x2` with the N300 cluster descriptor; a 2-device mesh buffer; what multichip support exists and what does not | A21 |
| B11 | b11-ttnn-on-sim | TT-NN on the simulator | ttnn tensors, layouts, sharding, memory dumps, format-ladder PCC, a small matmul with program configs (if ttnn runs under ttsim) | A09, A18, A19 |

Reading order in the index: A01–A03, then B00–B02, A04, B03, A05–A07, B07–B08, A08–A09, B05, A10–A12, B06, A13–A14, B04, then A15–A22 with B09–B11 placed beside them.
