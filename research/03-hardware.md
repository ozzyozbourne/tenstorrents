# Research: Tenstorrent hardware architecture

Sources: [TM] = tt-metal 7aad8aac, [SIM] = ttsim b3620244, [ISA] = tt-isa-documentation be833101 (cloned at /tmp/tt-isa-documentation). ISA URLs have the form https://github.com/tenstorrent/tt-isa-documentation/blob/main/<path>.

## Generations
| | Grayskull | Wormhole B0 | Blackhole A0 | Quasar (pre-silicon) |
|---|---|---|---|---|
| Tensix per die | 120 | 80 | 140 | 32 nodes in descriptor |
| Products | e75: 96, e150: 120 | n150: 72; n300: 2 × 64; Galaxy: 80 per chip | p100: 120; p150: 140 ⚠ (docs.tenstorrent.com says 120) | |
| NoC grid | ? | 10x12 | 17x12 | 11x8 |
| L1 per Tensix | 1 MB | 1464 KiB (1,499,136 B) | 1536 KiB | 4 MiB |
| DRAM | 8 GB LPDDR4 | 12 GiB GDDR6, 6 × 2 GiB, 192-bit bus, 288 GB/s at 12 GT/s | 32 GiB GDDR6, 8 × 4 GiB, 256-bit bus, 512 GB/s | 1 GiB per bank |
| Clock | 1.0 / 1.2 GHz | 1.0 GHz | 1.35 GHz | |
| Ethernet | none | 16 × 100 GbE | 14 tiles × 400G; ≤12 usable, 8 wired on p150 | |
| Host link | | PCIe 4 x16 | PCIe 5 x16 | |
| Large CPUs | | ARC | ARC + 16 × SiFive x280 (4 L2CPU clusters, NoC (8,3), (8,5), (8,7), (8,9)) | |

- WH worker columns: x=1-4 and 6-9; rows y=1-5 and 7-11. DRAM at x=0,5; Ethernet at y=0,6; ARC (0,10); PCIe (0,3) ([TM] tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml).
- BH worker columns: x=1-7 and 10-16; rows y=2-11. DRAM at x=0,9; Ethernet at y=1; PCIe (2,0), (11,0); ARC (8,0) ([TM] blackhole_140_arch.yaml).
- RISC-V counts *(derived)*:
  - WH: 400 + 16 = 416 RV32 cores.
  - BH: 700 + 28 (Ethernet, 2 each) + 24 (DRAM tiles) + 16 x280 = 768.
- Quasar worker node: 8 DM cores plus 4 Tensix engines with 4 TRISCs each; MX formats ([TM] tt_metal/hw/inc/internal/tt-2xx/quasar/core_config.h).

## Tensix tile ([ISA] WormholeB0/TensixTile/README.md)
- Contents: L1, 5 baby RV32IM cores, 2 NoC connections plus the overlay, 2 unpackers, FPU, SFPU (32 lanes × 32 bit), ThCon, 4 packers, mover, mailboxes and PIC.
- The five cores and their Metalium roles:
  - B (BRISC): data movement (DM0)
  - NC (NCRISC): data movement (DM1)
  - T0: unpack
  - T1: math
  - T2: pack
- WH core details:
  - In-order, 1 GHz. `fence` is a no-op. No A or C extensions; adds a `.ttinsn` instruction.
  - Data RAM: B 4K, T 2K each, NC 4K. NC also has a 16K IRAM.
- BH core details:
  - Adds Zicsr, Zaamo, Zba, Zbb and partial F/Zfh. T2 has a partial V extension.
  - Data RAM: B and NC 8K, T 4K.
  - 64 B L0 data cache, not coherent.
- Who can push instructions: B can push to any thread; NC cannot.
- L1 cost for baby cores: one store per 5 cycles; load latency ≥8 cycles (local RAM: 2). These cores are meant to direct the engines, not move data themselves ([ISA] L1.md).
- L1 on WH: 16 banks × 91.5 KiB, 128-bit ports, round-robin. Narrow writes and atomics are read-modify-write and take 5 cycles. Packers can accumulate in L1.
- Client bandwidth: an unpacker gets 4 × 128 b/cycle (5 when both run); each packer 1 × 128 b; each NoC 256 b/cycle in each direction.

## Coprocessor ([ISA] TensixCoprocessor/README.md)
- 3 threads, each issuing up to one instruction per cycle. The coprocessor does not fetch; the RISC-V cores push instructions to it, then run ahead asynchronously, so synchronization must be explicit.
- Frontend per thread: MOP expander → replay expander → wait gate (in order).
- Backend units: Sync, Unpack, Matrix (FPU), Pack, Vector (SFPU), Scalar (ThCon), Config, Mover, Misc.
- Push path: `sw` to INSTRN_BUF_BASE 0xFFE40000. `.ttinsn` is the same instruction rotated by 2 bits.
- MOP expansion is configurable through MOP_CFG and can produce up to 32,639 instructions. The replay buffer holds 32 entries.
- BH can fuse 4 `.ttinsn` instructions into one cycle.
- TTSync: a load that stalls until the coprocessor is idle.

## Sync unit ([ISA] SyncUnit.md)
- 7 mutexes and 8 semaphores (4-bit value, max 15). Instructions: SEMINIT, SEMPOST, SEMGET, SEMWAIT.
- TRISC cores can `lw` or `sw` semaphores directly.
- How Metalium uses them:
  - `tile_regs_acquire` / `commit` run on MATH; `tile_regs_wait` / `release` run on PACK ([TM] tt_metal/hw/inc/api/compute/reg_api.h:50-103).
  - In LLK, PACK issues `SEMWAIT(MATH_PACK, STALL_ON_ZERO)` followed by `SEMGET` ([TM] tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h:23-42).
  - MATH sets the semaphore max to 1 for SyncFull and 2 for SyncHalf (llk_math_common.h:98-108).

## Registers
- **SrcA/SrcB:** 2 banks × 64 × 16 × 19 bits each. One bank holds one tile (4 faces × 16 rows).
- **Dst:** 1024 × 16 × 16 bits = 32 KiB. Holds 16 BF16 tiles or 8 FP32 tiles, halved under half-sync (8 or 4). No reset value.
- **LReg:** 17 × 32 lanes (8 general, the rest constants).
- **Data flow:** Unpack0 → SrcA (or Dst), Unpack1 → SrcB. The FPU does Dst += f(SrcA, SrcB). SFPLOAD and SFPSTORE move between Dst and LReg. Packers move Dst → L1.
- FPU ops read directly from CBs (via unpack). SFPU ops need `copy_tile` into Dst first ([TM] METALIUM_GUIDE.md:406-444).

## FPU / SFPU throughput
- **MVMUL:** Dst(8×16) += SrcB(8×16) @ SrcA(16×16). Issues 1 per cycle with 5-cycle latency; each fidelity phase is a separate instruction.
- **Per Tensix at 1 GHz, by number of fidelity phases (1 / 2 / 3 / 4):**
  - MVMUL: 4.096 / 2.048 / 1.366 / 1.024 TF
  - GAPOOL: 2.048 → 0.512
  - ELWMUL: 0.256 → 0.064
  - ELWADD: 0.256 (with AddDst) or 0.128
  - SFPMAD: 0.064 TF on WH, 0.0864 on BH, i.e. 64× below LoFi MVMUL
- **ISA vs tech report.** The tech report matrix_engine.md gives different elementwise figures (0.128) and miscounts "4096 muladds" (it is 2048 MACs = 4096 FLOPs). Trust the ISA docs.
- **Opcode counts:** 128 on WH, 137 on BH.

## Tiles and formats
- Tile 32×32 = 4 faces of 16×16, because the engine natively multiplies 16×16 ([TM] tech_reports/tensor_layouts/tensor_layouts.md:59-62). Inputs are padded to multiples of 32.
- Tile byte sizes: FP32 4096; BF16 2048; Bfp8_b 1088 (1024 + 64 exponent bytes); Bfp4_b 576; Bfp2_b 320.
- BFP: 16 values share one 8-bit exponent (the "a" variants use 5 bits). Unpacking to BF16 is lossless. Packing rounds to nearest-even and saturates on overflow.
- Src registers are 19-bit: TF32 (1/8/10), BF16, FP16, INT8. FP32 is truncated to TF32 on entry. Full FP32 exists only in Dst and the SFPU. On WH, SrcA ignores the TF32 mantissa LSB.
- Src/Dst FP16 is not IEEE: infinity is non-standard and there is no NaN.

## Fidelity (verified)
- Each pass multiplies 5 bits of SrcA (hidden bit + 4) by 7 bits of SrcB (hidden bit + 6).
- Phases for BF16:
  - 0: high A × high B
  - 1: A's low 3 bits
  - 2: B's low bit
  - 3: low × low
- LoFi = phase 0; HiFi2 = phases 0-1; HiFi3 = 0-2; HiFi4 = 0-3.
- Cycles per 32×32×32 tile: 16 / 32 / 48 / 64.
- Phases needed for full precision: FP8, BFP4 and BFP2 need 1; BFP8 needs 2; BF16 needs 4. TF32 is never fully exact.
- Related config: `fp32_dest_acc_en`, `math_approx_mode`, `packer_l1_acc`.

## NoC
- Two NoCs, each a 2D torus: NoC0 runs right/down, NoC1 left/up.
- Packets: header + up to 256 flits.
  - WH: 32 B flits, 8 KiB max. BH: 64 B flits, 16 KiB max.
- Timing: 1 flit per cycle; 9 cycles per hop; about 5 cycles NIU → router.
- Link bandwidth *(derived)*: WH 32 GB/s, BH 86.4 GB/s.
- Routing: dimension-ordered (NoC0 goes right first, then down).
- Transactions: read, write (posted or inline), atomics, rectangle multicast to Tensix tiles. BH adds strided and L-shaped multicast.
- Deadlock avoidance: virtual channels plus a dateline. Ordering is weak, with completion counters.
- Coordinates: raw per NoC, plus a translated scheme (WH Tensix at x 18-25, y 18-27) that moves harvested rows to the end. Metalium works in logical coordinates.
- Compute grid: 8×8 on n150, 13×10 on p150 (the remaining rows/columns presumably go to dispatch, inferred).
- Harvesting: the n150 disables 1 row and the n300 2 rows per chip; which rows varies by chip.

## Memory
- No data caches. The METALIUM_GUIDE (:233-237) says the chips "deliberately omit cache hierarchies… eliminates cache coherency protocols". DRAM is reached only by explicit NoC DMA.
- No coherence across chips: "no concept of shared memory or global addressability" ([TM] tech_reports/EthernetMultichip/BasicEthernetGuide.md:188).
- DRAM banks: WH 12 × ~1 GiB (two views per controller); BH 8 × 4080 MiB.
- Allocation:
  - Interleaving is round-robin by page, starting at bank 0.
  - Lockstep: every bank gets the same offset.
  - DRAM is allocated bottom-up, L1 top-down ([TM] tech_reports/memory/allocator.md).
- Measured WH DRAM bandwidth: 267 GB/s out of 288 (>92%) ([TM] tech_reports/Saturating_DRAM_bandwidth/).

## Ethernet
- WH: 1 ERISC core per Ethernet tile, 256 KiB L1, 100 GbE. Latency about 530-620 ns one way.
- BH: 2 RISC-V cores per tile, 512 KiB, 400 GbE. Two tiles feed one 800G port.
- Only ERISC cores can send, and only into a peer's L1. Multi-hop traffic is store-and-forward through EDM / TT-Fabric.
- Topologies:
  - n300: 2 links between the chips, one reserved for dispatch.
  - T3K: 2x4.
  - WH Galaxy: 4x8 mesh, 16 links per chip.
- Fast dispatch typically runs on Ethernet cores.

## Peak math
- Peak = 4096 FLOP/cycle/core ÷ phases × cores × clock.
- p150: 140 × 4096 × 1.35 GHz = 774 / 387 / 194 TF ✓
- p100: 664 / 332 / 166 ✓
- n150: 295 / 147.5 / 73.7. The quoted FP8 figure is 262 (×0.89 ⚠); the 148 and 74 figures match.
- WH die: 328 ✓ (Hot Chips 2024).
- Measured matmul: about 190 TF on WH, 580 on BH; utilization 93-96% ([TM] tech_reports/GEMM_FLOPS/GEMM_FLOPS.md).
- Fewer than 8 input rows costs 1/8 of throughput.

## Links
- https://github.com/tenstorrent/tt-isa-documentation
- https://docs.tenstorrent.com/aibs/wormhole/specifications.html
- https://docs.tenstorrent.com/aibs/blackhole/specifications.html
- https://tenstorrent.com/hardware/blackhole
- https://docs.tenstorrent.com/aibs/grayskull/specifications.html
- https://tenstorrent.com/hardware/galaxy
- https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf
- https://www.eetimes.com/tenstorrent-productizes-risc-v-cpu-and-ai-ip
- https://riscv.epcc.ed.ac.uk/assets/files/hpcasia25/Tenstorrent.pdf
- https://github.com/tenstorrent/tt-bh-linux

## Unverified
- WH FP8 ×0.89 factor
- Grayskull FPU throughput
- p150 core count (120 vs 140)
- BH Ethernet link count
- BH L1 banking
- Real FIFO depth
- Quasar details
- Mapping of in0/in1 to SrcA/SrcB
- FP16 NaN encodings: the ISA docs and the tech report disagree
