# Research: why AI accelerators exist, GPU vs Tenstorrent, reading list

Compiled by a research subagent on 2026-10-30. Links were checked by HTTP fetch. [bot-blocked] means the content was confirmed through search excerpts. *(derived)* marks arithmetic done here.

## Corrections
- "Corsair" is d-Matrix's in-memory-compute accelerator, not a Tenstorrent product. Tenstorrent's systems are Galaxy, TT-LoudBox/T3000 and TT-QuietBox.
- HBFP (Drumond et al., NeurIPS 2018) and MSFP (Rouhani et al., NeurIPS 2020, Microsoft) are different works. The OCP MX spec (2023) descends from MSFP.
- There is no dedicated Chips and Cheese article on Tenstorrent.
- Tenstorrent's Hot Chips talks: HC32 (2020) and HC2024 (Blackhole).
- Announced vs shipped:
  - Blackhole: 140 Tensix cores and 745 TF FP8 at HC2024. The shipped p150 has 120 cores at 1.35 GHz and 664 TF BLOCKFP8.
  - Wormhole: 80 cores on the slide, 72 on the n150, 64 per chip on the n300 (harvesting).
- Do not use the "176 Tensix cores" figure; it appears on one page and contradicts every other source.

## First principles
- **Matmul dominance.** BERT: matmuls are 99.80% of FLOPs but 61% of runtime; normalization is 0.17% of FLOPs and 25.5% of runtime; element-wise ops are 0.03% and 13.5% (Ivanov et al., MLSys 2021).
- **Training cost.** About 6N FLOPs per token (Kaplan 2020). Attention's share is under 1% at 512 tokens and about 31% at 32K.
- **Matmul intensity** *(derived)*: 2N³ / 3N² elements ≈ N/3 FLOP/B in BF16. An element-wise add is about 0.17 FLOP/B.
- **Roofline** (Williams, Waterman and Patterson, CACM 2009): attainable = min(peak, BW × AI). Ridge points *(derived)*:
  - H100: 989 TF / 3.35 TB/s ≈ 295 FLOP/B
  - Wormhole n150: 74 TF / 288 GB/s ≈ 257 FLOP/B
  - Blackhole p150: 664 TF / 512 GB/s ≈ 1,300 FLOP/B, so data must be reused from SRAM
  - Blackhole SRAM bandwidth (HC2024 slide 14): local 94 TB/s, neighbour 47 TB/s, multicast 24 TB/s, gather 16 TB/s (3 hops) and 5 TB/s (10 hops).
- **Horace He's analogy:** compute = factory, bandwidth = trucks, overhead = paperwork.
- **Memory wall:** Wulf and McKee 1995. Gholami et al. 2024: FLOPS grew 3.0× per 2 years, DRAM bandwidth 1.6×, interconnect 1.4×.
- **Energy** (Horowitz ISSCC 2014, 45 nm; Sze's table):

  | Operation | Energy (pJ) |
  |---|---|
  | 8-bit add | 0.03 |
  | 32-bit add | 0.1 |
  | 8-bit multiply | 0.2 |
  | 32-bit multiply | 3.1 |
  | FP16 add / multiply | 0.4 / 1.1 |
  | FP32 add / multiply | 0.9 / 3.7 |
  | 32-bit SRAM read (8 KB) | 5 |
  | 32-bit DRAM read | 640 (≈173× an FP32 multiply) |

  An in-order core spends about 70 pJ per instruction on overhead.
- **Dennard scaling** (1974) ended in the mid-2000s, leading to dark silicon (Esmaeilzadeh, ISCA 2011). Under a fixed power budget, performance equals ops per joule, which favours specialization.

## Landscape
- **Kung 1982**, systolic arrays: data pulses through the array and is reused, like a bucket brigade.
- **TPU v1:**
  - 256×256 8-bit MACs, 92 TOPS, 28 MiB software-managed memory.
  - 15–30× faster than a K80 or Haswell, with 30–80× better TOPS/W.
  - Its determinism is better for tail latency than caches and out-of-order execution.
- **Eyeriss:** 168 PEs with a row-stationary dataflow; AlexNet convolutions at 35 fps and 278 mW.
- **Graphcore IPU:** tiles with private SRAM and bulk-synchronous exchange; 124.5 TF mixed precision.
- **SambaNova SN10:** 640 PCUs and 640 PMUs, more than 300 TF BF16, more than 300 MB on-chip, 150 TB/s.
- **Groq TSP:** deterministic and compiler-scheduled, 320×320 MXM, about 750 INT8 TOPS, no caches.
- **H100 SXM:**
  - 132 SMs, 528 tensor cores, warps of 32 threads (SIMT).
  - 256 KB L1/shared memory per SM (shared up to 228 KB), 50 MB L2.
  - 80 GB HBM3 at 3.35 TB/s; 989 TF dense BF16; NVLink 900 GB/s; 700 W.
  - Hopper's TMA and warp specialization move it toward Tenstorrent's reader/compute/writer split.

## GPU (H100 SXM) vs Tenstorrent (Blackhole p150)

| Dimension | GPU (H100 SXM) | Tenstorrent (Blackhole p150) |
|---|---|---|
| Compute unit | SM, SIMT warps | Tensix: 5 baby RISC-V cores, FPU, SFPU; one program per core |
| On-chip memory | Hardware caches plus shared memory | About 1.5 MB software-managed SRAM per core; 180 MB total |
| Data movement | Hardware caches; TMA | Explicit NoC reads, writes and multicasts in software |
| Interconnect | Crossbar to L2 | 2 NoCs, each a 2D torus, 64 B links at 83 GB/s |
| Native data unit | Element or thread | 32×32 tile made of four 16×16 faces |
| Off-chip memory | HBM3, 3.35 TB/s | GDDR6, 32 GB, 512 GB/s |
| Peak dense | 989 TF BF16 | 664 TF BLOCKFP8 |
| Control processors | Needs a host CPU | 16 SiFive X280 Linux-capable cores on-chip |
| Scale-out | NVLink + NVSwitch, InfiniBand | 10×400 GbE built in; switchless mesh. Galaxy: 32 chips in a 4×8 mesh |
| Formats | Down to FP8 | Includes BFP8/4/2 (16 elements share an exponent); math fidelity |
| Software | Mostly closed | Apache-2.0; public ISA docs |
| Power | 700 W | 300 W |
| Price | Not listed | p150 $1,399; Galaxy Blackhole $110K (April 2026) |

- **Tenstorrent's original thesis:** conditional execution and sparsity (Vasiljevic and Bajic, IEEE Micro 2021). These are design-intent claims with no independent benchmarks.
- **NoC hop latency:** about 9 cycles per hop (corsix part 3).

## History
- Founded in 2016 in Toronto by Ljubisa Bajic, Milos Trajkovic and Ivan Hamer.
- Jim Keller: CTO from about 2021, CEO from January 2023.
- Jasmina Vasiljevic leads the software stack; Davor Capalija co-presented HC2024.

| | Grayskull | Wormhole | Blackhole |
|---|---|---|---|
| Year | 2021 | 2022 | 2023; cards shipped April 2025 |
| Process | 12 nm | 12 nm | 6 nm |
| Tensix cores | 120 | 80 (72 on n150) | 140 as presented / 120 shipped |
| SRAM | 120 MB | 120 MB | 180 MB shipped |
| DRAM | LPDDR4, about 100 GB/s | 12 GB GDDR6, 288–336 GB/s | 32 GB GDDR6, 512 GB/s |
| Ethernet | None | 16×100 GbE | 10×400 GbE |
| Extra CPUs | | | 16 X280 |

- **Galaxy Wormhole:** 32 chips, 9.3 PF FP8, 12 kW.
- **Galaxy Blackhole:** 32 chips, about 23.8 PF FP8, $110K.
- **QuietBox:** 4 Blackhole chips, $11,999.
- **Matrix engine:** 8×16 × 16×16 per cycle = 4,096 FLOP/cycle, about 4 TF per core at 1 GHz in LoFi. The multiplier is 7b×5b with fidelity phases. 80 cores × 4 TF ≈ 320 TF, close to the HC2024 figure of 328.

## Number formats
- **bfloat16:** 1 sign, 8 exponent and 7 mantissa bits (Google Brain). Kalamkar 2019: bf16 training matches FP32 without loss scaling.
- **HBFP:** up to 8.5× throughput. **MSFP:** MSFP16 is 3× cheaper than bf16.
- **OCP MX:** block size 32 with an E8M0 scale; MXFP8, MXFP6, MXFP4 and MXINT8.
- **Tenstorrent BFP8/4/2:**
  - One shared exponent per 16 elements (one face row). "_b" means an 8-bit exponent.
  - BFP8_B: 1 sign bit plus 7 mantissa bits including an explicit hidden bit; round-to-nearest-even; saturates on overflow.
  - Lossless conversion to BF16 in the unpacker.
- **Fidelity:**
  - LoFi uses SrcA's hidden bit plus 4 mantissa bits and SrcB's hidden bit plus 6.
  - Throughput: LoFi 4, HiFi2 2, HiFi3 1.33, HiFi4 1 TF per core.
  - BF16 needs 4 phases for full precision; BFP8 needs 2.
  - Maximum multiplier accuracy is about TF32. Dst holds FP32.

## Bibliography (verified)
### First principles
- Horowitz 2014 PDF: https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf
- Sze, "Data movement dominates energy": https://www.nationalacademies.org/cdn/materials/9fba0a4f-e3a5-42ba-92e7-7f2865c62bb6
- Roofline: https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures (free: https://www.osti.gov/biblio/1407078)
- McKee, memory wall: http://svmoore.pbworks.com/w/file/fetch/59055930/p162-mckee.pdf
- AI and Memory Wall: https://arxiv.org/abs/2403.14123
- Dennard: https://ieeexplore.ieee.org/document/1050511
- Dark silicon: https://dl.acm.org/doi/10.1145/2000064.2000108
- Kaplan: https://arxiv.org/abs/2001.08361
- Ivanov: https://arxiv.org/abs/2007.00072
- Transformer FLOPs: https://www.adamcasson.com/posts/transformer-flops
- Sze survey: https://arxiv.org/abs/1703.09039

### Landscape
- Kung 1982: https://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf
- TPU v1: https://arxiv.org/abs/1704.04760
- Eyeriss: https://ieeexplore.ieee.org/document/7551407
- Cerebras HC31: https://old.hotchips.org/hc31/HC31_1.13_Cerebras.SeanLie.v02.pdf
- Graphcore IPU: https://arxiv.org/abs/1912.03413
- SambaNova: https://hc33.hotchips.org/assets/program/conference/day2/SambaNova%20HotChips%202021%20Aug%2023%20v1.pdf
- Groq HC34: https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf
- Hopper in depth: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth
- H100: https://www.nvidia.com/en-us/data-center/h100/
- CUDA programming model: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html

### Tenstorrent
- HC32 slides: https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_FPGA_Tenstorrent_v02.pdf
- HC32 video: https://www.youtube.com/watch?v=o5hhEJrHH4c
- IEEE Micro 2021: https://www.computer.org/csdl/magazine/mi/2021/02/09373921/1rPtnvMohpe
- HC2024 slides: https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf
- HC2024 video: https://www.youtube.com/watch?v=oeKRtNkx_Ek
- STH write-up: https://www.servethehome.com/tenstorrent-blackhole-and-metalium-for-standalone-ai-processing/
- The Register: https://www.theregister.com/on-prem/2024/08/27/tenstorrent-details-its-risc-v-packed-blackhole-chips/1322990
- SemiAnalysis 2021: https://newsletter.semianalysis.com/p/tenstorrent-wormhole-analysis-a-scale
- SemiAnalysis 2022: https://newsletter.semianalysis.com/p/tenstorrent-blackhole-grendel-and
- Next Platform 2020: https://www.nextplatform.com/ai/2020/04/07/changing-conditions-for-neural-network-processing/1651704
- Next Platform 2023: https://www.nextplatform.com/ai/2023/08/03/unleashing-an-open-source-torrent-on-cpus-and-ai-engines/1643791
- Keller and Bajic interview: https://www.youtube.com/watch?v=sMvudTBBQNw
- TT-Deploy: https://tenstorrent.com/en/newsroom/tt-deploy
- corsix part 1: https://www.corsix.org/content/tt-wh-part1
- corsix part 7: https://www.corsix.org/content/tt-wh-part7
- ISA docs: https://github.com/tenstorrent/tt-isa-documentation
- SrcA/SrcB: https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md
- tech_reports: https://github.com/tenstorrent/tt-metal/tree/main/tech_reports
- matrix_engine: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md
- METALIUM_GUIDE: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- Blackhole docs: https://docs.tenstorrent.com/aibs/blackhole/index.html
- Wormhole docs: https://docs.tenstorrent.com/aibs/wormhole/index.html
- Galaxy: https://tenstorrent.com/hardware/galaxy
- Galaxy Blackhole GA: https://www.theregister.com/2026/04/28/tenstorrent_galaxy_blackhole_ai_servers_ga
- FFT on Wormhole: https://arxiv.org/abs/2506.15437

### Number formats
- HBFP: https://arxiv.org/abs/1804.01526
- MSFP: https://proceedings.neurips.cc/paper/2020/hash/747e32ab0fea7fbd2ad9ec03daa3f840-Abstract.html
- OCP MX: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microscaling paper: https://arxiv.org/abs/2310.10537
- bfloat16 blog: https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus
- Kalamkar: https://arxiv.org/abs/1905.12322
- data_formats: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/data_formats.md

### Education
- 3Blue1Brown: https://www.3blue1brown.com/topics/neural-networks
- Brrr: https://horace.io/brrr_intro.html
- Boehm CUDA matmul: https://siboehm.com/articles/22/CUDA-MMM
- GPU MODE: https://github.com/gpu-mode/lectures
- Scaling book: https://jax-ml.github.io/scaling-book/tpus/
- PMPP: https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0

## Unverified
- Cerebras core counts.
- Conflicting Wormhole bandwidth figures (288 vs 336 GB/s).
- Grayskull TOPS (368 vs 276).
- SRAM per core is 1.5 MB in decimal, i.e. 1,464 KiB.
- Conditional-execution speedups.
