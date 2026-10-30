# Tenstorrent From Zero

A novice-to-expert series on Tenstorrent AI accelerators. It covers why they exist, how the hardware works, how to program it, and how to verify every idea on the [`ttsim`](https://github.com/tenstorrent/ttsim) simulator.

Open `index.html` in a browser. Every page is a self-contained HTML file with interactive animations; nothing is loaded from the network.

- **Series A** (`chapters/`): 22 chapters, from "why AI needs new hardware" through Tensix internals, the NoC, memory, TT-Metalium kernels, LLK, TT-NN, multichip, and running an LLM.
- **Series B** (`sim-labs/`): 12 hands-on simulator labs. Every command shown was run against ttsim, and the output on the page is the real output.
- `sim-labs/workbench/`: scripts to point tt-metal at ttsim, build your own host programs, and build instrumented copies of the simulator.
- `research/`: the research notes, writer briefs, and verification briefs used to produce the series.
- `OUTLINE.md`: the full plan.

## How the series was made
Each page went through the same steps:
1. Parallel research agents read tt-metal, ttsim and tt-isa-documentation.
2. A writer agent verified each claim against the primary source, then wrote the page and ran its code on the simulator.
3. An independent verifier agent re-checked the facts, code excerpts, simulator output and animations, and fixed what it found.

Pinned sources: tt-metal `7aad8aac`, ttsim `b3620244` (v1.15.0), tt-isa-documentation `be833101`.

ttsim is a functional, bit-exact model, not a timing model. Its clock counts are not silicon performance.
