# Tenstorrent From Zero

A novice-to-expert series on Tenstorrent AI accelerators. It covers why they exist, how the hardware works, how to program it, and how to verify every idea on the [`ttsim`](https://github.com/tenstorrent/ttsim) simulator.

Open `index.html` in a browser. Every page is a self-contained HTML file with interactive animations; nothing is loaded from the network.

- **Series A** (`chapters/`): 22 chapters, from "why AI needs new hardware" through Tensix internals, the NoC, memory, TT-Metalium kernels, LLK, TT-NN, multichip, and running an LLM.
- **Series B** (`sim-labs/`): 12 hands-on simulator labs. Every command shown was run against ttsim, and the output on the page is output a run produced (see "What has been re-verified" below for what repeats exactly and what does not).
- `sim-labs/workbench/`: scripts to point tt-metal at ttsim, build your own host programs, and build instrumented copies of the simulator.
- `research/`: the research notes, writer briefs, and verification briefs used to produce the series.
- `OUTLINE.md`: the full plan.

## How the series was made
Each page went through the same steps:
1. Parallel research agents read tt-metal, ttsim and tt-isa-documentation.
2. A writer agent verified each claim against the primary source, then wrote the page and ran its code on the simulator.
3. An independent verifier agent re-checked the facts, code excerpts, simulator output and animations, and fixed what it found.

Pinned sources: tt-metal `7aad8aac`, ttsim `b3620244` (v1.15.0), tt-isa-documentation `be833101`.

## What has been re-verified, and when
The pages were written and first run on one build machine in October 2026 (the pages call it "this orb"). In November 2026
two further rounds of independent verifier agents re-checked them on a second machine (Ubuntu 24.04) with the same pinned
commits: tt-metal `7aad8aac` built from source (C++ examples and Python `ttnn`), ttsim `b3620244` built from source.

- **Round 1 (17-20 November 2026)** checked facts, citations and code excerpts against the sources. tt-metal was not yet
  built on that machine, so Series A simulator output was not re-run; labs B00-B07 were re-run.
- **Round 2 (20-22 November 2026)** re-ran the simulator commands. Chapters A01-A17 and A20-A22: the commands shown were
  re-run and the deterministic output matched the pages, or the page was corrected. For A20-A22 this covers every counter
  table in A20 (19 `a20_count.py` cases plus the three matmul examples), the profiler transcript, all seven two-chip
  attempts in A21, and the three A22 scripts with their variants. The Python `ttnn` output of A05, A09, A18 and A19 was
  re-run in the same round. Labs B00-B11 were re-run in both rounds; each lab page states
  its own provenance and what was not re-run.
- **Not re-run:** the tt-metal build transcripts and timings in B00 (they are excerpts of the build machine's logs),
  anything described as measured on silicon (those numbers are quoted from Tenstorrent's reports), and vendor product
  pages that have changed since October 2026 (flagged on the page where that happened).

What repeats exactly and what does not:

- **Repeats exactly:** results, PCC values, hashes, byte and instruction counts, and ttsim's bracketed clock count
  (for example `[4501]`) for a plain run with the same binaries, kernels and seeds.
- **Varies between runs or machines:** wall-clock seconds and the KHz rate on ttsim's exit line; clock counts of runs
  with DPRINT or Watcher enabled (they add host polling whose amount depends on timing); clock counts of runs that
  end in a wall-clock timeout (A21's fabric run); outputs of programs that use unseeded random inputs; JIT cache hit
  counts, which depend on what is already in the kernel cache (set a private `TT_METAL_CACHE` to reproduce them);
  absolute paths, hash-named cache directories and addresses in logs.
- **Not shipped:** some pages and research notes cite files that existed only on the build machine:
  `refs/sim-runs/...` (run logs, including `refs/sim-runs/ttnn/ttnn_add.py`; the shipped replacement is
  `sim-labs/workbench/py/a05_add.py`) and `refs/build-logs/...` (build logs quoted in B00). Paths such as
  `/home/user/workspace/refs/tt-metal/...` inside quoted output are that machine's checkout; yours will differ.

ttsim is a functional, bit-exact model, not a timing model. Its clock counts are not silicon performance.
