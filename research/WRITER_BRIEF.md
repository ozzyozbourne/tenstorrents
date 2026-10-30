# Brief for chapter writers

You are writing one page of "Tenstorrent From Zero", a series that takes a reader from novice to expert on Tenstorrent AI accelerators and the ttsim simulator. The reader is a capable programmer who knows a bit about GPUs and wants to understand why this hardware is built this way, how it works, how to program it, and how to verify each idea in the simulator.

## Inputs
- `/home/user/workspace/tenstorrent-from-zero/OUTLINE.md`: find your chapter's row. It gives the scope, the opening question, the "aha" animation, and the neighbouring chapters. Do not teach things that belong to later chapters; point forward to them instead.
- `/home/user/workspace/tenstorrent-from-zero/research/*.md`: research notes with citations. Treat them as leads, not truth.
- Primary sources, which are what you must verify against:
  - `/home/user/workspace/refs/tt-metal` (commit 7aad8aac)
  - `/home/user/workspace/refs/ttsim` (commit b3620244)
  - `/tmp/tt-isa-documentation` (the authoritative ISA spec). If it is missing, clone it with `git clone --depth 1 https://github.com/tenstorrent/tt-isa-documentation /tmp/tt-isa-documentation`.
- `/home/user/workspace/tenstorrent-from-zero/assets/chapter-template.html`: copy its structure exactly (header, nav, callouts, pager). `assets/tt.js` provides `TT.canvas`, `TT.stepper`, `TT.box`, `TT.arrow`, `TT.text`, `TT.clear`, `TT.lerp`, `TT.smooth` and the colour palette `TT.C`; read it. You may write chapter-specific JS inline. No external scripts, fonts or CDNs: pages must work offline.

## Process
1. **Plan and research.** Before writing, list every factual claim, number and code excerpt you intend to use. Check each one in a primary source by reading the file and noting `path:lines`. Drop or flag anything you cannot verify. Where sources disagree, say so on the page and name both sources.
2. **Write the page** at the file path given in your task.
   - **3Blue1Brown style.** Open with the concrete question. Build intuition with pictures and small numbers before giving definitions. Name a concept only after the reader feels the need for it. Use analogies, then check where each analogy breaks. End with a recap.
   - **At least two interactive animations**, and three or more is better. One must be the outline's "aha" animation. Use canvas or SVG, play/step/restart controls (`TT.stepper`) or sliders, and narration text for each step. They must be correct: the numbers, sizes and orderings shown must match the hardware and the code. Keep the drawing logic simple and robust.
   - **Code examples:** real excerpts from tt-metal or ttsim, with a `.filename` line in the form `repo/path:lines @commit` above each `<pre>`. Keep excerpts short and explain each line that matters. Escape `<`, `>` and `&` inside `<pre>`.
   - **Simulator output:** if the page shows output from running something, you must have run it yourself in this orb (see "Running things" below) and pasted the real output, trimmed. Never invent output.
   - **Callouts:** `.callout.key`, `.callout.gpu` (comparison with a GPU), `.callout.warn` (traps), and `.callout.sim` (link to the matching Series B lab, `../sim-labs/bNN-*.html`, using the filenames in OUTLINE.md).
   - **Check yourself:** 4–6 quiz questions using `<details class="quiz">`, with reasoning in each answer.
   - **Further reading:** 6–12 links, each with one line on what it is. Prefer papers, the official docs (docs.tenstorrent.com), tt-isa-documentation, tech_reports on GitHub, Hot Chips slides and good blogs. Only include links you have verified resolve, using read_web_page or curl. The research notes contain many verified links.
   - **Length:** substantial. Roughly 2500–5000 words of prose plus figures. Depth matters more than breadth.
3. **Self-check.**
   - Open the page in a headless browser. Run `agent-browser skills get core --full` for usage. Use `agent-browser open file:///...`, then `agent-browser set viewport 1280 900 2`. Confirm there are no JS errors (check the console), take screenshots of each animation after pressing Play or Step, and inspect them with view_media.
   - Fix anything that renders wrong: overlapping text, figures off-canvas, or animations that do not advance.
   - Save screenshots to `/tmp/shots-<chapter>/`, not inside the repo.

## Running things in the simulator
```bash
export LAB_HOME=$HOME/ttlab/<your-chapter-id>   # unique per agent; never share
source /home/user/workspace/tenstorrent-from-zero/sim-labs/workbench/env.sh wh   # or bh
$EX/metal_example_add_2_integers_in_riscv        # prebuilt examples live in $EX
```
- **Kernel overlay:** put a modified copy of a kernel at `$LAB_HOME/kernels/<same relative path the example uses>` and run from `$LAB_HOME/run`. You never need to rebuild tt-metal.
- **Custom host programs:** `sim-labs/workbench/build_host.sh my.cpp` builds one in about 6 s. See `sim-labs/workbench/host/mm_fid.cpp`, a matmul with selectable fidelity and seed.
- **Instrumented simulator:** `sim-labs/workbench/build_ttsim.sh <name> wh` makes an editable copy of ttsim and builds it in about 4 s. It prints the `export TT_METAL_SIMULATOR=...` line to use.
- **Do not** modify or rebuild anything under `/home/user/workspace/refs/`, and do not run examples with cwd set to `TT_METAL_HOME`.
- Known facts are in `research/06-feasibility.md`. ttsim is functional and bit-exact, not a timing model, so clock counts are not performance. The device profiler yields no data. Only `wh_x2` with a 2x1 mesh works for multichip.

## Final message to return
- The path written and its word count.
- The list of animations and what each one shows.
- Claims you could not verify, and how you handled them.
- Commands you ran in the simulator, with their key outputs.
- Screenshot paths.
- Any open issues.
