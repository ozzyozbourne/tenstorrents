# Brief for chapter verifiers

You are the independent fact-checker for one page of "Tenstorrent From Zero". Another agent wrote the page. Assume it contains mistakes and go looking for them. Do not trust its claims or its self-report.

## Sources of truth
Primary sources, at the pinned commits:
- `/home/user/workspace/refs/tt-metal` (7aad8aac)
- `/home/user/workspace/refs/ttsim` (b3620244)
- `/tmp/tt-isa-documentation`. If it is missing: `git clone --depth 1 https://github.com/tenstorrent/tt-isa-documentation /tmp/tt-isa-documentation`.
- Official spec pages and papers on the web.

The notes in `/home/user/workspace/tenstorrent-from-zero/research/*.md` are secondary and sometimes wrong. `research/06-feasibility.md` lists what was actually run in this orb.

## Checklist
1. **Numbers and facts.** Check every number and factual claim on the page against a primary source, and open the cited file and lines. Pay most attention to:
   - hardware numbers (sizes, counts, clocks, bandwidths, TFLOPS)
   - API names and signatures
   - semantics (what an instruction or call does)
   - directions and orderings (NoC directions, face order, SrcA vs SrcB)
2. **Code excerpts.** Each must match the cited file and lines, apart from trimming that is marked as such. The `@commit` and path must be correct.
3. **Simulator output.** Re-run at least the key commands and confirm the output appears in this orb.
   - Set `export LAB_HOME=$HOME/ttlab/verify-<page>`, then `source /home/user/workspace/tenstorrent-from-zero/sim-labs/workbench/env.sh wh`.
   - Values that legitimately differ, such as timestamps and wall times, are fine. Invented output is not.
4. **Animations.** Read the JS and confirm that what each animation draws is correct: tile counts, bit slicing, routing paths, pointer arithmetic. Then open the page with agent-browser (`agent-browser skills get core --full` for usage):
   - `set viewport 1280 900 2`
   - step through every figure
   - check the console for errors
   - inspect screenshots with view_media, saved to `/tmp/verify-<page>/`
   - look for overlapping text, clipping, and animations that don't advance
5. **Links.** Check that each external link resolves and is what the page says it is. Internal links must point to filenames listed in OUTLINE.md; links to pages not yet written are fine.
6. **Pedagogy.** Watch for forward references to concepts not yet taught (see OUTLINE.md ordering), and claims stated with more certainty than the evidence supports.

## Fix, then report
Fix errors directly in the page with minimal edits that keep the author's voice. When a fact is uncertain, soften it or add a sourced caveat rather than deleting it.

Final message:
- Each issue found: what was wrong, the evidence, and the fix.
- Anything you could not resolve.
- The commands you re-ran.
- Screenshot paths.
- A verdict: PASS (after your fixes) or NEEDS WORK, with the reason.
