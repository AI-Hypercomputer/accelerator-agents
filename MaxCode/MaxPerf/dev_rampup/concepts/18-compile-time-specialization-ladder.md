<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Compile-Time Specialization Ladder

## Prerequisites

- [XLA Compilation Passes](08-xla-compilation-passes.md)
- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md)

## Leads to

- [HLO Barrier Removal](24-hlo-barrier-removal.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — traverses the ladder systematically

## What it is

The compile-time specialization ladder is a methodology for systematically exploring XLA compiler configurations from most conservative (default flags, maximum safety) to most aggressive (experimental flags, maximum performance, higher risk of issues). Each "rung" of the ladder enables a set of related compiler optimizations.

Typical ladder rungs (conservative → aggressive):
1. **Baseline**: default XLA flags, no specialization
2. **Standard optimizations**: enable well-tested aggressive fusion, layout optimization
3. **TPU-specific**: enable TPU-targeted passes (megacore fusion, memory folding)
4. **Experimental fusion**: relax fusion heuristic thresholds, allow larger fused regions
5. **Scheduling aggression**: enable speculative DMA prefetch, reduce barrier conservatism
6. **Full specialization**: shape-specialized compilation, workload-specific codegen

Each rung is validated for: numeric equivalence, no OOM errors, and performance improvement over the previous rung. If a rung fails validation, its flags are individually bisected to find the problematic one.

## Why it matters for MaxPerf

Method M8 (Compile-Time Specialization Ladder) uses this structure to find the best compiler configuration without brute-force search of the flag space (which has combinatorial explosion). By ordering flags by risk level and testing in groups, AutoRefactor can find the optimal configuration in O(rungs × flags-per-rung) experiments rather than O(2^total_flags). The ladder also provides a rollback path: if an aggressive rung breaks production, drop to the previous validated rung.

## Worked example

AutoRefactor runs the ladder on a GPT-3 training step:

| Rung | Flags enabled | Step time | Valid? |
|------|--------------|-----------|--------|
| 1 (baseline) | defaults | 142 ms | yes |
| 2 (standard) | +aggressive_fusion, +layout_opt | 128 ms | yes |
| 3 (TPU-specific) | +megacore, +memory_folding | 119 ms | yes |
| 4 (experimental) | +fusion_threshold=2x | 112 ms | yes |
| 5 (scheduling) | +speculative_prefetch, -conservative_barriers | 104 ms | yes |
| 6 (full) | +shape_specialization | 101 ms | **no** (NaN in layer 47) |

Result: rung 5 is optimal. Bisecting rung 6 reveals shape_specialization triggers a known XLA bug with dynamic shapes in the embedding layer. AutoRefactor reports rung 5 as the recommendation: 26.8% improvement over baseline.

## See also

- [XLA Compilation Passes](08-xla-compilation-passes.md) — the passes being configured
- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md) — fusion-related flags
- [HLO Barrier Removal](24-hlo-barrier-removal.md) — specific technique enabled at rung 5
- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md) — validation methodology
