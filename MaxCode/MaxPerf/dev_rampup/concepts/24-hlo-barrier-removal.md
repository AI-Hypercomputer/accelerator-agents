<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# HLO Barrier Removal

## Prerequisites

- [XLA Compilation Passes](08-xla-compilation-passes.md)
- [DMA Idle Time](03-dma-idle-time.md)
- [Graph Rewrite Patterns](23-graph-rewrite-patterns.md)
- [Compile-Time Specialization Ladder](18-compile-time-specialization-ladder.md)

## Leads to

- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — primary executor of barrier removal

## What it is

HLO barriers are control dependencies in the HLO graph that force one operation to complete before another begins, even when there's no data dependency between them. XLA inserts barriers conservatively to ensure correctness (e.g., preventing a write from racing with a read of the same buffer). However, many barriers are overly conservative — they prevent reordering that would be safe and would enable DMA/compute overlap.

Barrier removal identifies and eliminates unnecessary control dependencies. The process:
1. Identify control-dep edges in the scheduled HLO graph
2. For each edge, determine why it was inserted (buffer aliasing? collective ordering? side-effect serialization?)
3. Check if the constraint is actually necessary: do the ops truly share a buffer? Is the ordering requirement real?
4. Remove barriers that are provably unnecessary
5. Re-run the scheduler — it can now reorder ops to overlap DMA with compute

Categories of removable barriers:
- **Stale buffer aliases**: buffer assignment changed since the barrier was inserted, making it unnecessary
- **False sharing**: two ops access different regions of the same allocation
- **Collective over-serialization**: all-reduces that don't conflict but are serialized by default

## Why it matters for MaxPerf

Method M4 (Execution Graph Unblocking) targets barrier removal as its primary technique. When DMA idle time is high (from the diagnostic vector's `dma_idle_fraction`), unnecessary barriers are the likely cause — the scheduler can't overlap transfers because barriers force sequential execution. Removing even a few key barriers can enable the scheduler to produce a significantly better timeline, reclaiming the DMA idle gaps without any change to the computation itself.

## Worked example

xprof shows a pattern: `matmul(tile_N)` → 0.3ms DMA idle → `matmul(tile_N+1)`. The scheduled HLO reveals:
```
%matmul.1 = dot(...)                    // compute tile N
%copy.1 = copy-start(...)               // prefetch tile N+1 from HBM
ctrl: %copy.1 depends on %matmul.1      // BARRIER: why?
```

Investigation: the barrier exists because `%matmul.1` and `%copy.1` were assigned the same VMEM buffer in an earlier compilation pass. But after buffer reallocation (a later pass), they use different buffers. The barrier is stale.

AutoRefactor removes the control dependency. XLA's scheduler now overlaps the DMA prefetch with the matmul:
```
Timeline before: [matmul 0.5ms][idle 0.3ms][matmul 0.5ms]
Timeline after:  [matmul 0.5ms + DMA overlap][matmul 0.5ms + DMA overlap]
```

Time per pair: 1.3 ms → 1.0 ms (23% improvement for this loop body).

## See also

- [DMA Idle Time](03-dma-idle-time.md) — the symptom that barriers cause
- [XLA Compilation Passes](08-xla-compilation-passes.md) — where barriers are inserted and removed
- [Compile-Time Specialization Ladder](18-compile-time-specialization-ladder.md) — barrier removal as an aggressive rung
- [Graph Rewrite Patterns](23-graph-rewrite-patterns.md) — pattern-based identification of stale barriers
