<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# Pipeline Stage Balancing

## Prerequisites

- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md)
- [Collective Operations](11-collective-operations.md)

## Leads to

- [Experiment Protocol & Halt Rules](25-experiment-protocol-and-halt-rules.md)

## Used by agents

- [DeepResearch](../../agents/deep_research.md) — designs balanced stage partitions
- [MaxTile](../../agents/max_tile.md) — tunes microbatch sizes for balance

## What it is

Pipeline parallelism splits a model into sequential stages, each running on different chips. Pipeline stage balancing ensures all stages take approximately the same time, minimizing "pipeline bubbles" — idle time where chips wait for upstream stages to produce data or downstream stages to consume results.

A pipeline with N stages and microbatch size M has a theoretical bubble fraction of (N-1)/(N-1+M). With 8 stages and 16 microbatches: bubble = 7/23 ≈ 30%. If stages are unbalanced (some take 2x longer), actual bubbles are worse because fast stages idle waiting for slow ones.

Balancing strategies:
1. **Compute balancing**: partition layers so each stage has equal FLOP count
2. **Memory balancing**: partition so each stage has equal activation memory (avoiding recomputation on memory-constrained stages)
3. **Communication-aware balancing**: account for collective costs at stage boundaries
4. **Dynamic balancing**: adjust partition points based on profiled per-stage times

Perfect balance is often impossible — the goal is to minimize max-stage-time, which determines overall throughput.

## Why it matters for MaxPerf

Method M6 (Pipeline Stage Balancing) targets this directly. When xprof shows unequal stage times, DeepResearch analyzes per-layer costs and proposes new partition boundaries. MaxTile adjusts microbatch sizes to find the optimal bubble/memory tradeoff. Even a 5% reduction in max-stage-time translates directly to 5% throughput improvement across the entire pipeline.

## Worked example

A 96-layer model split into 8 pipeline stages (12 layers each). Profiling shows:
- Stages 1-6: 10.2 ms each (standard transformer layers)
- Stage 7: 13.8 ms (layers with larger MLP expansion ratio)
- Stage 8: 8.1 ms (final layers + output head)

Max stage = 13.8 ms → throughput limited by stage 7. Pipeline bubble waste: stages 1-6 idle 3.6 ms each, stage 8 idles 5.7 ms.

Rebalance: move 2 layers from stage 7 to stage 8. New times:
- Stages 1-6: 10.2 ms
- Stage 7: 10.5 ms (10 layers)
- Stage 8: 11.3 ms (14 layers)

Max stage drops from 13.8 → 11.3 ms. Throughput improvement: 13.8/11.3 = 22%.

## See also

- [Roofline Model & Bottleneck Analysis](14-roofline-model-and-bottleneck-analysis.md) — per-stage bottleneck classification
- [Collective Operations](11-collective-operations.md) — communication costs at stage boundaries
- [Payload Minimization](13-payload-minimization.md) — reducing inter-stage transfer cost
- [Diagnostic Vector](22-diagnostic-vector.md) — representing stage imbalance in diagnostics
