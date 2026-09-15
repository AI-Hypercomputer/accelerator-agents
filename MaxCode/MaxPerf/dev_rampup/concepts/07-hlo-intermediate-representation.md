<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

# HLO Intermediate Representation

## Prerequisites

- [JAX Tracing Model](06-jax-tracing-model.md)

## Leads to

- [XLA Compilation Passes](08-xla-compilation-passes.md)
- [Numeric Equivalence](10-numeric-equivalence.md)

## Used by agents

- [AutoRefactor](../../agents/auto_refactor.md) — reads and modifies HLO text for pass experiments
- [DeepResearch](../../agents/deep_research.md) — analyzes HLO structure for optimization opportunities
- [MaxShard](../../agents/max_shard.md) — applies graph rewrites at the HLO level

## What it is

HLO (High Level Operations) is XLA's intermediate representation — a directed acyclic graph of operations with explicit shapes, dtypes, and data flow. Each node is an operation (dot, add, reduce, custom-call, etc.) with typed inputs and outputs. The graph is organized into "computations" (analogous to functions) with a single entry point.

HLO is the primary interface between MaxPerf and the compiler. It is human-readable as text (HLO text format), can be serialized as protobuf, and can be dumped before/after each XLA pass. Key structural elements: instructions (ops), operands (data edges), control dependencies (ordering edges), metadata (source location, op name), and sharding annotations.

Understanding HLO means reading graphs like: `%dot.42 = f32[1024,4096] dot(%param.0, %param.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}` — a matrix multiply of two parameters producing a 1024x4096 f32 result.

## Why it matters for MaxPerf

HLO is the level at which MaxPerf agents reason about optimization. TPUDiagnoseAgent maps xprof ops back to HLO nodes. DeepResearch identifies graph patterns amenable to rewriting. AutoRefactor experiments with pass flags that transform HLO structure. MaxShard rewrites the graph to change sharding decisions. Every performance diagnosis starts with "what does the HLO look like?" and every optimization is validated by "does the new HLO produce the same outputs faster?"

## Worked example

An HLO dump shows:
```
%add.1 = f32[2048,8192] add(%dot.1, %broadcast.1)
%exp.1 = f32[2048,8192] exponential(%add.1)
%reduce.1 = f32[2048] reduce(%exp.1), dimensions={1}, to_apply=%sum
```
This is a softmax fragment: bias-add → exp → sum-reduce. These three ops are candidates for fusion into a single kernel (avoiding two HBM round-trips for the intermediates). If XLA hasn't fused them, AutoRefactor can investigate which pass failed and why.

## See also

- [JAX Tracing Model](06-jax-tracing-model.md) — how HLO is produced
- [XLA Compilation Passes](08-xla-compilation-passes.md) — transformations applied to HLO
- [XLA Fusion & Custom-Call Boundaries](09-xla-fusion-and-custom-call-boundaries.md) — how HLO ops get grouped
- [Graph Rewrite Patterns](23-graph-rewrite-patterns.md) — structural patterns in HLO for optimization
