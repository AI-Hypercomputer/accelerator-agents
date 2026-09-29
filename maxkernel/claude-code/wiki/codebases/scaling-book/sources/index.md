---
title: "Source: index.md"
type: source-page
doc: index.md
updated: 2026-07-03
---
# index.md — Part 0: Introduction and outline

The book's front page. States the central thesis — "A 20% win on benchmarks is irrelevant if it
comes at a 20% cost to roofline efficiency" and that "strong scaling" (linear throughput increase
with added chips) is the goal, bounded by when added communication exceeds the reduction in compute
time ([why-should-you-care](src:index.md#why-should-you-care)). Gives the full chapter outline
mapping each part (rooflines → TPUs → sharding → transformer math → training → applied training →
inference → applied inference → profiling → JAX → GPUs) to the question it answers
([high-level-outline](src:index.md#high-level-outline)).

## Feeds
- No dedicated topic page — this is the book's own table of contents/thesis statement; its content
  is distributed across every other topic page's cross-references. Represented by this source page.
