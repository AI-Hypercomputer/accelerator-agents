---
title: "Expert Parallelism (MoE)"
type: concept
tags: [stub, parallelism, moe]
created: 2026-04-22
updated: 2026-04-22
sources: 1
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->


Parallelism axis specific to Mixture-of-Experts models: expert weights are partitioned across devices; tokens are routed to their top-k experts via an `all_to_all` dispatch, computed locally, then returned via a second `all_to_all`. Orthogonal to DP/TP/PP/CP — stacks on top. The compute kernel on TPU is [tokamax `ragged_dot`](../codebases/tokamax.md) (`pallas_mosaic_tpu`); routing is `jax.lax.all_to_all` over an EP mesh axis. Bandwidth-rich over ICI, cliff at DCN — EP should ride an ICI axis.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [Ragged Dot](ragged-dot.md)
- [Sharding (GSPMD)](sharding.md)
- [Async Collectives](async-collectives.md)
- [ICI (Inter-Chip Interconnect)](ici.md)
- [DCN](dcn.md)

## Sources

- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — Section 7 (Expert Parallelism)
