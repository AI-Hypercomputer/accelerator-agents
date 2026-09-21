<!-- disableFinding(LINK_RELATIVE_G3DOC) -->
<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(LIST_NO_LINE) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->
<!-- disableFinding(WHITESPACE_LINES) -->
<!-- disableFinding(SPACES) -->
<!-- disableFinding(HTML_OPEN) -->
<!-- disableFinding(HTML_BROKEN) -->

---
title: "Rematerialization"
type: concept
tags: [stub, optimization]
created: 2026-04-22
updated: 2026-04-22
sources: 2
---

Recomputing activations during the backward pass to save HBM; selective AC trades ~2.7% extra compute for ~70% activation memory.

*Stub — expand when a hypothesis or experiment needs this concept in depth.*

## See also

- [HBM (High-Bandwidth Memory)](hbm.md)
- [Host Offload](host-offload.md)
- [Scan Over Layers](scan-over-layers.md)
- [Training Memory Budget](training-memory-budget.md)
- [HLO Op](hlo-op.md)

## Sources

- [xprof-mcp TPU optimization guide](../sources/2026-xprof-mcp-tpu-optimization.md) — `raw/code/xprof-mcp/docs/TPU_OPTIMIZATION.md`
- [xprof HLO Op Stats](../sources/2026-xprof-hlo-op-stats.md) — `raw/code/xprof/docs/hlo_op_stats.md`
- [Ultra-Scale Playbook](../sources/2025-ultrascale-playbook.md) — selective activation recomputation (Korthikanti et al.): 70% activation-memory reduction at 2.7% compute cost
