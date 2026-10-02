---
title: "Kernel class: dense compute near roofline (large pure GEMM/conv)"
type: concept
tags: [kernel-class, gemm, roofline, pallas, briefs-tier]
created: 2026-07-22
---

**Route here when** K1 shows a single large dense contraction already near its compute roofline, with no epilogue chain (else route to [gemm-conv-epilogue](gemm-conv-epilogue.md)).

- XLA lowers these near-optimally. **A hand kernel rarely wins**; the levers are a resource budget (scoped-vmem probe, BRIEFS §7 — optima can sit BELOW the default; response may be a saturating step) or nothing.
- **Refute with the envelope as evidence** (compute floor + operand-feed floor + HBM floor, BRIEFS §2): the bound is confirmed only by a parity-PASSING candidate that still loses (program.md Stop rule).
- **The exclusion-chain method (proving a local optimum)**: structure refutations so each eliminates a resource hypothesis with mechanism evidence — restrict operand A's traffic (~0% wall change ⇒ A wasn't binding) → cut MXU issue ops (wall-flat ⇒ not MXU-bound) → attack the remaining unit (the lever losing to its own overhead ⇒ the incumbent is locally optimal). The product is a reusable PROOF of why the frontier is where it is, not just a pile of losses.
- **Independent-convergence stop signal** (BRIEFS §5): several structurally different implementations piling at the same fraction of roofline ⇒ the residual is substrate; file the ceiling and stop.

## Levers (verified)

- **refute**: XLA dot_general is already near compute roofline
- **custom-grid-block-matmul**: custom Pallas block matrix multiply kernel
