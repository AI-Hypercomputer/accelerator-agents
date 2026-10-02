---
title: "Kernel class: streaming reduction (norms, cross-entropy, standalone softmax)"
type: concept
tags: [kernel-class, reduction, norm, cross-entropy, pallas, briefs-tier]
created: 2026-07-22
---

**Route here when** K1 shows a big reduction over a materialized axis without a dominating matmul producer (RMSNorm/LayerNorm, cross-entropy, standalone softmax). If a GEMM/conv produces the reduced tensor, route to [gemm-conv-epilogue](gemm-conv-epilogue.md) instead — the precondition test there decides.

- **Strategy**: tile the reduction axis, maintain running state in VMEM (running max/sum for softmax; mean/var for norms; logsumexp for CE); accumulate fp32. The intermediate never lands in HBM.
- **Cheap first rung before authoring a norm kernel**: XLA's default mean/variance lowering pays two reads; the `E[x²]−E[x]²` single-pass moment rewrite is near-free, needs no kernel (reassociates ⇒ tolerance-gated).
- **Check the overlap decomposition first** (BRIEFS §2): on compute-adjacent CE-class ops, eliminating GB of traffic that hides under a matmul buys ~6% — naive-wall vs compute-floor decides whether the class lever is worth authoring.
- **Cross-lane (last-dim) reductions route through the XLU** and cost markedly more than sublane reductions — layout choice is part of the design (index Hardware envelope).
- Parity: gate calibration + ULP-at-magnitude + non-degenerate probes (BRIEFS §4) — softmax/exp tails make max_rel meaningless near zero; grade max-abs at magnitude.
