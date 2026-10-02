---
title: "Kernel class: state-carry / chunked scan (SSM, retention, linear attention)"
type: concept
tags: [kernel-class, ssm, scan, retention, linear-attention, pallas, briefs-tier]
created: 2026-07-22
---

**Route here when** the op is a linear recurrence / chunked scan: running state carried across sequence chunks (Mamba-class SSD, retention, linear attention, any decay-weighted accumulation).

## Levers (verified)
*(section reviewed & adopted by master 2026-07-24; initially worker-authored — p6 ledger D20)*
- **Chunk sequence and carry state across steps**: Chunk the sequence; carry state in VMEM across grid steps (grid-derived static indices for the state walk).
- **Fold decay algebra into matmul operands**: Fold per-position decay/gate algebra into matmul operands where the algebra permits (a rank-1 decay fold into the operands).
- **Chunk-seam parity probe**: Measure per-chunk max error separately — seam drift shows as error growing with chunk index; uniform per-chunk error proves clean state carry.
- **Non-broadcast vector reshape (..., 1, C)**: Never broadcast small per-position vectors to lane width before pallas_call; reshape to (..., 1, C) with a matching blockspec and slice in-kernel.
- **bfloat16 MXU GEMM with fp32 accumulator tail**: Precision/factorization placement — MXU bfloat16 GEMM with minimal fp32 accumulator tail.
- **Pallas custom call grid kernel over chunks**: Fuse intra-chunk score calculation, decay mask, and value contraction into a grid pallas_call kernel.
- **Block-tiled VPU scan and inter-chunk projection**: Pipeline inter-chunk state carries and output projection in VMEM tile registers.
- **Tile-sensitivity re-derivation**: Tile-sensitivity does NOT transfer across structures within a family — re-derive after any layout change.
