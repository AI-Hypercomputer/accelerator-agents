---
title: "Qwen3 jax — manual per-iteration weight prefetch in the scan body (MaxText-style)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "up to ~620 ms/step (−11%) at bs4/seq8192 — the full exposed-AG-wait bucket"
confidence: medium
effort: L
origin: 2026-06-12-v031-ag-pipeline-flags
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — manual weight prefetch in the scan body

*Stub — the L-effort route to the 624 ms async-done bucket, opened when the
flag-level path (v030/v031/v032) is exhausted.*

*Hypothesis*: Restructuring the scan body so layer N+1's FSDP weight shards
are explicitly gathered (e.g. carried-ahead slice + `jax.lax` async dispatch
or double-buffered carry of gathered weights) while layer N computes —
replicating what MaxText's decoder achieves with XLA's pipeliner on their
loop structure — cuts the exposed all-gather wait (624 ms at bs4, 74% of
the MaxText gap) to near zero, measurable as ≥ +8% tok/s/chip.

*Rationale*: [gap analysis](../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)
— MaxText exposes ~9 ms of AG wait at the same shape with identical RS
bytes. Flag attempts: `while_loop_double_buffering` (absent, v030),
`collective_pipeliner`+`ici_ag_pipelining` (−1.8%, v031),
`pipelined_loop_unrolling` (v032). Cost: doubles the gathered-weights
working set (~+2 layer-weights of HBM — fits in bs4's 7.5 GiB margin).

*Falsification criterion*: prefetch lands but async-done stays > 300 ms or
step regresses (HBM pressure / scheduler interference) → refuted; the gap
is then documented as structural to GSPMD-in-scan on this stack.

## See also

- [Gap analysis](../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)
- [v031 (pipeliner flags regress)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v031-ag-pipeline-flags.md)
