---
title: "Qwen3 jax — overlap FSDP weight all-gathers across scan iterations (the 74% gap lever)"
type: hypothesis
model: qwen3-cc5-jax
variants: ["8B/v6e-8"]
status: refuted
expected_gain: "up to ~450 ms/step (−8%) at bs4/seq8192 — 74% of the residual MaxText gap"
confidence: medium
effort: S (flag probe) → M (structural)
origin: 2026-06-12-maxtext-vs-jax-bucket-diff
created: 2026-06-12
updated: 2026-06-12
---

# Qwen3 jax — scan all-gather overlap

> **REFUTED at the flag level (2026-06-12)** — three-probe ledger:
> [v030](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v030-ag-overlap.md)
> `while_loop_double_buffering` absent from this libtpu (+ unrolled
> fallback = 67.67 G OOM);
> [v031](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v031-ag-pipeline-flags.md)
> `collective_pipeliner`+`ici_ag_pipelining` fire-and-hurt (−1.8%; new
> 138 ms sync-AG bucket; AG-fusion helper flag retired);
> [v032](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v032-loop-unroll.md)
> `pipelined_loop_unrolling` silent no-op (HLO unchanged, step flat).
> The 624 ms exposed-AG bucket is unreachable by flags in this build. The
> surviving route is
> [manual weight prefetch](qwen3-jax-manual-weight-prefetch.md) (L).

*Hypothesis*: Getting layer N+1's FSDP weight all-gather launched under
layer N's compute inside the scanned decoder (via
`--xla_tpu_enable_while_loop_double_buffering=true`, or failing that, the
unrolled loop at bs4 under the v026 flag stack) cuts the exposed async-done
wait (624 ms at bs4 / 460 ms at bs3) toward MaxText's ~9 ms — measurable as
> +2% tok/s/chip over the 6,040 frontier.

*Rationale*: [bucket-diff analysis](../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)
attributes 74% of the MaxText gap to AG exposure; both stacks move the same
collective bytes. Our unrolled decoder historically exposed only ~4.5%
collectives (v015) — the regression came with scan (the while-loop boundary
blocks XLA's lookahead), but scan is required by offload (v024). The flag
is the cheap reconciliation; the unrolled fallback re-tests v024's liveness
OOM under the new flag stack that cut HBM peak by 7.7 GiB.

*Falsification criterion*: both arms ≤ 6,040 + 2% (6,161) → refuted; the
AG exposure is then structural to scan+GSPMD on this stack and the
remaining path is a manual weight-prefetch port (L) or accepting the gap.

## See also

- [Gap analysis (origin)](../analyses/2026-06-12-maxtext-vs-jax-bucket-diff.md)
- [v030 (the test)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v030-ag-overlap.md)
