---
title: "MaxText vs jax lane — xplane bucket diff at bs3/seq8192 (gap decomposition)"
type: analysis
tags: [qwen3-cc5, jax, maxtext, v6e-8, gap-analysis, collectives, profile]
created: 2026-06-12
updated: 2026-06-12
---

# MaxText vs jax lane — bucket diff @ bs3/seq8192

Profile-analyzer comparison of the MaxText reference trace
([mt-v001](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md),
3,535 ms/step) against our shape-identical
[v026](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v026-mt-flags-bs3.md)
(4,142 ms/step). Answers "where do their 607 ms live" after the easy
explanations (batch, flags, kernel blocks) were eliminated by v025–v028.

## Gap decomposition (per chip per step)

| Cause | v026 − MaxText | % of 607 ms | Actionability |
|---|---|---|---|
| **Collective async-wait (FSDP all-gather exposure)** | **+451 ms** | **74.3%** | **High — scheduling/structural, not algorithmic** |
| Conv fusion (lower MXU efficiency: 53.4% vs 61.9%) | +177 ms | 29.2% | Medium — partly feedback from the stall itself |
| Loop fusion (our chunked-CE while-loop overhead) | +70 ms | 11.5% | Low at this op-point (XLA-native CE does not fit: per-chip f32 logits ≈ 20 GB at bs4) |
| Data formatting | +15 ms | 2.5% | Low |
| RS fusion | +2 ms | 0.3% | Noise |
| Splash attention (**MaxText is SLOWER here**) | −89 ms | −14.7% | Our advantage — no action |
| Other | −19 ms | −3.1% | — |

## Key findings

1. **All-gather overlap is 51× better in MaxText**: they expose ~9 ms of
   AG wait per step (AG runs as a 50.5 ms *active overlapped* kernel); we
   expose **460 ms** of async-done blocking. Both stacks move the same RS
   bytes (143 vs 145 ms) — the difference is purely whether the *pre-matmul
   weight all-gather of layer N+1 is launched under layer N's compute*
   inside the scanned decoder loop.
2. **Splash invocation pattern**: MaxText runs 3 kernel calls/layer/step
   (1 fwd + 1 residuals-reusing fwd-dq + 1 dkv = 108/chip/step); we run 6
   (4 fwd + 2 dkv = 216). Despite double the invocations, our splash bucket
   is 89 ms *cheaper* (1,088 vs 1,177 ms) — their residuals-save overhead
   outweighs our remat re-runs. The attention kernel is NOT a porting
   target (consistent with v028's refutation).
3. **MXU efficiency** 61.9% vs 53.4% (+177 ms): partly weight-layout
   (their conv HBM-BW util 26.9% vs our 49.1% — we are more
   bandwidth-bound on the same matmuls), partly idle-time feedback from
   the AG stall. Re-measure after fixing the stall before treating layout
   as an independent lever.
4. Both pay QK-norm equally; no hidden algorithmic shortcut in their trace.

## The actionable lever

Ensure layer N+1's weight all-gather (`async-collective-start`) is emitted
*inside* the scan body before layer N's compute barrier, so XLA's
collective pipeliner can overlap across iterations. v027's HLO shows only
17 `all-gather-start/done` pairs in the optimized module with no
cross-iteration pipelining markers — the AGs triggered by the sharded
stacked weights inside the `while` body appear scheduled just-in-time.
Tested as [v030](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v030-ag-overlap.md):
(a) `--xla_tpu_enable_while_loop_double_buffering=true` (the XLA knob for
buffering loop collectives across iterations), and (b) the unrolled-loop
fallback at bs4 (the unrolled decoder historically exposed only ~4.5%
collectives — v015 — because XLA pipelines freely without a loop boundary;
v024's liveness OOM may no longer bind under the v026 flag stack that cut
HBM peak by 7.7 GiB).

## Sources

- MaxText trace: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-maxtext-v001-base/mt-v001-base-b/tensorboard/plugins/profile/2026_06_12_06_18_19/`
- jax v026 trace: `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v026-mtfl/plugins/profile/2026_06_12_07_21_13/`
- jax v027 HLO (AG audit): `gs://<your-bucket>/autoresearch/qwen3_cc5/2026-06-12-qwen3-jax-v027-bs4/hlo/`

## See also

- [mt-v001 (the reference run)](../experiments/qwen3_cc5_autoresearch_optimization/maxtext/experiments/2026-06-12-mt-v001-base.md)
- [v026](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v026-mt-flags-bs3.md), [v027 (frontier)](../experiments/qwen3_cc5_autoresearch_optimization/jax/experiments/2026-06-12-v027-bs4-mtfl.md)
- [Model page](../models/qwen3-cc5-jax.md)
- [Hypothesis: scan AG overlap](../hypotheses/qwen3-jax-scan-ag-overlap.md)
