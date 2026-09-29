---
title: "Llama 3 8B torchax — converged-stack bottleneck breakdown (xprof)"
type: observation
tags: [llama3, torchax, profile, bottleneck, mxu, pallas, splash, tokamax, ce]
created: 2026-04-26
updated: 2026-04-26
---

xprof breakdown of the **2026-04-26 program-target frontier** stack (scan +
splash + tokamax CE w/ autotune + fp32 master + bf16 compute + VMEM=98 KiB
+ `nothing_saveable` scan remat) at **bs=3 seq=8192**, 49,627 tok/s,
**34.8 % MFU**, 51.9 % MXU utilization. Profile run: `exp 61b`.

## What was observed

Per-chip per-step time (averaged over 7 measured steps, all 8 chips):

| Category | ms / step | % | HBM-BW util | Note |
|----------|----------:|--:|------------:|------|
| Convolution fusion (QKV/O/FFN matmuls) | 1,821 | 46.0 % | 0.54 | productive matmul, headroom = MXU<->HBM tile pattern |
| Custom-call (Pallas) — total | 1,265 | 32.0 % | — | |
| ↳ `splash_mha_fwd_residuals` | 543 | 13.7 % | — | attention forward |
| ↳ `splash_mha_dkv_no_residuals` | 423 | 10.7 % | — | attention backward (dQ/dK/dV) |
| ↳ `linear_softmax_cross_entropy_loss_bwd_pallas_mosaic_tpu` | 263 | 6.6 % | — | tokamax CE bwd |
| ↳ `linear_softmax_cross_entropy_loss_fwd_pallas_mosaic_tpu` | 36 | 0.9 % | — | tokamax CE fwd |
| Loop fusion (RMSNorm, silu, residual, mul) | 537 | 13.6 % | 0.77 | already HBM-saturated |
| Data formatting (copy / bitcast / reshape) | 144 | 3.6 % | 0.76 | already HBM-saturated |
| Misc | ~210 | 5 % | — | |
| **Total** | **3,964** | **100 %** | | |

MXU utilization 51.9 % vs MFU 34.8 % ⇒ **48 % of step time is non-matmul**;
within the matmul time, achieved/peak ≈ 0.67 (matmul-time MFU when
matmuls run).

## How to see it

Profile artifacts in GCS:

```
gs://<your-bucket>/autoresearch/profiles/llama3-8b-exp61b-profile-bs3/
└── plugins/profile/2026_04_26_03_36_18/
    ├── gke-tpu-8861e1ea-jqvs.{trace.json.gz, xplane.pb}
    └── gke-tpu-8861e1ea-nbcm.{trace.json.gz, xplane.pb}
```

Local browse with the canonical xprof OSS server:

```
xprof --logdir=gs://<your-bucket>/autoresearch/profiles \
      --port=<free-port>
```

Then `mcp__xprof__list_runs`, `get_overview`, `get_op_profile` etc. against
that endpoint.

## Implications

- **Tokamax CE bwd is asymmetrically expensive** (263 ms = 7 × the 36 ms
  fwd, 6.6 % of step). The mosaic-TPU bwd recomputes logit blocks in VMEM
  and runs at only ~21 % MXU efficiency by the back-of-envelope
  `2*B*L*H*V` accounting. Two hypotheses worth filing:
  - Test `implementation="chunked_xla"` instead of `"mosaic_tpu"` —
    materializes a chunk of logits at a time but uses XLA matmuls (which
    may schedule better than the Pallas implementation). Risk:
    chunked_xla needs activation memory that may push us out of bs=3
    fitting envelope.
  - Custom CE kernel with persistent logit residuals on bwd (save
    `(B*L_block, V_block)` blocks computed in fwd to skip recompute). Memory
    cost ≈ block-size × occurrences ≈ tens of MiB → manageable.

- **Splash dkv bwd (10.7 %, 423 ms)** is the single biggest custom-call
  cost. This is the attention backward kernel; replacing with a different
  Pallas attention bwd is a deep rewrite. MaxText's reference at 44.6 %
  MFU likely has a more efficient dkv kernel — pull in their attention
  bwd impl for comparison.

- **Loop fusion at 0.77 HBM-BW util** means RMSNorm, silu, residual-add,
  etc. are bandwidth-bound on HBM and already running near hardware
  limit. No throughput to recover from this layer of the stack — except
  by reducing how much vector work is done (fusing into matmul prologues
  / epilogues). Fusion is XLA's job; we have limited leverage.

- **Convolution fusion HBM-BW util is 0.54** — there is matmul-pattern
  headroom. The 46 % of step in matmuls is already > 50 % MXU efficiency,
  but if XLA could pick better tile shapes, could climb closer to 70 %.
  Not directly tunable from the trainer; would need MaxText-style XLA
  flags that match the tiles to v6e's 256×256 MXU.

## Updated breakdown — 2026-04-26 latest frontier (exp 79 profile, on the exp 74b stack)

After all the day's wins (chunked_xla CE + tokamax-splash + base2/fuse_recip/mlc), step time dropped 3,964 → 3,748 ms / chip; MFU climbed 34.8 % → 36.8 %; MXU utilization 51.9 % → 55.0 %.

| Category | ms/step (per chip) | %  | HBM-BW util | Notes |
|----------|-------------------:|----|------------:|-------|
| Convolution fusion (matmul) | 2,040 | **54.4 %** | 0.54 | up from 46.0 % — relatively bigger share now that other costs collapsed |
| Custom-call (splash MHA) total | 816 | 21.8 % | 0.03 | down from 24.4 % absolute (~150 ms savings) |
| ↳ splash dkv bwd | 430 | 11.5 % | — | |
| ↳ splash fwd | 386 | 10.3 % | — | |
| ↳ tokamax CE (fwd+bwd) | ~0 | ~0 % | — | **collapsed from 7.5 %** by chunked_xla impl |
| Loop fusion (RMSNorm/silu/residual/mul) | 570 | 15.2 % | 0.75 | already saturated |
| Misc / data formatting | ~322 | ~8.6 % | — | |
| **Total** | **3,748** | **100 %** | | |

**During matmul, MXU efficiency = 73.9 %** (369 TFLOPs/sec/chip ÷ 0.544 share ÷ 918 peak). Matmul is well-utilized when it runs; the gap to MaxText 44.6 % MFU is *time-share*, not matmul-tile efficiency.

To climb past 36.8 %, would need: (a) a custom Pallas matmul prologue that fuses the bf16 weight cast into the matmul (skip the bf16 materialization HBM round-trip), or (b) a faster splash dkv kernel (the upstream/tokamax variant we use is already configured optimally for this shape — see exp 67/68/80 sweep refutations and exp 74 mlc win), or (c) MaxText-style layer fusion patterns that compress the loop-fusion 15.2 % share.

## Seen in

- [exp 55b — fp32 master + tokamax CE program-target best](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md) (this is the converged stack)
- [exp 56](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md#follow-ups-tested-post-acceptance-all-2026-04-26) — autotune variant (essentially the same trace; this profile was captured from exp 61b which is autotune-enabled).
- exp 61b — short profile capture run (9 steps, 7 measured, profile_step=5).

## See also

- [pallas-kernel-directory § cross-entropy](../analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) — tokamax CE kernel surveyed.
- [exp 55b page](../experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-26-exp55b-fp32-master-tokamax-ce-bs3-seq8k-accepted.md) for the stack composition this trace measured.

## Sources

- `gs://<your-bucket>/autoresearch/profiles/llama3-8b-exp61b-profile-bs3/` — xplane.pb + trace.json.gz from both v6e-8 worker hosts.
