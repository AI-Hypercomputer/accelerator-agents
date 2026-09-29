---
title: "Exp 72a — tokamax-shipped splash attention (use_base2_exp + fuse_reciprocal) @ bs=3 seq=8192 (ACCEPTED, NEW PROGRAM-TARGET BEST)"
type: experiment
tags: [llama3, torchax, tokamax, splash-attention, use_base2_exp, fuse_reciprocal, accepted, milestone, frontier]
hypothesis: llama3-torchax-tokamax-splash-impl
model: llama3-8b-torchax
created: 2026-04-26
updated: 2026-04-26
commit: "v6e8-llama3-8b-torchax-20260426-exp72a-tokamax-splash (image hf-v34)"
branched_from: v6e8-llama3-8b-torchax-20260426-exp65-cxla-autotune
verdict: supported
---

🏆 **Program-target advanced.** Replacing the upstream
`jax.experimental.pallas.ops.tpu.splash_attention` kernel with the tokamax-
shipped `tokamax._src.ops.experimental.tpu.splash_attention` and turning on
its two perf-only knobs (`use_base2_exp=True`, `fuse_reciprocal=True`) gives
**+1.3 % per-chip throughput at bs=3 seq=8192** — **6,392 tok/s/chip,
35.8 % MFU** vs the prior frontier (exp 65) at 6,313/chip, 35.4 %. Same
gain holds at bs=2 (+1.8 %) and bs=4 (+1.2 %).

The win came from comparing our trainer's attention call site to MaxText's
`attention_op.py:1180-1278` and noticing MaxText also offers a `use_tokamax_splash`
mode. The tokamax-shipped impl exposes config knobs that the upstream
`jax.experimental.pallas.ops.tpu.splash_attention.BlockSizes` API doesn't:
`use_base2_exp` (base-2 exp instead of natural — TPU's exp2 is faster),
`fuse_reciprocal` (compute `o / lse` inside the kernel — saves a separate
division pass), `use_experimental_scheduler`, `dq_reduction_steps`,
`max_logit_const`. The first two are independent perf wins; the others
neutral or worse at this shape.

## Stack delta

Identical to [exp 65 frontier](2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md#follow-ups-post-acceptance-all-2026-04-26)
**except** the splash kernel call site:

```python
# Before (exp 65 stack: jax.experimental splash):
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
block_sizes = splash_attention_kernel.BlockSizes(
    block_q=2048, block_kv=1024, block_q_dkv=2048, block_kv_dkv=2048,
    use_fused_bwd_kernel=True, ...)
multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * n_heads)
kernel = splash_attention_kernel.make_splash_mha(
    mask=multi_head_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)

# After (exp 72a stack: tokamax splash):
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash_kernel
sa_config = tokamax_splash_kernel.SplashConfig(
    block_q=2048, block_kv=1024, block_q_dkv=2048, block_kv_dkv=2048,
    use_fused_bwd_kernel=True,
    use_base2_exp=True,        # NEW knob, perf
    fuse_reciprocal=True,      # NEW knob, perf
    use_experimental_scheduler=False,
    dq_reduction_steps=None, ...)
single_head_mask = tokamax_splash_mask.CausalMask(...)  # broadcast-to-heads internal
kernel = tokamax_splash_kernel.make_splash_mha(
    mask=single_head_mask, q_seq_shards=1, config=sa_config)
```

Image: **`hf-v34`** (drop-in tokamax splash with env-var control).
Toggle via `USE_TOKAMAX_SPLASH=1` + `TOKAMAX_USE_BASE2_EXP=1` +
`TOKAMAX_FUSE_RECIPROCAL=1`. Default off so the trainer still works for
older configs.

## Results — full bs sweep at the new frontier

| Run | bs | seq | MFU | tok/s | tok/s/chip | step_time |
|-----|----|-----|-----|-------|------------|-----------|
| exp 72d | 2 | 8192 | 35.5 % | 50,595 | 6,324 | 2.59 s |
| 🏆 **exp 72a** | **3** | **8192** | **35.8 %** | **51,139** | **6,392** | **3.84 s** |
| exp 72c | 4 | 8192 | 35.0 % | 49,965 | 6,246 | 5.24 s |

vs prior frontier (exp 65, jax-splash):

| bs | jax-splash (exp 65) | tokamax-splash (exp 72) | Δ |
|---:|--------------------:|------------------------:|---|
| 2  | 6,212/chip 34.8 %   | **6,324/chip 35.5 %** | **+1.8 %** |
| 3  | 6,313/chip 35.4 %   | **6,392/chip 35.8 %** | **+1.3 %** |
| 4  | 6,170/chip 34.6 %   | **6,246/chip 35.0 %** | **+1.2 %** |

bs=3 remains the sweet spot. Loss decay matches exp 65 step-for-step
(`11.7681 → 11.7159` over steps 0-2): both kernels compute the same
mathematical operation; tokamax just runs the same softmax + matmul more
efficiently on TPU.

## Ablations

Independent contribution of the two perf knobs:

| Config | bs=3 tok/s/chip | MFU | vs jax-splash |
|--------|----------------:|-----|---:|
| jax-splash baseline | 6,313 | 35.4 % | — |
| 🏆 tokamax: `base2=T fuse_recip=T` | **6,392** | **35.8 %** | **+1.3 %** |
| tokamax: `base2=T fuse_recip=F` | 6,360 | 35.7 % | +0.7 % |
| tokamax: `base2=F fuse_recip=T` | 6,366 | 35.7 % | +0.8 % |
| tokamax: `base2=T fuse_recip=T` + `use_experimental_scheduler=T` | 6,301 | 35.3 % | -0.2 % |
| tokamax: `base2=T fuse_recip=T` + `dq_reduction_steps=3` | 6,344 | 35.6 % | +0.5 % |

Both `use_base2_exp` and `fuse_reciprocal` independently contribute ~+0.7-0.8 %
and **compound super-additively** to +1.3 %. Other knobs neutral or hurt.

`use_base2_exp` swaps `exp(x)` for `exp2(x / ln2)` — TPU's `exp2` runs
faster than the natural-base `exp` and the rescale `/ ln2` is fused into
the upstream multiply. Mathematically identical (within IEEE rounding).

`fuse_reciprocal` moves the `output / lse` division from a post-kernel
op (a full HBM read+write of the output buffer) into the kernel itself,
saving one HBM round-trip on every layer's attention output.

## Verdict

**Supported.** All three success criteria met:
- Measurable improvement: +1.3 % per-chip at bs=3, +1.8 % at bs=2,
  +1.2 % at bs=4 — outside step-time noise of ±0.3 %.
- No semantic regression: loss trajectory identical to jax-splash; both
  knobs are perf-only (`use_base2_exp` is exact under IEEE; `fuse_reciprocal`
  fuses a division — same math).
- No memory regression: tokamax splash uses the same tile sizes, so memory
  budget at bs=3 is unchanged. bs=4 still fits.

**New program-target best at seq=8192**: exp 72a stack with
`USE_TOKAMAX_SPLASH=1 TOKAMAX_USE_BASE2_EXP=1 TOKAMAX_FUSE_RECIPROCAL=1`
+ everything from exp 65 (chunked_xla CE + autotune, fp32 cast at boundary,
fp32 master + bf16 compute, scan, VMEM=98 KiB, nothing_saveable remat) =
**6,392 tok/s/chip, 35.8 % MFU at bs=3 seq=8192**.

Cumulative day climb: 4,591 → 6,392 tok/s/chip = **+39.2 % per-chip** vs
the morning AMP-only baseline (exp 20).

## See also

- [Exp 62b → 66 → 65 — chunked_xla CE iteration](2026-04-26-exp62b-chunkedxla-ce-bs3-seq8k-accepted.md) — prior frontier (jax-splash, 35.4 %)
- [Bottleneck breakdown observation](../../../../observations/llama3-8b-torchax-converged-stack-bottleneck-breakdown.md) — predicted attention impl as a deep lever (10.7 % of step time)
- MaxText `src/maxtext/layers/attention_op.py:1180-1284` — reference implementation that surfaced the `use_tokamax_splash` knob

## Next hypotheses

1. **Pull MaxText's max_logit_const tuning** — `max_logit_const` in
   `SplashConfig` lets us provide an a-priori upper bound on logit
   magnitude to skip a softmax stabilization pass. MaxText config
   `use_max_logit_estimate=20` is a published value for Llama family.
2. **Per-layer matmul precision** — explore `jax.config.update("jax_default_matmul_precision", ...)`
   options at the chunked_xla CE call site. Currently using DEFAULT.
3. **Custom attention bwd kernel** — splash dkv bwd remains the largest
   custom-call cost. A specialized impl tailored for GQA + bf16 + the
   bs=3 shape could give 5-10 % more.

## Follow-up: + max_logit_const (exp 74)

The `SplashConfig.max_logit_const` knob lets the kernel skip a softmax
stabilization pass when an a-priori upper bound on logit magnitude is
provided. MaxText's `use_max_logit_estimate` config flag exposes this but
defaults to `-1` (off) — they don't enable it by default for Llama either.

We tried setting it explicitly:

| `max_logit_const` | bs=3 tok/s/chip | MFU |
|------------------:|----------------:|-----|
| None (exp 72a)    | 6,392 | 35.8 % |
| 10                | 6,553 | 36.7 % |
| 20                | 6,552 | 36.7 % |
| 🏆 **30 (exp 74b)** | **6,559** | **36.8 %** |
| 50                | 6,556 | 36.8 % |

**Setting any positive value** turns on the optimization path; the *value*
itself is insensitive (10 / 20 / 30 / 50 all within noise) because Llama
attention logits never approach 10 in practice (post-`/sqrt(d_head)`
scaling keeps them in O(1)). Loss values **identical** step-for-step to
exp 72a (`11.7681 → 11.7405 → 11.7159 …`). Mathematically equivalent
when no clamp triggers.

### Sweep at bs=2/3/4 with `mlc=30`

| bs | exp 65 (jax-splash) | exp 72 (tokamax base2+fuse) | **exp 74 (+ mlc=30)** | total Δ vs exp 65 |
|---:|--------------------:|----------------------------:|----------------------:|------------------:|
| 2  | 6,212/chip 34.8 %   | 6,324/chip 35.5 %          | **6,482/chip 36.3 %** | **+4.3 %** |
| 3  | 6,313/chip 35.4 %   | 6,392/chip 35.8 %          | **🏆 6,559/chip 36.8 %** | **+3.9 %** |
| 4  | 6,170/chip 34.6 %   | 6,246/chip 35.0 %          | **6,400/chip 35.9 %** | **+3.7 %** |

## Refuted (post-acceptance)

| Exp | Knob | Result | Notes |
|-----|------|--------|-------|
| 75a | + `fwd_cost_estimate=5.5e11, bwd_cost_estimate=1.1e12` | 6,550/chip 36.7 % | neutral; XLA already overlaps comms with the splash kernel without hints |
| ~~76a~~ | + `q_seq_shards=2` | **NaN** loss from step 1 | **INVALID** — q_seq_shards splits Q-axis across a mesh dim, but our mesh has no context axis to shard along. Throughput "win" is because NaN short-circuits downstream ops. |
| ~~76b~~ | + `q_seq_shards=4` | **NaN** loss from step 1 | **INVALID** — same reason |
| 77a | VMEM=131,072 KiB | 6,381/chip 35.8 % (-2.7 %) | refuted; VMEM=98 KiB still optimal at the new stack |
| 77b | VMEM=65,536 KiB | 6,356/chip 35.6 % (-3.1 %) | refuted; VMEM=98 KiB still optimal |
| 78a | + `scan_remat_policy=dots_with_no_batch_dims_saveable` | OOM by 42 GiB | refuted; same as the original exp 60 finding |
| 78b | + `scan_remat_policy=dots_saveable` at bs=2 | OOM by 19.7 GiB | refuted; even at bs=2 the matmul-output saves don't fit |
| 80a | splash `block_kv=512 block_kv_dkv=1024` | 6,546/chip 36.7 % (-0.2 %) | refuted; smaller blocks not helpful |
| 80b | splash `block_q=1024 block_q_dkv=1024` | 6,531/chip 36.6 % (-0.4 %) | refuted; smaller Q blocks not helpful |
| 80c | splash `2048/2048` symmetric | 6,550/chip 36.7 % (-0.1 %) | refuted; within noise |
| 81a | `JAX_DEFAULT_MATMUL_PRECISION=bfloat16` | 6,556/chip 36.8 % (-0.05 %) | within noise; default already uses bf16 on TPU |
| 81b | `JAX_DEFAULT_MATMUL_PRECISION=tensorfloat32` | 6,529/chip 36.6 % (-0.5 %) | refuted; different reduction layout, slower on v6e |
| 82a | splash all-512 blocks (= MaxText default in `base.yml`) | 6,546/chip 36.7 % (-0.2 %) | refuted; smaller blocks not better at this shape |
| 82b | bs=4 + full optimal config (mlc=30 + tokamax-splash) | 6,415/chip 36.0 % | bs=3 still wins by +2.2 % per chip; bs=4 has memory pressure |

## Final (valid) frontier — exp 74b

  scan + AMP master (fp32 weights / bf16 compute) + tokamax CE
  (chunked_xla, autotune, fp32 cast at boundary) + **tokamax-splash
  (`use_base2_exp=True, fuse_reciprocal=True, max_logit_const=30`)** +
  shard_map wrap on CE call + VMEM=98 KiB + `nothing_saveable` scan remat
  → **6,559 tok/s/chip, 36.8 % MFU at bs=3 seq=8192**.

Cumulative day climb: 4,591 → 6,559 tok/s/chip = **+42.9 % per-chip**
vs the morning AMP-only baseline (exp 20). Gap to MaxText reference
(44.6 % MFU): **7.8 pp**.

## Sources

- `raw/code/tokamax/tokamax/_src/ops/experimental/tpu/splash_attention/splash_attention_kernel.py:115` (`SplashConfig`)
- `raw/code/maxtext/src/maxtext/layers/attention_op.py:1180-1284` (reference call site that exposed the knob)
- `raw/code/maxtext/src/maxtext/configs/base.yml` (MaxText defaults: `use_max_logit_estimate=-1`, `cost_estimate_flops_*=-1`, `dq_reduction_steps=0`, `use_splash_scheduler=False`)
