---
title: "Qwen3-8B jax v027 — named-offload (proj+mlpwi) + CE + bs3 @ seq8192"
type: experiment
hypothesis: qwen3-jax-host-offload-bs3-seq8192
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-named-offload-2026-06-02
status: invalid
verdict: invalid
tags: [qwen3-cc, jax, named-offload, tokamax-ce, splash, remat, seq8192, bs3, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v027 — named-offload + CE + bs3 @ seq8192

## Hypothesis under test

**Hypothesis**: Upgrading `--offload_remat` from `offload_dot_with_no_batch_dims` to the MaxText
**named-offload** recipe — `save_and_offload_only_these_names` offloading the tagged `query_proj`/`key_proj`/
`value_proj`/`out_proj`/`mlpwi` (the batch-dim activations, `bf16[B,8192,4096/12288]`) to host DRAM —
shaves the **+2.34G** that [v026](2026-06-02-v026-offload-ce-s8k-bs3.md) was short, finally **fitting bs3
at seq8192**. (CE still required to drop the `[B,L,V]` logit wall.)

**Mechanism**: image `qwen3-8b-jax:v027-offload-named` (new named-offload policy + `checkpoint_name` tags on
proj/mlpwi outputs; CPU equivalence PASS). Launch `--use_remat --use_splash --offload_remat --use_tokamax_ce
--tokamax_ce_impl=mosaic_tpu --batch_size=3 --seqlen=8192` + the v018 7-flag scheduler bundle.

**Predicted signal**: bs3 **fits** (no OOM) — the milestone the prior two runs missed. MFU/tok-s-chip
measured vs v009 (30.4% / 5,305). Per the [MFU-gap analysis](../../../../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md), named-offload **alone** (without scan + collective-overlap) is *not* expected to beat
v009 on MFU — this run **verifies the memory lever fits bs3** and measures whether batch amortizes at all
on the unrolled/synchronous-RS graph (isolating the scan+overlap contribution for the next step).

**Falsification criterion (of the fit)**: still OOM at bs3 (named-offload insufficient). For MFU: ≤ 30.4%
would confirm the analysis (fit ≠ win without scan+overlap); > 30.4% would be a bonus partial win.

## Setup

- **Hardware**: v6e-8, image `qwen3-8b-jax:v027-offload-named` (named-offload policy active — confirmed in log, though the startup print label still reads the stale "offload_dot_with_no_batch_dims" string; the actual policy is `save_and_offload_only_these_names`).
- **Launch**: `--use_remat --use_splash --offload_remat --use_tokamax_ce --tokamax_ce_impl=mosaic_tpu --batch_size=3 --seqlen=8192` + 7 LIBTPU scheduler flags.

## Results

**Compile-time HBM OOM, +2.34G — IDENTICAL to [v026](2026-06-02-v026-offload-ce-s8k-bs3.md).** Named-offload had **no effect on the footprint**. The dominant HBM temps are NOT the layer activations:
- ranks 1–2: **`f32[4096,151936]` = 2.32G each (4.64G total)** — the tokamax CE backward's LM-head weight all-gathered to f32 (`PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp` + the convert/all-gather temp).
- rank 3: `f32[3,8192,4096]` 0.38G (bf16→f32 activation cast).
- ranks 4–20: `bf16[3,8192,4096]` 192 MiB each (FSDP activations / dot intermediates).

## Profile

No profile (crash at compile pre-step-0). HLO dumped to `.../2026-06-02-qwen3-jax-v027-named-offload-ce-s8k-bs3/hlo/`.

## Verdict

**Invalid** (compile OOM, no steps). **Key correction to the v026/MFU-gap diagnosis**: the bs3 blocker is **not** the batch-dim layer activations (which named-offload targets) — it's the **tokamax CE backward's `f32[4096,151936]` LM-head weight all-gather (4.64G)**. Named-offload was active but the projection-activation savings don't touch the CE-weight path, so the deficit stayed at exactly +2.34G. **bs3-fitting via offload is a dead-end** without separate CE-weight surgery (bf16-CE path, or chunked/vocab-tiled CE).

**Strategic pivot**: this decouples the real win from bs3. The [MFU-gap analysis](../../../../analyses/2026-06-02-maxtext-vs-jax-qwen3-8b-mfu-gap.md) drivers — **MXU occupancy + collective overlap (both from scan-over-layers)** — apply at **bs1** (the v009 shape that already fits), independent of the bs3 memory fight. The next experiment implements **scan** and tests it at **bs1 seq8192 + the collective-overlap flags**, directly measuring the scan benefit without the CE-weight wall.

## Next hypotheses

- [scan-over-layers @ bs1 seq8192 + overlap flags](../../../../hypotheses/qwen3-jax-scan-layers.md) — implement `nnx.scan` over the 36 layers; test at the v009 bs1 shape (fits) + async-collective-fusion flags. Targets the RS-overlap (12.9%→~4%) + MXU-occupancy gains the profile attributed to scan. **The real lever — next.**
- bs3 via bf16/chunked CE-weight — only if scan@bs1 shows batch would help further; deprioritized (CE surgery + bs3 still won't amortize per v016).
