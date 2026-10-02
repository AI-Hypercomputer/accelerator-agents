---
title: "Qwen3-8B jax v025 — host-offload remat @ seq8192 bs2 (reopened frontier)"
type: experiment
hypothesis: qwen3-jax-host-offload-bs3-seq8192
model: qwen3-cc-jax
variant: "8B/v6e-8"
commit: jax-offload-2026-06-02
status: invalid
verdict: invalid
tags: [qwen3-cc, jax, offload-remat, host-offload, splash, remat, seq8192, v6e-8]
created: 2026-06-02
updated: 2026-06-02
---

# Qwen3-8B jax v025 — host-offload remat @ seq8192 bs2

## Hypothesis under test

**Hypothesis**: The MaxText reference proved seq8192 has ~31% headroom (MaxText 45.3% / 6,942 tok/s/chip
bs3 vs our [v009](2026-06-02-v009-splash-s8k-vmem.md) 30.4% / 5,305 bs1), achieved by **host-offloading
activations** (`decoder_layer_input` + `*_proj=offload`) to fit a bigger batch. Porting the JAX-native
analog — per-layer remat with `offload_dot_with_no_batch_dims(offload_dst="pinned_host")` instead of
`nothing_saveable` — should let **bs2** fit at seq8192 and raise MFU/tok-s-chip above v009's bs1 ceiling
(and above v016's CE-bs2 29.5%, which fit bs2 a different way without throughput gain).

**Mechanism**: new `--offload_remat` flag (this experiment's only new variable). When set, the per-layer
`jax.checkpoint` policy parks matmul activations in host DRAM instead of recomputing, freeing HBM. Run:
`--use_remat --use_splash --offload_remat --batch_size=2 --seqlen=8192` + the v018 7-flag scheduler bundle,
image `qwen3-8b-jax:v025-offload`. Semantics-preserving (offload changes *where* activations live, not the
math — CPU equivalence test passes; offload is numerically identical by JAX's remat contract).

**Predicted signal**: bs2 fits (no OOM); MFU > 30.4% / tok-s-chip > 5,305 (v009); loss ~12.07 stable; exit 0.
If it clears v009, v026 scales to bs3 (chasing the MaxText 6,942 ceiling).

**Falsification criterion**: OOM at bs2 (offload insufficient), or MFU ≤ 30.4% (host-transfer overhead
> the batch-occupancy gain — offload is the wrong lever for qwen3 at this shape).

## Setup

- **Hardware**: v6e-8 (fsdp=8), image `qwen3-8b-jax:v025-offload` (offload_remat code baked in).
- **Launch**: `--use_remat --use_splash --offload_remat --batch_size=2 --seqlen=8192` + 7 LIBTPU scheduler flags. Both `[remat] ... offload_dot_with_no_batch_dims (pinned_host) ON` and `[attn] splash ON` confirmed in log.

## Results

**Compile-time HBM OOM, pre-step-0** (no steps ran): used **38.79G of 31.25G HBM** (+7.55G / 24% over). The `offload_remat` policy activated correctly and offloaded the no-batch-dim attention/MLP dots — but the dominant consumers are the **LM-head cross-entropy intermediates**, which it doesn't cover:
- `f32[2,8192,151936]` vocab logits = **9.27G**
- `f32[16384,151936]` log_softmax intermediate = **9.27G**
- → **18.54G combined (48% of HLO temp)** — these are *batch-dim* dots, outside `offload_dot_with_no_batch_dims`' scope.

## Profile

No profile/HLO train-step module (crash before step 0). Only `module_0087.jit__identity_fn` + a memory-usage report were emitted to `gs://.../2026-06-02-qwen3-jax-v025-offload-remat-s8k-bs2/hlo/`.

## Verdict

**Invalid** (compile-time OOM, no steps executed — like v001/v002/v006/v007). The offload-remat lever *works* (it activated and freed activation HBM) but is **insufficient alone** to fit bs2 at seq8192: the `[B,L,V]` LM-head logit + log_softmax materialization (18.54G of f32, vocab=151936) is the real wall, and it's a batch-dim op the offload policy can't reach. **Key finding: the seq8192 batch ceiling is the LM-head logit wall, not the layer activations** — which is exactly what tokamax streamed CE removes ([v016](2026-06-02-v016-s8k-ce-bs2.md) fit bs2 via CE). So the path to MaxText's bs3 is the **combination**: CE (drop the logit wall) + offload-remat (free the activation HBM) + bs3. Offload alone, or CE alone (v016 bs2 = 29.5%, didn't beat bs1), aren't enough individually.

## Next hypotheses

- [offload-remat + tokamax-CE + bs3 @ seq8192](../../../../hypotheses/qwen3-jax-host-offload-bs3-seq8192.md) — the full MaxText-analog memory stack: CE streams the loss (removes the 18.54G logit wall), offload-remat frees activation HBM, together enabling **bs3** (MaxText's batch → 45.3% / 6,942 ceiling). **Dispatched as v026** (reuses the v025-offload image; CE code already present).
- If v026 also can't beat v009 (30.4%): the qwen3 jax stack doesn't amortize batch at seq8192 the way MaxText does (v016 bs2 already hinted at this) — the gap would then be MaxText's matmul/layer-fusion efficiency, not batch, and seq8192 returns to ~ceiling for our stack.
