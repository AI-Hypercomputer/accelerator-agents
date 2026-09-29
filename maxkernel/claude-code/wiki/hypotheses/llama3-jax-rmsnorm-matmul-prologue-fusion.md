---
title: "Pallas RMSNorm+matmul-prologue fusion (Llama 3 8B JAX)"
type: hypothesis
tags: [llama3, jax, pallas, mosaic-tpu, rms-norm, matmul, fusion, deep-work]
created: 2026-04-27
updated: 2026-04-27
model: llama3-8b-jax
status: refuted
expected_gain: "+3-6 % step time (claimed) — invalidated by 2026-04-27 HLO inspection"
confidence: medium
effort: L
origin: jax-exp28b-profile-2026-04-26
---

Custom Pallas TPU kernel that fuses **RMSNorm + bf16 cast + the matmul that follows** (the QKV projections after `input_layernorm`, the gate/up projections after `post_attention_layernorm`) so the post-norm activation never round-trips through HBM. Targets the 9.2 % loop-fusion line and a portion of the 20.6 % non-MXU matmul time observed in [exp 28b's profile](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md#profile).

> [!warning] **Refuted 2026-04-27** — HLO inspection of exp 28b shows XLA already fuses this. The Q-proj fusion (`%fusion.316 = ... kind=kOutput, calls=%fused_computation.47`) inlines the entire RMSNorm body via `%fusion.301 = ... calls=%fused_computation.25`. Inside `fused_computation.25`: cast bf16→f32, broadcast rsqrt scale, multiply, broadcast RMSNorm weight, multiply, cast f32→bf16, bitcast. Then `%convolution.107 = bf16[4,8192,8,128] convolution(fusion.301, fusion.302)` is the matmul, reading the RMSNorm output directly. **One single Mosaic `kind=kOutput` kernel.** Same pattern repeats for K-proj (`fused_computation.48`), gate-proj (`fused_computation.31`), and up-proj (`fused_computation.32`) — all consume `add_rsqrt_fusion.5` and produce the matmul output in one kernel. A custom Pallas RMSNorm+matmul kernel cannot beat what XLA is already emitting; the only way to win would be to add a tiling/scheduling improvement XLA misses, which is unlikely. **Hypothesis withdrawn.** See `raw/profiles/2026-04-27-jax-hlodump-exp28b/module_0262.jit_train_step.cl_854318611.after_optimizations.hlo` (extracted to local `/tmp/hlo-exp28b-renamed/`) for the HLO evidence — search for `fused_computation.25` and `fused_computation.47`.

## Statement

Replacing `_rmsnorm(x, w) → _matmul(...)` in [`model/modeling_llama3.py:_decoder_call`](../experiments/llama3_8B_autoresearch_optimization/jax/model/modeling_llama3.py) with a Pallas-Mosaic kernel that consumes the residual `x` + RMSNorm weight, materialises the normalised value in VMEM, and feeds the MXU directly will reduce step time by **3–6 %** at bs=4 seq=8192 with no semantic change.

## Rationale

The exp 28b xprof breakdown shows:

- **Loop fusion**: 1,570 ms / step (9.2 %), 1,547 GiB/step memory traffic. Of that, **RMSNorm + bf16 cast across 4 sites/layer × 32 layers** is ~6 % of step time (each RMSNorm does an HBM read-write of the (B, T, hidden) activation = 4 × 8192 × 4096 × 2 B ≈ 256 MiB per site, sharded by 8).
- **Conv fusion (matmul)**: 60.1 %, MXU util 65.8 %. The 34.2 % MXU-missing portion is partly the matmul prologue (HBM→VMEM DMA of the post-norm activation) — a Pallas kernel that produces the activation in VMEM eliminates that DMA.

By fusing, both costs collapse into a single kernel:
- Kernel reads `x` + `w` once, writes nothing intermediate
- Matmul block reads from VMEM (free)
- Bwd uses standard custom_vjp (recompute or save-residual depending on policy)

MaxText doesn't ship this; tokamax's `layer_norm` page is documented as "TPU falls back to XLA"; axlearn / tpu-inference don't have a fused norm+matmul kernel. The closest reference is alphafold3 v3.0.1's GPU Triton fused-GLU pattern (different fusion target, same shape of optimisation).

## Proposed experiment

1. Author `pallas_rmsnorm_matmul.py` — Mosaic-TPU kernel:
   - Inputs: `x : (B*T, hidden) bf16`, `norm_w : (hidden,) bf16`, `mat_w : (hidden, out) bf16`, `eps : f32`
   - Block tile: same as standard matmul (256×128 or 512×128 depending on out)
   - Pipeline: prefetch `x` block → RMSNorm in VMEM → emit MXU tile → continue
   - Custom_vjp: re-run forward in bwd (recompute, mirrors `nothing_saveable` for these activations)
2. Drop in for the four sites in `_decoder_call`: input_layernorm→Q/K/V proj, post_attn_layernorm→gate/up proj. (out_proj and down_proj already do not have a norm immediately before them.)
3. Run with the exp 28b stack otherwise unchanged (bs=4, full SC offload, bkv=1024, MaxText XLA flag stack).

## Measurement

| Metric | Method | Pass criterion |
|--------|--------|----------------|
| tok/s/chip | trainer reported `avg throughput / 8` | ≥ 8,000 (i.e. **+3.0 %** over 7,768 baseline) — minimum to count as supported |
| MFU | trainer reported | ≥ 44.5 % |
| Loss step 0–8 | trainer log | within bf16 noise of exp 28b (Δ ≤ 0.005 per step) |
| HBM peak | xprof memory profile | not above exp 28b peak + 5 % |
| Loop-fusion % | xprof op_profile | ≤ 5 % (down from 9.2 %) |
| MXU util | xprof overview | ≥ 70 % (up from 65.8 %) |

## Risks

- **Custom_vjp correctness**. RMSNorm grad is non-trivial (rsqrt + dot); needs unit-test against `jax.grad(_rmsnorm)` at fp32 reference.
- **Block-size mismatch**. Picking RMSNorm block size that doesn't align with the matmul's MXU tile waste a cycle; needs autotune.
- **Compile-time regression**. Custom Pallas kernels add HLO bytes and can blow compile-cache hit rate. Acceptable if first-step compile < 90 s.
- **Bf16 accumulator** (RMSNorm rsqrt) destroying logit precision — must keep the rsqrt accumulator in fp32, mirror the existing `_rmsnorm` body. Validate via loss-trajectory match.

## Dependencies

- Confirmation that the splash kernel inputs do not require an RMSNorm stage (they don't — splash takes Q/K/V directly).
- Familiarity with the Pallas-Mosaic API; reference: `jax.experimental.pallas.ops.tpu.flash_attention.py` for the kernel-pipeline pattern, `tokamax/_src/ops/rms_norm.py` for the fp32-accumulator reference (CPU/GPU only path).
- This work would share ~80 % of its scaffolding with the [SwiGLU + down_proj fusion hypothesis](llama3-jax-pallas-swiglu-downproj-fusion.md); should be undertaken as a single project.

## See also

- [Pallas SwiGLU + down_proj fusion (sibling hypothesis)](llama3-jax-pallas-swiglu-downproj-fusion.md)
- [JAX exp 28b frontier writeup](../experiments/llama3_8B_autoresearch_optimization/jax/experiments/2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md)
- Concept: [rms-norm](../concepts/layer-norm.md) / [pallas-kernel](../concepts/pallas-kernel.md) / [mosaic-kernel](../concepts/mosaic-kernel.md)

## Sources

- `raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/` — profile attesting to the 9.2 % loop-fusion bottleneck.
- `raw/code/jax/jax/experimental/pallas/ops/tpu/flash_attention.py` — Mosaic-TPU kernel pipeline reference.
