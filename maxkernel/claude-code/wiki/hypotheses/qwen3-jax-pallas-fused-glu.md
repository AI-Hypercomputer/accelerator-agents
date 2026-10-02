---
title: "qwen3-jax-pallas-fused-glu"
type: hypothesis
tags: [qwen3, jax, pallas, fused-glu]
model: qwen3-ag
lane: jax
status: refuted
created: 2026-06-27
updated: 2026-06-27
---

# Hypothesis: Fused GLU Pallas kernel

**Status**: refuted
**Target**: `qwen3-ag` `jax` (8B/v6e-8)
**Topic**: Pallas kernels / MLP Fusion

## Background

Our current optimal baseline (v009) achieves 32.5% MFU using `jax.remat` + `nnx.scan` over layers, running `seqlen=8192` with `batch_size=1` per chip.
In Retrospective v3, Fused GLU was identified as the next highest ROI optimization to explore. The MLP blocks in Qwen3 use a SwiGLU variant. A custom Pallas kernel can fuse the up-projection, activation function, and element-wise multiplication, significantly reducing HBM read/write traffic during the MLP step.

## Hypothesis under test

**Hypothesis**: By replacing the standard SwiGLU MLP implementation with a fused Pallas GLU kernel, we can reduce HBM bandwidth pressure in the MLP blocks, increasing overall MFU.

**Mechanism**: We will port a fused GLU Pallas kernel (referencing the AlphaFold3 Pallas catalog or similar implementations) and replace the standard up-projection and gate logic in the Qwen3 MLP blocks.

**Predicted signal**: We expect to see a reduction in memory bandwidth utilization in the profile for the MLP block, and a corresponding decrease in step time, pushing MFU above the 32.5% baseline.

**Falsification criterion**: The step time regresses or remains identical due to XLA already performing optimal fusion for the GLU operations, or compilation fails due to custom VJP limits.

## Implementation Plan

1. Integrate a Pallas Fused GLU implementation.
2. Replace the MLP computation in the Qwen3 model.
3. Validate compilation and custom VJP (to avoid the crash seen in v011).
4. Run the experiment on top of the v009 baseline (batch=1, seqlen=8192).

## See also

- [Retrospective v3](../analyses/2026-06-27-qwen3-jax-retrospective-3.md)
- [Pallas Kernel Catalog](../analyses/2026-04-23-pallas-kernel-directory.md)
## Verdict
**Refuted.** The MFU regression is caused by redundant recomputation and excessive HBM traffic introduced by the custom VJP backward pass:

1. **Redundant Matmuls (3x instead of 2x)**: Because `fused_glu` is wrapped in a `jax.custom_vjp` that does not output the intermediate pre-activations (`u` and `gate`), the backward pass `fused_glu_bwd` explicitly recomputes them using `jnp.dot`. Furthermore, because the entire layer is wrapped in `jax.remat`, XLA re-evaluates the forward Pallas kernel (`shard_map.509`) during the backward pass to produce `out` for downstream gradients (e.g., `dw_down`). This results in the heavy `up` and `gate` projections being computed **three times** per step (Forward Pallas, Remat Pallas, Backward JAX), whereas the baseline computes them only twice (Forward, Remat).
2. **HBM Materialization in Backward Pass**: The backward pass uses standard `jnp.dot`, which forces the allocation and materialization of the massive `8192 x 12288` tensors (`u`, `gate`, `du`, `dgate`) into HBM. This completely negates any memory bandwidth savings achieved by fusing them in the forward pass.
3. **Performance**: The Pallas kernel itself (`shard_map.510`) takes ~2.93ms per call (~70 TFLOPs/s per chip), which is heavily bottlenecked by HBM memory transfers of the inputs. Standard XLA `dot_general` is highly optimized and may pipeline TMA loads better than this custom loop over `K`.

**Next Steps**: To realize MFU gains from a fused GLU, a custom Pallas *backward* kernel is strictly required to compute `dx`, `dw_up`, and `dw_gate` on the fly without materializing `u` and `gate` into HBM, or we should rely on standard XLA fusions for elementwise epilogues.
