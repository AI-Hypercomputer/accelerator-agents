---
title: 'Module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.kernels.gmm.megablox_gmm_backend`/gmm().
symbols:
  gmm: ''
---
# Module: [`python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py)

## Functions
- `gmm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, preferred_element_type: jnp.dtype = jnp.float32, rhs_scale: jnp.ndarray | None = None, rhs_bias: jnp.ndarray | None = None, tiling: tuple[int, int, int] | None = None, group_offset: jnp.ndarray | None = None, existing_out: jnp.ndarray | None = None, interpret: bool | None = None, maybe_quantize_lhs: bool = True, zero_initialize: bool = True, acc_dtype: jnp.dtype | None = None, activation_quantized_dtype: jnp.dtype | None = None, v2_tile_info: Any = None)` — [`L15`](../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/kernels/gmm/megablox_gmm_backend.py#L15) — Dispatch GMM to v2 or v1, with optional activation quantization. — documented in [python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2](../../../../../../concepts/python-sgl_jax-srt-kernels-gmm-megablox_gmm_kernel-gmm_v2.md)

