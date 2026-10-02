---
title: 'Module: ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.deepseek_attn._pallas_impl_fwd`/
symbols:
  _deepseek_attn_pallas_call: _deepseek_attn_pallas_call().
  deepseek_attn_pallas_impl: deepseek_attn_pallas_impl().
  _sparse_mask_to_bias: _sparse_mask_to_bias().
  _deepseek_attn_fwd: _deepseek_attn_fwd().
  _deepseek_attn_bwd: _deepseek_attn_bwd().
  ROPE_NONE: ROPE_NONE.
  ROPE_FUSED: ROPE_FUSED.
  ROPE_DECOUPLED: ROPE_DECOUPLED.
  _build_sparse_mask: _build_sparse_mask().
  __all__: __all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py)

## Functions
- `_build_sparse_mask(topk_indices: jnp.ndarray, seq_len: int)` — [`L33`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L33) — Build a dense float32 keep-mask from top-k indices.
- `_deepseek_attn_bwd(rope_mode, causal, softmax_scale, block_b, block_q, block_k, residuals, do)` — [`L220`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L220) — Backward rule for ``custom_vjp``.
- `_deepseek_attn_fwd(q, kv_latent, w_kc, w_vc, b_q, b_k, sparse_mask, rope_mode, causal, softmax_scale, block_b, block_q, block_k)` — [`L180`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L180) — Forward rule for ``custom_vjp``: run raw forward and save residuals.
- `_deepseek_attn_pallas_call(q, kv_latent, w_kc, w_vc, b_q, b_k, sparse_mask, *, rope_mode, causal, softmax_scale, block_b, block_q, block_k, save_residuals: bool = False)` — [`L77`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L77) — Run the DeepSeek TPU forward pass via the raw Flash-MLA Pallas call.
- `_sparse_mask_to_bias(sparse_mask: jnp.ndarray)` — [`L61`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L61) — Convert a float keep-mask into the additive bias Flash MLA expects.
- `deepseek_attn_pallas_impl(q, kv_latent, w_kc, w_vc, b_q, b_k, sparse_mask, rope_mode, causal, softmax_scale, block_b, block_q, block_k)` — [`L140`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L140) — DeepSeek Sparse Attention forward with explicit TPU ``custom_vjp``.

## Module values
- `ROPE_DECOUPLED` — [`L30`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L30)
- `ROPE_FUSED` — [`L29`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L29)
- `ROPE_NONE` — [`L28`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L28)
- `__all__` — [`L252`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_fwd.py#L252)

