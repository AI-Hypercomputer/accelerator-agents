---
title: 'Module: ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.deepseek_attn._pallas_impl_bwd`/_
symbols:
  _deepseek_attn_bwd_impl: deepseek_attn_bwd_impl().
  _sparse_mask_to_bias: sparse_mask_to_bias().
  __all__: _all__.
---
# Module: [`ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py)

## Functions
- `_deepseek_attn_bwd_impl(rope_mode, causal, softmax_scale, block_q, block_k, residuals, do)` — [`L41`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py#L41) — Compute DeepSeek TPU gradients using the Flash-MLA backward kernels.
- `_sparse_mask_to_bias(sparse_mask)` — [`L25`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py#L25) — Convert a float keep-mask into an additive bias for Flash MLA backward.

## Module values
- `__all__` — [`L97`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/deepseek_attn/_pallas_impl_bwd.py#L97)

