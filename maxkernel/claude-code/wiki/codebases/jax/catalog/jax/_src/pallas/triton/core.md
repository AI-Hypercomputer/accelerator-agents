---
title: 'Module: jax/_src/pallas/triton/core.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/triton/core.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.triton.core`/CompilerParams#
symbols:
  CompilerParams: ''
  CompilerParams.num_warps: num_warps.
  CompilerParams.num_stages: num_stages.
---
# Module: [`jax/_src/pallas/triton/core.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/core.py)

## Classes
### `CompilerParams`
- def: [`jax/_src/pallas/triton/core.py:22`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/core.py#L22)
- doc: Compiler parameters for Triton.
- signature: `class CompilerParams:`
- members:
  - `num_stages` — [`L32`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/core.py#L32)
  - `num_warps` — [`L31`](../../../../../../../../raw/code/jax/jax/_src/pallas/triton/core.py#L31)
- used by: [`pallas_call_lowering`](pallas_call_registration.md#pallas_call_lowering), [`_mha_backward`](../../../experimental/pallas/ops/gpu/attention.md#_mha_backward), [`mha`](../../../experimental/pallas/ops/gpu/attention.md#mha), [`paged_attention_unbatched`](../../../experimental/pallas/ops/gpu/paged_attention.md#paged_attention_unbatched), [`decode_attn_unbatched`](../../../experimental/pallas/ops/gpu/decode_attention.md#decode_attn_unbatched), [`softmax`](../../../experimental/pallas/ops/gpu/softmax.md#softmax), [`_preprocess_backward`](../../../experimental/pallas/ops/gpu/attention.md#_preprocess_backward), [`layer_norm_backward`](../../../experimental/pallas/ops/gpu/layer_norm.md#layer_norm_backward), [`rms_norm_backward`](../../../experimental/pallas/ops/gpu/rms_norm.md#rms_norm_backward), [`layer_norm`](../../../experimental/pallas/ops/gpu/layer_norm.md#layer_norm), [`rms_norm`](../../../experimental/pallas/ops/gpu/rms_norm.md#rms_norm), [`gpu_lowering`](../pallas_call.md#_pallas_call_lowering.gpu_lowering), [`layer_norm_forward`](../../../experimental/pallas/ops/gpu/layer_norm.md#layer_norm_forward), [`rms_norm_forward`](../../../experimental/pallas/ops/gpu/rms_norm.md#rms_norm_forward)

