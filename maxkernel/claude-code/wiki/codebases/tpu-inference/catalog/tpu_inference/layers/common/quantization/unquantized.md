---
title: 'Module: tpu_inference/layers/common/quantization/unquantized.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/common/quantization/unquantized.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.common.quantization.unquantized`/Unquantized
symbols:
  UnquantizedFusedMoEMethod: FusedMoEMethod#
  UnquantizedLinearMethod: LinearMethod#
  UnquantizedLinearMethod._apply_fused: LinearMethod#_apply_fused().
  UnquantizedLinearMethod._apply_split: LinearMethod#_apply_split().
  UnquantizedFusedMoEMethod.__init__: FusedMoEMethod#__init__().
  UnquantizedLinearMethod.__init__: LinearMethod#__init__().
---
# Module: [`tpu_inference/layers/common/quantization/unquantized.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py)

## Classes
### `UnquantizedFusedMoEMethod`
- def: [`tpu_inference/layers/common/quantization/unquantized.py:25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py#L25)
- doc: Shared base for jax and vllm unquantized fused-MoE methods.
- signature: `class UnquantizedFusedMoEMethod:`
- protocol/private: `__init__`[`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py#L33)
- uses (calls/refs, reference-scoped): [`UnquantizedFusedMoEMethod`](../../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod)
- used by: [`VllmUnquantizedFusedMoEMethod`](../../vllm/quantization/unquantized.md#VllmUnquantizedFusedMoEMethod), [`UnquantizedFusedMoEMethod`](../../jax/quantization/unquantized.md#UnquantizedFusedMoEMethod)

### `UnquantizedLinearMethod`
- def: [`tpu_inference/layers/common/quantization/unquantized.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py#L38)
- doc: Implements the forward method for unquantized linear layers.
- signature: `class UnquantizedLinearMethod:`
- members:
  - `_apply_fused(self, x_jax: jax.Array, weight_jax: jax.Array, bias_jax: Optional[jax.Array]=None, einsum_str: str='...k,kn->...n')` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py#L47) — Applies fused linear operation.
  - `_apply_split(self, x_jax: jax.Array, weights: Sequence[jax.Array], bias_jax: Optional[Sequence[jax.Array]]=None)` — [`L75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py#L75) — Applies split linear operation.
- protocol/private: `__init__`[`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/common/quantization/unquantized.py#L44)
- uses (calls/refs, reference-scoped): [`UnquantizedLinearMethod`](../../jax/quantization/unquantized.md#UnquantizedLinearMethod)
- used by: [`apply_jax`](../../jax/quantization/unquantized.md#UnquantizedLinearMethod.apply_jax), [`VllmUnquantizedLinearMethod`](../../vllm/quantization/unquantized.md#VllmUnquantizedLinearMethod), [`UnquantizedLinearMethod`](../../jax/quantization/unquantized.md#UnquantizedLinearMethod), [`apply`](../../vllm/quantization/unquantized.md#VllmUnquantizedLinearMethod.apply)

