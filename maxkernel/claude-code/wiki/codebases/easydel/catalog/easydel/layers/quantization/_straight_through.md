---
title: 'Module: easydel/layers/quantization/_straight_through.py'
type: catalog
provenance: extracted
module: easydel/layers/quantization/_straight_through.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.quantization._straight_through`/
symbols:
  straight_through: straight_through().
  _straight_through_ejkernel: _straight_through_ejkernel().
  straight_through_mxfp8: straight_through_mxfp8().
  straight_through_nvfp8: straight_through_nvfp8().
  straight_through_mxfp4: straight_through_mxfp4().
  straight_through_nf4: straight_through_nf4().
  straight_through_8bit: straight_through_8bit().
  straight_through_1bit: straight_through_1bit().
  _ejkernel_dequantized: _ejkernel_dequantized().
  _straight_through_cast: _straight_through_cast().
---
# Module: [`easydel/layers/quantization/_straight_through.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py)

## Functions
- `_ejkernel_dequantized(weights: jax.Array, *, mode: str, group_size: int, bits: int)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L82)
- `_straight_through_cast(weights: jax.Array, *, dtype: jnp.dtype)` — [`L117`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L117)
- `_straight_through_ejkernel(weights: jax.Array, *, mode: str, group_size: int, bits: int)` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L106)
- `straight_through(array: jax.Array, config: QuantizationConfig | None = None, dtype: QuantizationType | str | None = None, group_size: int | None = None)` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L237) — Unified straight-through estimator for all supported quantization types.
- `straight_through_1bit(weights: jax.Array, axis: int | None = None)` — [`L225`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L225) — Apply straight-through estimation with binary quantization.
- `straight_through_8bit(weights: jax.Array, axis: int | None = None, *, group_size: int = 64)` — [`L208`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L208) — Apply straight-through estimation with 8-bit affine quantization.
- `straight_through_mxfp4(weights: jax.Array)` — [`L176`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L176) — Apply straight-through estimation with MXFP4 (E2M1) quantization.
- `straight_through_mxfp8(weights: jax.Array)` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L121) — Apply straight-through estimation with MXFP8 microscaling quantization.
- `straight_through_nf4(weights: jax.Array, block_size: int = 64)` — [`L203`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L203) — Apply straight-through estimation with NF4 quantization via ejkernel.
- `straight_through_nvfp8(weights: jax.Array)` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_straight_through.py#L149) — Apply straight-through estimation with NVIDIA FP8 (E4M3) quantization.

