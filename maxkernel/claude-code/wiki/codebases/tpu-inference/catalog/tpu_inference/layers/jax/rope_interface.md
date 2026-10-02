---
title: 'Module: tpu_inference/layers/jax/rope_interface.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/rope_interface.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.rope_interface`/
symbols:
  apply_rope: apply_rope().
  get_rope_scaling: get_rope_scaling().
  get_rope_theta: get_rope_theta().
  normalize_rope_scaling: normalize_rope_scaling().
  apply_rope_scaling: apply_rope_scaling().
  apply_longrope: apply_longrope().
---
# Module: [`tpu_inference/layers/jax/rope_interface.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py)

## Functions
- `apply_longrope(inputs: jax.Array, positions: jax.Array, head_dim: int, rope_scaling: Dict[str, Any], original_max_position_embeddings: int, max_position_embeddings: int, rope_theta: float = 10000)` — [`L242`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py#L242)
- `apply_rope(inputs: jax.Array, positions: jax.Array, head_dim: int, rope_theta: float = 10000, rope_scaling: Dict[str, Any] = None, rope_input_ordering: str = "split", *, rope_proportion: float = 1)` — [`L48`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py#L48) — Applies Rotary Positional Embedding using the sine and cosine strategy. — documented in [tpu_inference-layers-jax-attention](../../../../concepts/tpu_inference-layers-jax-attention.md)
- `apply_rope_scaling(freqs: jax.Array, rope_scaling: Dict[str, Any])` — [`L295`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py#L295)
- `get_rope_scaling(config: Any)` — [`L35`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py#L35)
- `get_rope_theta(config: Any, default: float = 10000)` — [`L41`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py#L41)
- `normalize_rope_scaling(rope_scaling: Any)` — [`L22`](../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/rope_interface.py#L22)

