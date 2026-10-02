---
title: 'Module: tpu_inference/layers/jax/moe/sparse_moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/moe/sparse_moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.moe.sparse_moe`/
symbols:
  sparse_moe_func: sparse_moe_func().
  sparse_moe_distributed_fwd: sparse_moe_distributed_fwd().
  _process_weight_for_qwix: _process_weight_for_qwix().
---
# Module: [`tpu_inference/layers/jax/moe/sparse_moe.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/sparse_moe.py)

## Functions
- `_process_weight_for_qwix(qwix_quantized_weight_dtype: jnp.dtype, name: str, weight_param: Union[ptq.WithAux, nnx.Param], channelwise_axes: Optional[List[int]] = [], tiled_axes: dict = {})` — [`L299`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/sparse_moe.py#L299) — Extracts weight value, applies quantization if needed,
- `sparse_moe_distributed_fwd(moe_instance, x_TD: jax.Array, router_weights_TX: jax.Array, selected_experts_TX: jax.Array, kernel_gating: jax.Array, kernel_up_proj: jax.Array, kernel_down_proj: jax.Array)` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/sparse_moe.py#L43) — The sparse MoE forward pass with fully distributed logic.
- `sparse_moe_func(weights: UnfusedMoEWeights, x_TD: jax.Array, gating_output: Tuple[jax.Array, jax.Array], layer: Union[RoutedExperts, JaxMoE], mesh: Mesh)` — [`L234`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/sparse_moe.py#L234) — documented in [tpu_inference-layers-jax-moe](../../../../../concepts/tpu_inference-layers-jax-moe.md)

