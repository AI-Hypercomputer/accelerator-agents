---
title: 'Module: tpu_inference/layers/jax/moe/dense_moe.py'
type: catalog
provenance: extracted
module: tpu_inference/layers/jax/moe/dense_moe.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.layers.jax.moe.dense_moe`/dense_moe_f
symbols:
  dense_moe_fwd: wd().
  dense_moe_fwd_preapply_router_weights: wd_preapply_router_weights().
  dense_moe_func: unc().
---
# Module: [`tpu_inference/layers/jax/moe/dense_moe.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/dense_moe.py)

## Functions
- `dense_moe_func(weights: UnfusedMoEWeights, x_TD: jax.Array, gating_output: Tuple[jax.Array, jax.Array], cast_dtype: jnp.dtype, num_local_experts: int, apply_expert_weight_before_computation: bool, activation_ffw_td: Sharding, activation_ffw_ted: Sharding, hidden_act: str, mesh: Mesh)` — [`L108`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/dense_moe.py#L108) — Forward pass of the dense MoE layer.  This is a naive implementation
- `dense_moe_fwd(weights: UnfusedMoEWeights, x_TD: Float, cast_dtype: jnp.dtype, activation_ffw_td: Sharding, hidden_act: str, full_weights_TE: jax.Array, mesh: Mesh)` — [`L28`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/dense_moe.py#L28) — Forward pass of the dense Moe layer where we don't pre-apply the weights.
- `dense_moe_fwd_preapply_router_weights(weights: UnfusedMoEWeights, x_TD: Float, cast_dtype: jnp.dtype, activation_ffw_ted: Sharding, hidden_act: str, full_weights_TE: jax.Array, mesh: Mesh)` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/layers/jax/moe/dense_moe.py#L65) — Forward pass of the dense Moe layer where we pre-apply the weights.

