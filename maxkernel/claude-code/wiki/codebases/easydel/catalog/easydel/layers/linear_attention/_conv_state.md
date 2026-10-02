---
title: 'Module: easydel/layers/linear_attention/_conv_state.py'
type: catalog
provenance: extracted
module: easydel/layers/linear_attention/_conv_state.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.linear_attention._conv_state`/
symbols:
  apply_conv_with_state: apply_conv_with_state().
  apply_manual_depthwise_conv: apply_manual_depthwise_conv().
  shift_conv_state_left: shift_conv_state_left().
---
# Module: [`easydel/layers/linear_attention/_conv_state.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/linear_attention/_conv_state.py)

## Functions
- `apply_conv_with_state(x: Float[Array, "batch seq_len dim"], conv_layer: nn.Conv, conv_state: Float[Array, "batch dim d_conv"] | None, *, is_inference: bool, d_conv: int, output_dtype: jnp.dtype, activation: tp.Callable[[Array], Array] | None = jax.nn.silu, reuse_partial_state: bool = False)` — [`L121`](../../../../../../../raw/code/EasyDeL/easydel/layers/linear_attention/_conv_state.py#L121) — Run a causal depthwise convolution and maintain its rolling state.
- `apply_manual_depthwise_conv(conv_state: Array, kernel: Array, *, output_dtype: jnp.dtype, activation: tp.Callable[[Array], Array] | None = jax.nn.silu)` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/linear_attention/_conv_state.py#L81) — Compute a depthwise convolution from the cached state and kernel.
- `shift_conv_state_left(conv_state: Array, new_value: Array)` — [`L54`](../../../../../../../raw/code/EasyDeL/easydel/layers/linear_attention/_conv_state.py#L54) — Shift a causal convolution cache one position left and append the newest token.

