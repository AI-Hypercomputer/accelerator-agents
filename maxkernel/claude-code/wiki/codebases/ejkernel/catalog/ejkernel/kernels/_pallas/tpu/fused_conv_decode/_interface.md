---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_conv_decode/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_conv_decode/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_conv_decode._interface`/fused_conv_decode().
symbols:
  fused_conv_decode: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_conv_decode/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_conv_decode/_interface.py)

## Functions
- `fused_conv_decode(conv_state: Float[Array, "num_slots conv_dim d_conv"], new_tokens: Float[Array, "num_slots conv_dim"], kernel: Float[Array, "conv_dim d_conv"], *, output_dtype: jnp.dtype, activation: Callable[[Array], Array] | None = jax.nn.silu)` — [`L32`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_conv_decode/_interface.py#L32) — Fused conv-state shift, depthwise convolution, and activation (TPU Pallas).

