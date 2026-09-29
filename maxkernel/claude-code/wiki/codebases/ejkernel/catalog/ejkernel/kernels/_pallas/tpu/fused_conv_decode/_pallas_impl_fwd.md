---
title: 'Module: ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.fused_conv_decode._pallas_impl_fwd`/
symbols:
  fused_conv_decode: fused_conv_decode().
  _fused_conv_decode_kernel: _fused_conv_decode_kernel().
---
# Module: [`ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py)

## Functions
- `_fused_conv_decode_kernel(conv_state_ref, new_tokens_ref, kernel_ref, updated_state_ref, conv_output_ref)` — [`L26`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py#L26) — Pallas kernel that fuses conv-state shift, depthwise convolution, and SiLU.
- `fused_conv_decode(conv_state: Float[Array, "num_slots conv_dim d_conv"], new_tokens: Float[Array, "num_slots conv_dim"], kernel: Float[Array, "conv_dim d_conv"], *, output_dtype: jnp.dtype)` — [`L81`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/fused_conv_decode/_pallas_impl_fwd.py#L81) — Pallas-accelerated fused conv-state update targeting TPU.

