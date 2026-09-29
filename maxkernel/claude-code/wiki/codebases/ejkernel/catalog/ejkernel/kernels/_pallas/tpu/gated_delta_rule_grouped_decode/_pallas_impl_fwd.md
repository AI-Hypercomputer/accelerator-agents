---
title: 'Module: ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.gated_delta_rule_grouped_decode._pallas_impl_fwd`/
symbols:
  gated_delta_rule_grouped_decode: gated_delta_rule_grouped_decode().
  _gated_delta_rule_grouped_decode_kernel: _gated_delta_rule_grouped_decode_kernel().
---
# Module: [`ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py)

## Functions
- `_gated_delta_rule_grouped_decode_kernel(query_ref, key_ref, value_ref, beta_ref, decay_ref, state_ref, output_ref, state_out_ref)` — [`L26`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py#L26) — Pallas kernel body for a single grouped GDR (gated delta-rule) decode step.
- `gated_delta_rule_grouped_decode(query: Float[Array, "batch num_k_heads head_dim"], key: Float[Array, "batch num_k_heads head_dim"], value: Float[Array, "batch num_k_heads expand_ratio value_dim"], beta: Float[Array, "batch num_k_heads expand_ratio"], decay: Float[Array, "batch num_k_heads expand_ratio"], recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"])` — [`L122`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_pallas_impl_fwd.py#L122) — Pallas-accelerated grouped GDR (gated delta-rule) decode step targeting TPU.

