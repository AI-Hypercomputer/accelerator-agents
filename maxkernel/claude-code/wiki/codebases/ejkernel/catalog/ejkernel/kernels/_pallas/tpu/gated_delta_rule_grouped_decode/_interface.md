---
title: 'Module: ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.gated_delta_rule_grouped_decode._interface`/gated_delta_rule_grouped_decode().
symbols:
  gated_delta_rule_grouped_decode: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_interface.py)

## Functions
- `gated_delta_rule_grouped_decode(query: Float[Array, "batch num_k_heads head_dim"], key: Float[Array, "batch num_k_heads head_dim"], value: Float[Array, "batch num_k_heads expand_ratio value_dim"], beta: Float[Array, "batch num_k_heads expand_ratio"], decay: Float[Array, "batch num_k_heads expand_ratio"] | None, recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"])` — [`L29`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/gated_delta_rule_grouped_decode/_interface.py#L29) — Grouped GDR (gated delta-rule) decode step using the TPU Pallas kernel.

