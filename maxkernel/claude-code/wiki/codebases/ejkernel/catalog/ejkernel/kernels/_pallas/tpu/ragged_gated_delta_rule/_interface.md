---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule._interface`/
symbols:
  ragged_gated_delta_rule: ragged_gated_delta_rule().
  _decode_path: _decode_path().
  ragged_gated_delta_rule_decode: ragged_gated_delta_rule_decode().
  ragged_gated_delta_rule.decode_fn: ragged_gated_delta_rule().decode_fn().
  ragged_gated_delta_rule.prefill_fn: ragged_gated_delta_rule().prefill_fn().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py)

## Functions
- `_decode_path(query, key, value, beta, decay, recurrent_state, state_indices, use_qk_l2norm)` — [`L37`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py#L37) — Execute the Pallas decode path: L2 norm → gather → kernel → scatter.
- `decode_fn(_)` — [`L202`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py#L202)
- `prefill_fn(_)` — [`L214`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py#L214)
- `ragged_gated_delta_rule(query: Float[Array, "num_tokens num_heads qk_head_dim"], key: Float[Array, "num_tokens num_heads qk_head_dim"], value: Float[Array, "num_tokens num_heads v_head_dim"], beta: Float[Array, "num_tokens num_heads"], decay: Float[Array, "num_tokens num_heads"] | None, recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"], query_start_loc: Int[Array, num_requests_plus_1], state_indices: Int[Array, num_requests], *, chunk_size: int = 64, use_qk_l2norm: bool = True)` — [`L119`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py#L119) — Ragged GDR with Pallas TPU decode kernel and XLA chunked prefill fallback.
- `ragged_gated_delta_rule_decode(query: Float[Array, "num_tokens num_heads qk_head_dim"], key: Float[Array, "num_tokens num_heads qk_head_dim"], value: Float[Array, "num_tokens num_heads v_head_dim"], beta: Float[Array, "num_tokens num_heads"], decay: Float[Array, "num_tokens num_heads"], recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"], state_indices: Int[Array, num_requests], *, use_qk_l2norm: bool = True)` — [`L61`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_interface.py#L61) — Direct decode-only Pallas path without ``lax.cond`` dispatch.

