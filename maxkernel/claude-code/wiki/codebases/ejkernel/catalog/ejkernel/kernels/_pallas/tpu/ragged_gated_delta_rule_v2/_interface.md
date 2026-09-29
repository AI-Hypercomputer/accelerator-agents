---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_interface.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_interface.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule_v2._interface`/ragged_gated_delta_rule_v2().
symbols:
  ragged_gated_delta_rule_v2: ''
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_interface.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_interface.py)

## Functions
- `ragged_gated_delta_rule_v2(mixed_qkv: Float[Array, "num_tokens mixed_dim"], b: Float[Array, "num_tokens num_value_heads"], a: Float[Array, "num_tokens num_value_heads"], recurrent_state: Float[Array, "num_slots num_value_heads qk_head_dim v_head_dim"], A_log: Float[Array, num_value_heads], dt_bias: Float[Array, num_value_heads], query_start_loc: Int[Array, num_requests_plus_1], state_indices: Int[Array, num_requests], distribution: Int[Array, three], has_initial_state: Bool[Array, num_requests] | None = None, *, n_kq: int, n_v: int, d_k: int, d_v: int, chunk_size: int = 64, use_qk_norm_in_gdn: bool = True, apply_silu_in_gdr: bool = False, use_recurrent_scan_prefill: bool = False, mask_initial_state: bool = False, kernel_tile_policy: str = "auto", use_fused_gdn_decode: bool = False, runtime_dtype: object | None = None)` — [`L29`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule_v2/_interface.py#L29) — Run the packed-inference ragged GDN v2 TPU Pallas kernel.

