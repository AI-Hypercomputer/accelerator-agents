---
title: 'Module: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.reference.ragged_gated_delta_rule_ref`/
symbols:
  _l2_normalize: _l2_normalize().
  _recurrent_gated_delta_rule_step: _recurrent_gated_delta_rule_step().
  ragged_gated_delta_rule: ragged_gated_delta_rule().
---
# Module: [`tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py)

## Functions
- `_l2_normalize(x: jnp.ndarray, eps: float=1e-06)` — [`L22`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py#L22) — L2 normalize along last dimension.
- `_recurrent_gated_delta_rule_step(query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray, g: jnp.ndarray, beta: jnp.ndarray, state: Optional[jnp.ndarray]=None)` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py#L40) — Single-step recurrent update for decode.
- `ragged_gated_delta_rule(mixed_qkv, b, a, recurrent_state, A_log, dt_bias, query_start_loc, state_indices, distribution, has_initial_state, *, n_kq, n_v, d_k, d_v)` — [`L100`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/reference/ragged_gated_delta_rule_ref.py#L100) — Applies the gated delta rule over ragged sequences and updates recurrent state.

