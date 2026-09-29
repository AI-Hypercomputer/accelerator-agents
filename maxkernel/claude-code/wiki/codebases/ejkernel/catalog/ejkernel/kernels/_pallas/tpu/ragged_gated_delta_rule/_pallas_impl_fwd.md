---
title: 'Module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._pallas.tpu.ragged_gated_delta_rule._pallas_impl_fwd`/
symbols:
  _F32: _F32.
  run_ragged_gdr_decode_pallas: run_ragged_gdr_decode_pallas().
  _gdr_decode_kernel: _gdr_decode_kernel().
---
# Module: [`ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py)

## Functions
- `_gdr_decode_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_ref, out_ref, state_out_ref, *, num_heads: int)` — [`L34`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py#L34) — Pallas kernel for single-token ragged GDR decode.
- `run_ragged_gdr_decode_pallas(query, key, value, beta, decay, gathered_state, *, use_l2norm=False)` — [`L89`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py#L89) — Launch the Pallas decode kernel on pre-gathered per-token states.

## Module values
- `_F32` — [`L31`](../../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_pallas/tpu/ragged_gated_delta_rule/_pallas_impl_fwd.py#L31)

