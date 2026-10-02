---
title: 'Module: ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.gated_delta_rule._xla_impl_bwd`/_
symbols:
  _MATMUL_PRECISION: MATMUL_PRECISION.
  _chunk_gdr_bwd: chunk_gdr_bwd().
  _chunk_gdr_bwd._cast_grad: chunk_gdr_bwd()._cast_grad().
  _chunk_gdr_bwd.fwd_state_scan: chunk_gdr_bwd().fwd_state_scan().
  _chunk_gdr_bwd.rev_chunk_scan: chunk_gdr_bwd().rev_chunk_scan().
---
# Module: [`ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py)

## Functions
- `_cast_grad(x)` — [`L293`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py#L293)
- `_chunk_gdr_bwd(chunk_size: int, use_qk_l2norm: bool, res: tuple, g: tuple[Float[Array, ...], Float[Array, ...]])` — [`L38`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py#L38) — Analytical backward pass for chunked GDR in float32.
- `fwd_state_scan(state, inputs)` — [`L121`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py#L121)
- `rev_chunk_scan(d_state_next, inputs)` — [`L145`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py#L145)

## Module values
- `_MATMUL_PRECISION` — [`L35`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_bwd.py#L35)

