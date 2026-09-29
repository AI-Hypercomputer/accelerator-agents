---
title: 'Module: ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd`/_
symbols:
  _MATMUL_PRECISION: MATMUL_PRECISION.
  _recurrent_gdr_fwd: recurrent_gdr_fwd().
  _l2norm: l2norm().
  _chunk_gdr_fwd_core: chunk_gdr_fwd_core().
  _l2norm_with_inv: l2norm_with_inv().
  _l2norm_bwd: l2norm_bwd().
  _single_step_gdr_fwd: single_step_gdr_fwd().
  _strict_lower_inverse: strict_lower_inverse().
  _recurrent_gdr_fwd.scan_body: recurrent_gdr_fwd().scan_body().
  _recurrent_gdr_fwd.scan_body_seg: recurrent_gdr_fwd().scan_body_seg().
  _chunk_gdr_fwd_impl: chunk_gdr_fwd_impl().
  _chunk_gdr_fwd_core.chunk_step: chunk_gdr_fwd_core().chunk_step().
  _chunk_gdr_fwd_neumann: chunk_gdr_fwd_neumann().
  _chunk_gdr_fwd_rule: chunk_gdr_fwd_rule().
  _chunk_gdr_bwd_rule: chunk_gdr_bwd_rule().
  _chunk_gdr_fwd: chunk_gdr_fwd().
  _strict_lower_inverse._solve_one: strict_lower_inverse()._solve_one().
  _recurrent_gdr_fwd._solve_one: recurrent_gdr_fwd()._solve_one().
---
# Module: [`ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py)

## Functions
- `_chunk_gdr_bwd_rule(chunk_size, use_qk_l2norm, res, g)` — [`L756`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L756) — Backward rule for the custom_vjp of the Neumann-series chunked GDR.
- `_chunk_gdr_fwd(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True, seg_ids: Int[Array, "batch seq_len"] | None = None)` — [`L781`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L781) — Exact multi-token chunked GDR forward path (triangular-solve formulation).
- `_chunk_gdr_fwd_core(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm, save_residual: bool)` — [`L472`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L472) — Shared chunked forward path, optionally capturing backward residuals.
- `_chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm)` — [`L441`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L441) — Core implementation for the Neumann-series chunked GDR forward.
- `_chunk_gdr_fwd_neumann(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, chunk_size: int = 64, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True)` — [`L669`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L669) — Chunked forward pass for gated delta rule with custom backward.
- `_chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm)` — [`L720`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L720) — Forward rule for the custom_vjp of the Neumann-series chunked GDR.
- `_l2norm(x: Float[Array, ...], axis: int = -1, eps: float = 0.000001)` — [`L54`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L54) — Apply L2 normalization along specified axis.
- `_l2norm_bwd(grad_y, y, inv_norm)` — [`L86`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L86) — Backward pass for y = l2norm(x).
- `_l2norm_with_inv(x: Float[Array, ...], axis: int = -1, eps: float = 0.000001)` — [`L69`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L69) — Returns both normalized tensor and inverse norm.
- `_recurrent_gdr_fwd(query: Float[Array, "batch num_heads seq_len head_dim"], key: Float[Array, "batch num_heads seq_len head_dim"], value: Float[Array, "batch num_heads seq_len d_state"], beta: Float[Array, "batch num_heads seq_len"], decay: Float[Array, "batch num_heads seq_len"] | None, initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None, use_qk_l2norm: bool = True, chunk_size: int = 64, seg_ids: Int[Array, "batch seq_len"] | None = None)` — [`L146`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L146) — Chunked forward pass for Gated Delta Rule using exact triangular solve.
- `_single_step_gdr_fwd(query: Float[Array, "batch num_heads 1 head_dim"], key: Float[Array, "batch num_heads 1 head_dim"], value: Float[Array, "batch num_heads 1 d_state"], beta: Float[Array, "batch num_heads 1"], decay: Float[Array, "batch num_heads 1"] | None, recurrent_state: Float[Array, "batch num_heads head_dim d_state"], use_qk_l2norm: bool = True)` — [`L842`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L842) — Single-step GDR update optimized for autoregressive inference.
- `_solve_one(lhs)` — [`L122`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L122) — Invert one unit-lower-triangular matrix ``(I + L)`` via a single triangular solve.
- `_solve_one(m)` — [`L315`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L315) — Invert one ``[C, C]`` unit-lower-triangular intra-chunk system via ``solve_triangular``.
- `_strict_lower_inverse(matrix_strict_lower: jax.Array)` — [`L101`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L101) — Compute the inverse of (I + L) where L is a batch of strict-lower-triangular matrices.
- `chunk_step(state, inputs)` — [`L591`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L591) — Process one chunk in the inter-chunk ``lax.scan``: emit its output and update the recurrent state.
- `scan_body(state, inputs)` — [`L371`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L371) — Process one chunk: apply inter-chunk state contribution and update state.
- `scan_body_seg(carry, inputs)` — [`L395`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L395) — Segment-aware chunk step for sequence packing.

## Module values
- `_MATMUL_PRECISION` — [`L51`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gated_delta_rule/_xla_impl_fwd.py#L51)

