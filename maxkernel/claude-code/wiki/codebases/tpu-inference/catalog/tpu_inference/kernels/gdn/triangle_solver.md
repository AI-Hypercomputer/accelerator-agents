---
title: 'Module: tpu_inference/kernels/gdn/triangle_solver.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/triangle_solver.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.triangle_solver`/
symbols:
  TriangleSolverImpl.__call__: TriangleSolverImpl#__call__().
  decompose_triangular_matrix_inverse_pallas: decompose_triangular_matrix_inverse_pallas().
  newton_schulz_inverse_pallas_kernel: newton_schulz_inverse_pallas_kernel().
  newton_schulz_inverse_pallas: newton_schulz_inverse_pallas().
  decompose_triangular_matrix_inverse_pallas_kernel: decompose_triangular_matrix_inverse_pallas_kernel().
  TriangleSolverImpl: TriangleSolverImpl#
  TriangleSolverImpl.GAUSSIAN: TriangleSolverImpl#GAUSSIAN.
  newton_schulz_inverse_ref: newton_schulz_inverse_ref().
  local_forward_substitution: local_forward_substitution().
  TriangleSolverImpl.NEWTON_SCHULZ: TriangleSolverImpl#NEWTON_SCHULZ.
---
# Module: [`tpu_inference/kernels/gdn/triangle_solver.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py)

## Classes
### `TriangleSolverImpl`  ·  implements/extends Enum, str
- def: [`tpu_inference/kernels/gdn/triangle_solver.py:200`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L200)
- signature: `class TriangleSolverImpl(str, enum.Enum):`
- members:
  - `GAUSSIAN` — [`L201`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L201)
  - `NEWTON_SCHULZ` — [`L202`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L202)
- protocol/private: `__call__`[`L205`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L205)
- uses (calls/refs, reference-scoped): [`decompose_triangular_matrix_inverse_pallas`](triangle_solver.md#decompose_triangular_matrix_inverse_pallas), [`newton_schulz_inverse_pallas`](triangle_solver.md#newton_schulz_inverse_pallas)

## Functions
- `decompose_triangular_matrix_inverse_pallas(A, *, n_block_size=64, block_size=16)` — [`L151`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L151) — Inverts unit lower triangular matrices using a block-wise approach in Pallas.
- `decompose_triangular_matrix_inverse_pallas_kernel(A_ref, x_ref, *, block_size=16)` — [`L121`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L121)
- `local_forward_substitution(A, b)` — [`L93`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L93) — Solves A X = B for unit lower triangular matrix A using forward substitution.
- `newton_schulz_inverse_pallas(A, *, block_size=64)` — [`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L70) — Newton-Schulz iteration for unit lower triangular matrices on Pallas.
- `newton_schulz_inverse_pallas_kernel(A_ref, x_ref)` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L66)
- `newton_schulz_inverse_ref(A, n=None)` — [`L25`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/triangle_solver.py#L25) — Inverse of unit lower triangular matrix using Newton-Schulz iteration.

