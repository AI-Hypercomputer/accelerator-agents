---
title: 'Module: tpu_inference/kernels/sparse_core/core_map_helper.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/sparse_core/core_map_helper.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.sparse_core.core_map_helper`/
symbols:
  kernel: kernel().
  _empty_out_ref: _empty_out_ref().
---
# Module: [`tpu_inference/kernels/sparse_core/core_map_helper.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/core_map_helper.py)

## Functions
- `_empty_out_ref(out_type)` — [`L29`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/core_map_helper.py#L29)
- `kernel(body, *, out_type, mesh, scratch_types=(), compiler_params=None, interpret=False, cost_estimate=None, debug=False, name=None, metadata=None)` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/sparse_core/core_map_helper.py#L37) — Drop-in replacement for ``pl.kernel`` that lowers via ``core_map``.

