---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/utils.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/utils.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.utils`/
symbols:
  align_to: align_to().
  get_dtype_packing: get_dtype_packing().
  has_bank_conflicts: has_bank_conflicts().
  broadcast_minor: broadcast_minor().
  strided_load: strided_load().
  strided_store: strided_store().
  convert_to_target_bitwidth: convert_to_target_bitwidth().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/utils.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py)

## Functions
- `align_to(a, b)` — [`L22`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L22) — Returns 'a' aligned to 'b'.
- `broadcast_minor(src, shape)` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L27) — Broadcasts 'src' to 'shape' in the minor dimension.
- `convert_to_target_bitwidth(val, target_bitwidth: int, kv_dtype: jnp.dtype)` — [`L86`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L86) — Converts a value to a target bitwidth.
- `get_dtype_packing(dtype)` — [`L40`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L40)
- `has_bank_conflicts(stride: int, distance=24, num_banks=32)` — [`L128`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L128)
- `strided_load(ref, start_row, num_rows, step, *, dtype=None)` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L44) — Loads data from HBM with strided access, handling 128-lane alignment.
- `strided_store(ref, start, sz, step, val)` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/utils.py#L66) — Stores data to HBM with strided access, handling 128-lane alignment.

