---
title: 'Module: tpu_inference/kernels/gdn/v2/compute_schedule_v2.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v2/compute_schedule_v2.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v2.compute_schedule_v2`/compute_schedule_table_v2().
symbols:
  compute_schedule_table_v2: ''
---
# Module: [`tpu_inference/kernels/gdn/v2/compute_schedule_v2.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/compute_schedule_v2.py)

## Functions
- `compute_schedule_table_v2(query_start_loc: jax.Array, decode_tokens: int | jax.Array, num_valid_seqs: int | jax.Array, max_tokens: int, chunk_size: int, BT: int | None=None, alignment: int=8)` — [`L19`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v2/compute_schedule_v2.py#L19) — Compute number of iterations in grid and work each iteration will do

