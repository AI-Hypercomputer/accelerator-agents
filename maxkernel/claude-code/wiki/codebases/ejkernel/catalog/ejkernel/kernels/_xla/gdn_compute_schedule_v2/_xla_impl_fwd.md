---
title: 'Module: ejkernel/kernels/_xla/gdn_compute_schedule_v2/_xla_impl_fwd.py'
type: catalog
provenance: extracted
module: ejkernel/kernels/_xla/gdn_compute_schedule_v2/_xla_impl_fwd.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.kernels._xla.gdn_compute_schedule_v2._xla_impl_fwd`/compute_schedule_table_v2().
symbols:
  compute_schedule_table_v2: ''
---
# Module: [`ejkernel/kernels/_xla/gdn_compute_schedule_v2/_xla_impl_fwd.py`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gdn_compute_schedule_v2/_xla_impl_fwd.py)

## Functions
- `compute_schedule_table_v2(query_start_loc: jax.Array, decode_tokens: int | jax.Array, num_valid_seqs: int | jax.Array, max_tokens: int, chunk_size: int, BT: int | None = None, alignment: int = 8)` — [`L36`](../../../../../../../../raw/code/ejkernel/ejkernel/kernels/_xla/gdn_compute_schedule_v2/_xla_impl_fwd.py#L36) — Build the per-iteration work schedule table for ragged GDN v2 kernels.

