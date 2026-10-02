---
title: 'Module: tpu_inference/kernels/gdn/v3/metadata.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v3/metadata.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v3.metadata`/compute_
symbols:
  compute_batched_seq_metadata: batched_seq_metadata().
  compute_per_seq_metadata: per_seq_metadata().
---
# Module: [`tpu_inference/kernels/gdn/v3/metadata.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/metadata.py)

## Functions
- `compute_batched_seq_metadata(cfg: config.GDNConfig, seq_lens: jax.Array, query_start_loc: jax.Array, state_indices: jax.Array, end_seq: jax.Array)` — [`L22`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/metadata.py#L22) — Metadata for computing multiple sequences per tile.
- `compute_per_seq_metadata(cfg: config.GDNConfig, seq_lens: jax.Array, query_start_loc: jax.Array, state_indices: jax.Array, start_seq: jax.Array, end_seq: jax.Array)` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/metadata.py#L55) — Metadata for computing single sequence per tile.

