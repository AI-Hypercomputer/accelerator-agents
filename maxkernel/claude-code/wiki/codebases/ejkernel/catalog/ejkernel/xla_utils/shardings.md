---
title: 'Module: ejkernel/xla_utils/shardings.py'
type: catalog
provenance: extracted
module: ejkernel/xla_utils/shardings.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.xla_utils.shardings`/
symbols:
  get_corrected_named_sharding: get_corrected_named_sharding().
  reorder_sequence: reorder_sequence().
---
# Module: [`ejkernel/xla_utils/shardings.py`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/shardings.py)

## Functions
- `get_corrected_named_sharding(shape: tuple[int, ...], partition_spec: PartitionSpec, mesh: Mesh)` — [`L68`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/shardings.py#L68) — Create a valid NamedSharding by correcting PartitionSpec based on shape and mesh.
- `reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False)` — [`L181`](../../../../../../raw/code/ejkernel/ejkernel/xla_utils/shardings.py#L181) — Reorder sequence dimension for ring attention communication patterns.

