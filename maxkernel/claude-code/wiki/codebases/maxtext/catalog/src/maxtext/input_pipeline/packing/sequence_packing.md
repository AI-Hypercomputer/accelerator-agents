---
title: 'Module: src/maxtext/input_pipeline/packing/sequence_packing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/packing/sequence_packing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.packing.sequence_packing`/
symbols:
  pack_dataset: pack_dataset().
  _pack_with_tf_ops: _pack_with_tf_ops().
  _pack_with_tf_ops.map_fn: _pack_with_tf_ops().map_fn().
  _pack_with_tf_ops.map_fn.body_fn: _pack_with_tf_ops().map_fn().body_fn().
  _pack_with_tf_ops.map_fn.body_fn.false_fn: _pack_with_tf_ops().map_fn().body_fn().false_fn().
  AUTOTUNE: AUTOTUNE.
  _pack_with_tf_ops.write_packed_example: _pack_with_tf_ops().write_packed_example().
  pack_dataset.my_fn: pack_dataset().my_fn().
  _pack_with_tf_ops.map_fn.body_fn.true_fn: _pack_with_tf_ops().map_fn().body_fn().true_fn().
---
# Module: [`src/maxtext/input_pipeline/packing/sequence_packing.py`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py)

## Functions
- `_pack_with_tf_ops(dataset: tf.data.Dataset, keys: list[str], key2length: dict[str, int], pad_id: int)` — [`L96`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L96) — Helper-function for packing a dataset which has already been batched.
- `body_fn(i, partial, outputs)` — [`L142`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L142) — Body function for while_loop.
- `false_fn()` — [`L163`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L163)
- `map_fn(x)` — [`L125`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L125) — Internal function to flat_map over.
- `my_fn(x)` — [`L90`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L90)
- `pack_dataset(dataset: tf.data.Dataset, key2length: int | dict[str, int], pad_id: int, keys: None | list[str] = None)` — [`L22`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L22) — Creates a 'packed' version of a dataset on-the-fly.
- `true_fn()` — [`L166`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L166)
- `write_packed_example(partial, outputs)` — [`L114`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L114)

## Module values
- `AUTOTUNE` — [`L19`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/sequence_packing.py#L19)

