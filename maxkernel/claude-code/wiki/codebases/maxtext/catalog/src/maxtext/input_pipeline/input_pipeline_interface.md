---
title: 'Module: src/maxtext/input_pipeline/input_pipeline_interface.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/input_pipeline_interface.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.input_pipeline_interface`/
symbols:
  create_data_iterator: create_data_iterator().
  get_process_loading_real_data: get_process_loading_real_data().
  create_process_specific_iterator: create_process_specific_iterator().
---
# Module: [`src/maxtext/input_pipeline/input_pipeline_interface.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_interface.py)

## Functions
- `create_data_iterator(config: pyconfig.HyperParameters, mesh)` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_interface.py#L62) — Create train and eval data iterators given configs and mesh.
- `create_process_specific_iterator(config: pyconfig.HyperParameters, mesh, process_indices, input_iterator)` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_interface.py#L49) — If the current process's index is among the `process_indices`, a real
- `get_process_loading_real_data(data_sharding, global_batch_size_to_load, global_batch_size_to_train_on, max_target_length, mesh)` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/input_pipeline_interface.py#L34) — Get list of processes loading data from GCS when expansion_factor_real_data != -1

