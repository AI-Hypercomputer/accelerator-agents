---
title: 'Module: src/maxtext/input_pipeline/olmo_grain_data_processing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/olmo_grain_data_processing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.olmo_grain_data_processing`/
symbols:
  _make_loader_for_host: _make_loader_for_host().
  make_olmo_grain_train_iterator: make_olmo_grain_train_iterator().
  make_olmo_grain_eval_iterator: make_olmo_grain_eval_iterator().
  _build_path_remap: _build_path_remap().
  _detect_resumed_step: _detect_resumed_step().
---
# Module: [`src/maxtext/input_pipeline/olmo_grain_data_processing.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_grain_data_processing.py)

## Functions
- `_build_path_remap(config)` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_grain_data_processing.py#L56)
- `_detect_resumed_step(config)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_grain_data_processing.py#L66) — Return the step number of the latest checkpoint, or 0 for a fresh run.
- `_make_loader_for_host(config, *, process_indices: List[int], seed: int)` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_grain_data_processing.py#L87) — Construct an OLMo grain DataLoader for the current data-loading host. — documented in [maxtext-input_pipeline-olmo_data](../../../../concepts/maxtext-input_pipeline-olmo_data.md)
- `make_olmo_grain_eval_iterator(config, global_mesh, process_indices)` — [`L168`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_grain_data_processing.py#L168) — Eval iterator for ``dataset_type=olmo_grain``.
- `make_olmo_grain_train_iterator(config, global_mesh, process_indices)` — [`L148`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/olmo_grain_data_processing.py#L148) — Train iterator for ``dataset_type=olmo_grain``.

