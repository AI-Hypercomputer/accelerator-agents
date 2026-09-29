---
title: 'Module: src/maxtext/utils/pipeline_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/pipeline_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.pipeline_utils`/
symbols:
  create_gradient_accumulation_scan: create_gradient_accumulation_scan().
  create_pipeline_stage.execute_pipeline_stage_flax: create_pipeline_stage().execute_pipeline_stage_flax().
  create_gradient_accumulation_scan.run_single_microbatch_custom: create_gradient_accumulation_scan().run_single_microbatch_custom().
  create_gradient_accumulation_scan.run_single_microbatch_custom_fwd: create_gradient_accumulation_scan().run_single_microbatch_custom_fwd().
  create_gradient_accumulation_scan.run_pipeline_microbatches_custom: create_gradient_accumulation_scan().run_pipeline_microbatches_custom().
  create_gradient_accumulation_scan.run_pipeline_microbatches_custom_fwd: create_gradient_accumulation_scan().run_pipeline_microbatches_custom_fwd().
  create_pipeline_stage.execute_pipeline_stage_flax.execute_pipeline_stage_pure: create_pipeline_stage().execute_pipeline_stage_flax().execute_pipeline_stage_pure().
  PipelineContext: PipelineContext#
  derive_stage_weight_partition_specs._process_pps: derive_stage_weight_partition_specs()._process_pps().
  advance_rng_state._fold_if_rng: advance_rng_state()._fold_if_rng().
  advance_rng_state._fold_if_rng.folded: advance_rng_state()._fold_if_rng().folded().
  get_mesh_axis_dim_indices: get_mesh_axis_dim_indices().
  derive_stage_weight_partition_specs: derive_stage_weight_partition_specs().
  strip_pipeline_repeat_logical_axis: strip_pipeline_repeat_logical_axis().
  remove_logically_partition: remove_logically_partition().
  create_pipeline_stage: create_pipeline_stage().
  advance_rng_state: advance_rng_state().
  flatten_nnx_state: flatten_nnx_state().
  PipelineContext.__init__: PipelineContext#__init__().
  create_pipeline_stage.execute_pipeline_stage_flax.execute_pipeline_stage_pure_fwd: create_pipeline_stage().execute_pipeline_stage_flax().execute_pipeline_stage_pure_fwd().
  get_mesh_axis_dim_indices.find_axis_index: get_mesh_axis_dim_indices().find_axis_index().
  remove_gathered_mesh_axes: remove_gathered_mesh_axes().
  strip_pipeline_repeat_logical_axis._remove_from_spec: strip_pipeline_repeat_logical_axis()._remove_from_spec().
  remove_logically_partition._remove_logically_partition_leaf: remove_logically_partition()._remove_logically_partition_leaf().
  create_gradient_accumulation_scan.run_single_microbatch_custom_fwd._run: create_gradient_accumulation_scan().run_single_microbatch_custom_fwd()._run().
  create_gradient_accumulation_scan.run_single_microbatch_custom_bwd: create_gradient_accumulation_scan().run_single_microbatch_custom_bwd().
  create_gradient_accumulation_scan.run_pipeline_microbatches_custom_bwd: create_gradient_accumulation_scan().run_pipeline_microbatches_custom_bwd().
  create_pipeline_stage.execute_pipeline_stage_flax.execute_pipeline_stage_pure_bwd: create_pipeline_stage().execute_pipeline_stage_flax().execute_pipeline_stage_pure_bwd().
  flatten_nnx_state._is_var: flatten_nnx_state()._is_var().
  create_flax_pipeline_scan: create_flax_pipeline_scan().
  is_static_param: is_static_param().
  is_spec_leaf: is_spec_leaf().
  unflatten_nnx_state: unflatten_nnx_state().
  arrays_to_linen_collection: arrays_to_linen_collection().
  linen_collection_to_arrays: linen_collection_to_arrays().
  PipelineContext.__slots__: PipelineContext#__slots__.
---
# Module: [`src/maxtext/utils/pipeline_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py)

## Classes
### `PipelineContext`
- def: [`src/maxtext/utils/pipeline_utils.py:528`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L528)
- doc: Non-pytree wrapper holding pipeline methods + Python config.
- signature: `class PipelineContext:`
- protocol/private: `__init__`[`L543`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L543), `__slots__`[`L536`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L536)

## Functions
- `_fold_if_rng(x)` — [`L437`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L437)
- `_is_var(x)` — [`L466`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L466)
- `_process_pps(path, pps)` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L77)
- `_remove_from_spec(spec)` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L147)
- `_remove_logically_partition_leaf(v)` — [`L157`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L157)
- `_run(l, b)` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L203)
- `advance_rng_state(state, iteration)` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L414) — Fold loop_iteration into all RNG keys to produce unique dropout masks per scan step.
- `arrays_to_linen_collection(arrays, keys)` — [`L504`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L504) — Convert list of arrays + key names to a Linen-style flat dict.
- `create_flax_pipeline_scan(pipeline_stage_fn, length, remat_policy, use_scan=True)` — [`L365`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L365) — Wraps the pipeline stage execution in `flax.linen.remat` and `flax.linen.scan`.
- `create_gradient_accumulation_scan(model, length, deterministic=True, model_mode=None, logical_partition_spec=None)` — [`L167`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L167) — Creates a memory-efficient `jax.lax.scan` loop for pipeline microbatches with a custom VJP.
- `create_pipeline_stage(length, deterministic, model_mode, logical_partition_spec, physical_partition_spec, positions, segment_ids)` — [`L252`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L252) — Builds an execution block for a single pipeline stage.
- `derive_stage_weight_partition_specs(physical_partition_spec, axes_to_remove)` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L59) — Derives the physical partition specs for weights inside the scanned pipeline loop.
- `execute_pipeline_stage_flax(model, carry)` — [`L281`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L281) — A non-pure Flax closure of the pipeline stage.
- `execute_pipeline_stage_pure(loop_state, w_curr, pipeline_weights)` — [`L308`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L308)
- `execute_pipeline_stage_pure_bwd(residuals, g_outputs)` — [`L340`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L340)
- `execute_pipeline_stage_pure_fwd(loop_state, w_curr, pipeline_weights)` — [`L311`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L311)
- `find_axis_index(pspec)` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L43)
- `flatten_nnx_state(state)` — [`L458`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L458) — Flatten nnx.State to (arrays, treedef, is_var_flags, var_types, var_metadata).
- `folded(k)` — [`L442`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L442)
- `get_mesh_axis_dim_indices(physical_partition_spec, axis_name="fsdp")` — [`L26`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L26) — Finds the tensor dimension index sharded across a specific physical mesh axis.
- `is_spec_leaf(x)` — [`L453`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L453) — Predicate matching leaves in the bsw_pps treedef, which can be either P or None (if no sharding).
- `is_static_param(path, v)` — [`L405`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L405) — Predicate matching nnx.Param and FP8 _overwrite_with_gradient variables.
- `linen_collection_to_arrays(collection, keys)` — [`L512`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L512) — Extract arrays from Linen-style flat dict in key order.
- `remove_gathered_mesh_axes(pps, is_moe_block_0, axes_to_remove)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L93) — Strips FSDP and specific MoE mesh axes from a PartitionSpec.
- `remove_logically_partition(weights)` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L154) — Removes LogicallyPartitioned wrapper from weights.
- `run_pipeline_microbatches_custom(loop_state, bsw, positions, segment_ids)` — [`L222`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L222)
- `run_pipeline_microbatches_custom_bwd(residuals, g_final_state)` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L239)
- `run_pipeline_microbatches_custom_fwd(loop_state, bsw, positions, segment_ids)` — [`L225`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L225)
- `run_single_microbatch_custom(lightweight_state, bsw, pos_arg, seg_arg)` — [`L199`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L199)
- `run_single_microbatch_custom_bwd(res, g_out)` — [`L214`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L214)
- `run_single_microbatch_custom_fwd(lightweight_state, bsw, pos_arg, seg_arg)` — [`L202`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L202)
- `strip_pipeline_repeat_logical_axis(full_logical_spec)` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L135) — Removes 'circular_repeats' from a logical PartitionSpec PyTree.
- `unflatten_nnx_state(arrays, treedef, is_var_flags, var_types, var_metadata)` — [`L488`](../../../../../../../raw/code/maxtext/src/maxtext/utils/pipeline_utils.py#L488) — Reconstruct nnx.State from flattened arrays + metadata.

