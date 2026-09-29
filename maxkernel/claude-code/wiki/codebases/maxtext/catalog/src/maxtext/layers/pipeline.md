---
title: 'Module: src/maxtext/layers/pipeline.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/pipeline.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.pipeline`/
symbols:
  PipelineBase.config: PipelineBase#config.
  NNXCircularPipeline.__call__: NNXCircularPipeline#__call__().
  NNXPipeline.__call__: NNXPipeline#__call__().
  NNXCircularPipeline.run_one_iteration: NNXCircularPipeline#run_one_iteration().
  PipelineBase.mesh: PipelineBase#mesh.
  NNXCircularPipeline._stage_fn_for_scope: NNXCircularPipeline#_stage_fn_for_scope().
  NNXPipeline.run_one_iteration: NNXPipeline#run_one_iteration().
  PipelineBase.init_states: PipelineBase#init_states().
  PipelineBase.get_microbatch_and_repeat_ids: PipelineBase#get_microbatch_and_repeat_ids().
  PipelineBase.shard_dim_by_stages: PipelineBase#shard_dim_by_stages().
  PipelineBase.get_iteration_inputs: PipelineBase#get_iteration_inputs().
  PipelineBase.get_new_loop_state: PipelineBase#get_new_loop_state().
  PipelineBase.num_stages: PipelineBase#num_stages.
  PipelineBase.stages_in_spec: PipelineBase#stages_in_spec.
  PipelineBase.__init__: PipelineBase#__init__().
  PipelineBase.microbatches_per_stage: PipelineBase#microbatches_per_stage.
  PipelineBase.input_sharding: PipelineBase#input_sharding.
  PipelineBase.output_sharding: PipelineBase#output_sharding.
  PipelineBase._maybe_shard_with_logical: PipelineBase#_maybe_shard_with_logical().
  PipelineBase._update_shift: PipelineBase#_update_shift().
  PipelineBase.stages_in_logical: PipelineBase#stages_in_logical.
  PipelineBase._update_state_io: PipelineBase#_update_state_io().
  PipelineBase.realign_output_microbatches: PipelineBase#realign_output_microbatches().
  PipelineBase._gather_leaf: PipelineBase#_gather_leaf().
  NNXCircularPipeline.get_current_weights_from_bsw: NNXCircularPipeline#get_current_weights_from_bsw().
  PipelineBase.state_io_spec: PipelineBase#state_io_spec.
  PipelineBase._maybe_shard_with_name: PipelineBase#_maybe_shard_with_name().
  PipelineBase.state_io_sharding: PipelineBase#state_io_sharding.
  PipelineBase.pipeline_microbatch_size: PipelineBase#pipeline_microbatch_size.
  PipelineBase.stages_in_sharding: PipelineBase#stages_in_sharding.
  PipelineBase.need_circ_storage: PipelineBase#need_circ_storage().
  PipelineBase._rotate_right_and_update: PipelineBase#_rotate_right_and_update().
  PipelineBase.get_main_vmap_func_for_iterations: PipelineBase#get_main_vmap_func_for_iterations().
  NNXPipeline.get_current_stage_weights: NNXPipeline#get_current_stage_weights().
  NNXCircularPipeline.from_all_variables_to_repeat_weights: NNXCircularPipeline#from_all_variables_to_repeat_weights().
  PipelineBase.state_io_logical: PipelineBase#state_io_logical.
  PipelineBase.forwarding_delay: PipelineBase#forwarding_delay.
  PipelineBase.spmd_axis_name: PipelineBase#spmd_axis_name.
  PipelineBase.get_current_stage_weights: PipelineBase#get_current_stage_weights().
  PipelineBase: PipelineBase#
  PipelineBase.iterations_to_complete_first_microbatch_one_repeat: PipelineBase#iterations_to_complete_first_microbatch_one_repeat().
  PipelineBase.iterations_to_complete_first_microbatch: PipelineBase#iterations_to_complete_first_microbatch().
  PipelineBase.get_pipeline_remat_policy: PipelineBase#get_pipeline_remat_policy().
  PipelineBase.vmap_gather: PipelineBase#vmap_gather().
  PipelineBase._rotate_right: PipelineBase#_rotate_right().
  PipelineBase.use_circ_storage: PipelineBase#use_circ_storage.
  NNXPipeline.scan_body: NNXPipeline#scan_body().
  PipelineBase.vmap_parallel_gather: PipelineBase#vmap_parallel_gather().
  PipelineBase._shift_right: PipelineBase#_shift_right().
  PipelineBase.permute_output_micro_per_stage_dim: PipelineBase#permute_output_micro_per_stage_dim().
  PipelineBase.all_gather_over_fsdp: PipelineBase#all_gather_over_fsdp().
  PipelineBase.get_logical_spec_repeats_removed: PipelineBase#get_logical_spec_repeats_removed().
  NNXPipeline.get_current_repeat_from_stages: NNXPipeline#get_current_repeat_from_stages().
  NNXPipeline._scatter_update: NNXPipeline#_scatter_update().
  NNXCircularPipeline.get_main_vmap_func_for_iterations: NNXCircularPipeline#get_main_vmap_func_for_iterations().
  NNXCircularPipeline._gather_one: NNXCircularPipeline#_gather_one().
  NNXCircularPipeline.from_repeat_weights_to_bsw: NNXCircularPipeline#from_repeat_weights_to_bsw().
  NNXCircularPipeline._shard_map_gather_weights: NNXCircularPipeline#_shard_map_gather_weights().
  NNXCircularPipeline._apply_sharding_hint: NNXCircularPipeline#_apply_sharding_hint().
  NNXCircularPipeline._scatter_update_mutables: NNXCircularPipeline#_scatter_update_mutables().
  PipelineBase.remat_policy: PipelineBase#remat_policy.
  PipelineBase._stamp_at_current_trace: PipelineBase#_stamp_at_current_trace().
  NNXPipeline: NNXPipeline#
  NNXPipeline.gather_weights_for_stages_in: NNXPipeline#gather_weights_for_stages_in().
  NNXCircularPipeline: NNXCircularPipeline#
  NNXCircularPipeline.gather_weights_across_stages_vmap: NNXCircularPipeline#gather_weights_across_stages_vmap().
  NNXCircularPipeline.select_weights_from_bsw: NNXCircularPipeline#select_weights_from_bsw().
  PipelineBase.get_weight_sharding: PipelineBase#get_weight_sharding().
  PipelineBase.layers: PipelineBase#layers.
  NNXCircularPipeline.weight_prefetching: NNXCircularPipeline#weight_prefetching().
  create_nnx_pipeline: create_nnx_pipeline().
  create_pipeline: create_pipeline().
  PipelineBase._gather_one: PipelineBase#_gather_one().
  NNXCircularPipeline.gather_microbatch_inputs_vmap: NNXCircularPipeline#gather_microbatch_inputs_vmap().
  NNXCircularPipeline._run_iter_local: NNXCircularPipeline#_run_iter_local().
  NNXCircularPipeline._run_iter_local_fwd: NNXCircularPipeline#_run_iter_local_fwd().
  NNXCircularPipeline._run_microbatches: NNXCircularPipeline#_run_microbatches().
  NNXCircularPipeline._run_microbatches_fwd: NNXCircularPipeline#_run_microbatches_fwd().
  NNXCircularPipeline.inner_body: NNXCircularPipeline#inner_body().
  NNXCircularPipeline._execute_stage: NNXCircularPipeline#_execute_stage().
  NNXCircularPipeline._execute_stage_fwd: NNXCircularPipeline#_execute_stage_fwd().
  NNXCircularPipeline._run_iter_bubble: NNXCircularPipeline#_run_iter_bubble().
  NNXCircularPipeline._run_iter_bubble_fwd: NNXCircularPipeline#_run_iter_bubble_fwd().
  NNXCircularPipeline.bubble_body: NNXCircularPipeline#bubble_body().
  PipelineBase.batch_axis_name: PipelineBase#batch_axis_name.
  PipelineBase.seq_len_axis_name: PipelineBase#seq_len_axis_name.
  PipelineBase.select_state_or_input: PipelineBase#select_state_or_input().
  PipelineBase._shift_left: PipelineBase#_shift_left().
  NNXPipeline.unbox_val: NNXPipeline#unbox_val().
  NNXCircularPipeline._from_repeat_weights_to_bsw_shardmap: NNXCircularPipeline#_from_repeat_weights_to_bsw_shardmap().
  NNXCircularPipeline._gather_tensor_along_axes: NNXCircularPipeline#_gather_tensor_along_axes().
  NNXCircularPipeline._from_repeat_weights_to_bsw_hint: NNXCircularPipeline#_from_repeat_weights_to_bsw_hint().
  NNXCircularPipeline.fetch_active_stage_weights: NNXCircularPipeline#fetch_active_stage_weights().
  NNXCircularPipeline.unbox_val: NNXCircularPipeline#unbox_val().
  NNXCircularPipeline.scan_fn: NNXCircularPipeline#scan_fn().
  NNXCircularPipeline._full_stage: NNXCircularPipeline#_full_stage().
  Pipeline: Pipeline.
  CircularPipeline: CircularPipeline.
  PipelineBase._remove_fsdp_from_physical_partition_spec: PipelineBase#_remove_fsdp_from_physical_partition_spec().
  PipelineBase.build_batched_rngs: PipelineBase#build_batched_rngs().
  NNXPipeline.is_lp: NNXPipeline#is_lp().
  NNXCircularPipeline.is_lp: NNXCircularPipeline#is_lp().
  NNXCircularPipeline._run: NNXCircularPipeline#_run().
  PipelineBase._setup_pipeline_attributes: PipelineBase#_setup_pipeline_attributes().
  PipelineBase._rotate_left: PipelineBase#_rotate_left().
  PipelineBase.get_spec: PipelineBase#get_spec().
  PipelineBase.func_to_vmap: PipelineBase#func_to_vmap().
  PipelineBase._identity_slice: PipelineBase#_identity_slice().
  PipelineBase.is_leaf: PipelineBase#is_leaf().
  PipelineBase._remove_from_spec: PipelineBase#_remove_from_spec().
  PipelineBase.create_stage_fn: PipelineBase#create_stage_fn().
  NNXPipeline._update_one_stage: NNXPipeline#_update_one_stage().
  NNXCircularPipeline.func_to_vmap: NNXCircularPipeline#func_to_vmap().
  NNXCircularPipeline._gather_repeat_leaf: NNXCircularPipeline#_gather_repeat_leaf().
  NNXCircularPipeline.should_skip_gather: NNXCircularPipeline#should_skip_gather().
  NNXCircularPipeline._update_one_stage: NNXCircularPipeline#_update_one_stage().
  NNXCircularPipeline._run_iter_local_bwd: NNXCircularPipeline#_run_iter_local_bwd().
  NNXCircularPipeline._run_microbatches_bwd: NNXCircularPipeline#_run_microbatches_bwd().
  NNXCircularPipeline._execute_stage_bwd: NNXCircularPipeline#_execute_stage_bwd().
  NNXCircularPipeline._run_iter_bubble_bwd: NNXCircularPipeline#_run_iter_bubble_bwd().
---
# Module: [`src/maxtext/layers/pipeline.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py)

## Classes
### `NNXCircularPipeline`  ·  implements/extends PipelineBase
- def: [`src/maxtext/layers/pipeline.py:966`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L966)
- doc: Implements an circular pipeline schedule with asynchronous weight prefetching.
- signature: `class NNXCircularPipeline(PipelineBase):`
- members:
  - `__call__(self, inputs: jnp.ndarray, segment_ids: jnp.ndarray, positions: jnp.ndarray, deterministic: bool, model_mode=MODEL_MODE_TRAIN, logical_partition_spec=None)` — [`L1313`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1313) — Entry point for the Circular Pipeline Module. Sets up microbatch schedules and executes scans. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `_stage_fn_for_scope(scope, carry)` — [`L1435`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1435) — One repeat: prefetch w_next, run microbatch scan, return updated carry. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `bubble_body(inner_carry, _)` — [`L1754`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1754) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `fetch_active_stage_weights(self, bsw, loop_iteration, physical_partition_spec=None)` — [`L1147`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1147) — The module fetches the actively prefetched weights — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `from_all_variables_to_repeat_weights(self, weights_state, loop_iteration)` — [`L1030`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1030) — Gathers weights corresponding to the repeat IDs for current iteration.
  - `from_repeat_weights_to_bsw(self, repeat_weights, physical_partition_spec, axes_to_gather=("fsdp", "fsdp_transpose", "context", "expert"), use_shardmap=False)` — [`L1041`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1041) — Executes the FSDP-like all-gathers to fully materialize a block of weights for the BSW.
  - `func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)` — [`L983`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L983)
  - `gather_microbatch_inputs_vmap(self, xs, ids, ids_dim)` — [`L997`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L997) — Slices out the specific sequence inputs (e.g., positions, segments) for the current microbatch. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `gather_weights_across_stages_vmap(self, weights_state, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights)` — [`L1017`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1017) — Uses jax.vmap to dynamically slice and gather weights for specific pipeline repeats. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `get_current_weights_from_bsw(self, bsw, loop_iteration, physical_partition_spec)` — [`L1153`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1153) — Pulls the fully gathered parameters for the current repeat from the BSW dual-buffer. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `get_main_vmap_func_for_iterations(self)` — [`L976`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L976) — Returns main stage function vmapped by number of stages. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `inner_body(inner_carry, _)` — [`L1517`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1517)
  - `is_lp(x)` — [`L1367`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1367)
  - `run_one_iteration(self, loop_state, bsw, pipeline_weights_graph, layers_metrics, current_layer_mutables, positions, segment_ids, deterministic, model_mode, logical_partition_spec=None)` — [`L1211`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1211) — Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel, — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `scan_fn(scan_loop_state, scan_bsw)` — [`L1513`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1513)
  - `select_weights_from_bsw(bsw_inner, repeat_id)` — [`L1187`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1187)
  - `should_skip_gather(axis_name, path_keys)` — [`L1075`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1075) — Defines specific rule-based exceptions for gathering certain axes.
  - `unbox_val(x)` — [`L1370`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1370)
  - `weight_prefetching(self, weights_state, physical_partition_spec, loop_iteration)` — [`L1138`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1138) — Triggers asynchronous FSDP-like all-gathers for the next pipeline steps.
- protocol/private: `_apply_sharding_hint`[`L1115`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1115), `_execute_stage`[`L1585`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1585), `_execute_stage_bwd`[`L1611`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1611), `_execute_stage_fwd`[`L1592`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1592), `_from_repeat_weights_to_bsw_hint`[`L1114`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1114), `_from_repeat_weights_to_bsw_shardmap`[`L1056`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1056), `_full_stage`[`L1598`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1598), `_gather_one`[`L1006`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1006), `_gather_repeat_leaf`[`L1020`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1020), `_gather_tensor_along_axes`[`L1100`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1100), `_run`[`L1471`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1471), `_run_iter_bubble`[`L1719`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1719), `_run_iter_bubble_bwd`[`L1747`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1747), `_run_iter_bubble_fwd`[`L1726`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1726), `_run_iter_local`[`L1467`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1467), `_run_iter_local_bwd`[`L1489`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1489), `_run_iter_local_fwd`[`L1470`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1470), `_run_microbatches`[`L1506`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1506), `_run_microbatches_bwd`[`L1561`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1561), `_run_microbatches_fwd`[`L1511`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1511), `_scatter_update_mutables`[`L1292`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1292), `_shard_map_gather_weights`[`L1097`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1097), `_update_one_stage`[`L1296`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1296)
- uses (calls/refs, reference-scoped): [`config`](pipeline.md#PipelineBase.config), [`mesh`](pipeline.md#PipelineBase.mesh), [`init_states`](pipeline.md#PipelineBase.init_states), [`get_microbatch_and_repeat_ids`](pipeline.md#PipelineBase.get_microbatch_and_repeat_ids), [`shard_dim_by_stages`](pipeline.md#PipelineBase.shard_dim_by_stages), [`get_iteration_inputs`](pipeline.md#PipelineBase.get_iteration_inputs), [`get_new_loop_state`](pipeline.md#PipelineBase.get_new_loop_state), [`num_stages`](pipeline.md#PipelineBase.num_stages), [`input_sharding`](pipeline.md#PipelineBase.input_sharding), [`output_sharding`](pipeline.md#PipelineBase.output_sharding), [`realign_output_microbatches`](pipeline.md#PipelineBase.realign_output_microbatches), [`_maybe_shard_with_name`](pipeline.md#PipelineBase._maybe_shard_with_name), [`pipeline_microbatch_size`](pipeline.md#PipelineBase.pipeline_microbatch_size), [`forwarding_delay`](pipeline.md#PipelineBase.forwarding_delay), [`spmd_axis_name`](pipeline.md#PipelineBase.spmd_axis_name), [`PipelineBase`](pipeline.md#PipelineBase), [`get_pipeline_remat_policy`](pipeline.md#PipelineBase.get_pipeline_remat_policy), [`_stamp_at_current_trace`](pipeline.md#PipelineBase._stamp_at_current_trace), [`layers`](pipeline.md#PipelineBase.layers), [`_remove_fsdp_from_physical_partition_spec`](pipeline.md#PipelineBase._remove_fsdp_from_physical_partition_spec)
- used by: [`get_main_vmap_func_for_iterations`](pipeline.md#PipelineBase.get_main_vmap_func_for_iterations), [`PipelineBase`](pipeline.md#PipelineBase), [`create_nnx_pipeline`](pipeline.md#create_nnx_pipeline), [`CircularPipeline`](pipeline.md#CircularPipeline)

### `NNXPipeline`  ·  implements/extends PipelineBase
- def: [`src/maxtext/layers/pipeline.py:698`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L698)
- doc: Original Pipeline implementation adapted for NNX.
- signature: `class NNXPipeline(PipelineBase):`
- members:
  - `__call__(self, inputs: jnp.ndarray, segment_ids: jnp.ndarray, positions: jnp.ndarray, deterministic: bool, model_mode=MODEL_MODE_TRAIN, logical_partition_spec=None)` — [`L818`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L818) — The main method that maps the series of decoder layer inputs to final layer outputs. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `gather_weights_for_stages_in(w, spec=None)` — [`L712`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L712)
  - `get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None)` — [`L708`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L708) — Fetches the weights for the current repeat from the stages.
  - `get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None)` — [`L701`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L701)
  - `is_lp(x)` — [`L877`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L877)
  - `run_one_iteration(self, loop_state, pipeline_weights_graph, pipeline_weights_state, positions, segment_ids, deterministic, model_mode, logical_partition_spec=None)` — [`L741`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L741) — Executes the logic for a single microbatch iteration, including routing inputs and weights, and advancing buffers.
  - `scan_body(carry, _)` — [`L915`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L915) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `unbox_val(x)` — [`L880`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L880)
- protocol/private: `_scatter_update`[`L798`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L798), `_update_one_stage`[`L802`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L802)
- uses (calls/refs, reference-scoped): [`config`](pipeline.md#PipelineBase.config), [`mesh`](pipeline.md#PipelineBase.mesh), [`init_states`](pipeline.md#PipelineBase.init_states), [`get_microbatch_and_repeat_ids`](pipeline.md#PipelineBase.get_microbatch_and_repeat_ids), [`shard_dim_by_stages`](pipeline.md#PipelineBase.shard_dim_by_stages), [`get_iteration_inputs`](pipeline.md#PipelineBase.get_iteration_inputs), [`get_new_loop_state`](pipeline.md#PipelineBase.get_new_loop_state), [`num_stages`](pipeline.md#PipelineBase.num_stages), [`input_sharding`](pipeline.md#PipelineBase.input_sharding), [`output_sharding`](pipeline.md#PipelineBase.output_sharding), [`_maybe_shard_with_name`](pipeline.md#PipelineBase._maybe_shard_with_name), [`get_main_vmap_func_for_iterations`](pipeline.md#PipelineBase.get_main_vmap_func_for_iterations), [`pipeline_microbatch_size`](pipeline.md#PipelineBase.pipeline_microbatch_size), [`forwarding_delay`](pipeline.md#PipelineBase.forwarding_delay), [`PipelineBase`](pipeline.md#PipelineBase), [`get_pipeline_remat_policy`](pipeline.md#PipelineBase.get_pipeline_remat_policy), [`vmap_gather`](pipeline.md#PipelineBase.vmap_gather), [`all_gather_over_fsdp`](pipeline.md#PipelineBase.all_gather_over_fsdp), [`get_logical_spec_repeats_removed`](pipeline.md#PipelineBase.get_logical_spec_repeats_removed), [`permute_output_micro_per_stage_dim`](pipeline.md#PipelineBase.permute_output_micro_per_stage_dim), [`vmap_parallel_gather`](pipeline.md#PipelineBase.vmap_parallel_gather), [`_stamp_at_current_trace`](pipeline.md#PipelineBase._stamp_at_current_trace), [`layers`](pipeline.md#PipelineBase.layers)
- used by: [`get_current_stage_weights`](pipeline.md#PipelineBase.get_current_stage_weights), [`PipelineBase`](pipeline.md#PipelineBase), [`create_nnx_pipeline`](pipeline.md#create_nnx_pipeline), [`Pipeline`](pipeline.md#Pipeline)

### `PipelineBase`  ·  implements/extends Module
- def: [`src/maxtext/layers/pipeline.py:45`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L45)
- doc: Base module that implements shared pipelining logic across stages.
- signature: `class PipelineBase(nnx.Module):`
- members:
  - `_maybe_shard_with_logical(self, inputs, logical_axes)` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L110) — Wrapper of maybe_shard_with_logical
  - `_maybe_shard_with_name(self, inputs, sharding_name)` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L122) — Wrapper of maybe_shard_with_name
  - `_remove_fsdp_from_physical_partition_spec(physical_partition_spec)` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L196) — Removes 'fsdp' and 'fsdp_transpose' from physical partition spec.
  - `_setup_pipeline_attributes(self)` — [`L51`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L51) — Initializes the configuration, calculating num_stages, delay, axes, and partition specs.
  - `_stamp_at_current_trace(weights)` — [`L568`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L568) — Pass each leaf through a no-op dynamic_slice so JAX creates new arrays — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `all_gather_over_fsdp(self, variables, logical_partition_spec)` — [`L601`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L601) — all-gathers the variables over fsdp if fsdp is in the logical partition spec. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `build_batched_rngs(shape)` — [`L655`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L655)
  - `create_stage_fn(r)` — [`L668`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L668)
  - `func_to_vmap(graph, state, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)` — [`L546`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L546)
  - `get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None)` — [`L585`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L585) — Gets the current weights used for one iteration. Outputs a pytree whose arrays have leading dimension of stages, e.g.
  - `get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift)` — [`L131`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L131) — Construct stages_in: the global array that is operated on for this iteration, shape same as — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `get_logical_spec_repeats_removed(self, full_logical)` — [`L627`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L627) — Returns a new logical spec with 'circular_repeats' removed.
  - `get_main_vmap_func_for_iterations(self)` — [`L543`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L543) — Returns vmapped function that runs one pipeline iteration across stages.
  - `get_microbatch_and_repeat_ids(self, loop_iteration)` — [`L176`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L176) — Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `get_new_loop_state(self, output, loop_state)` — [`L366`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L366) — Update the various buffers given the output of the most recent iteration — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `get_pipeline_remat_policy(self)` — [`L186`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L186) — Returns the pipeline remat policy for this pipeline. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `get_spec(x)` — [`L505`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L505)
  - `get_weight_sharding(self, *init_args)` — [`L500`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L500) — Returns a pytree of logical-name PartitionSpecs mirroring the params state.
  - `init_states(self, inputs)` — [`L220`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L220) — Initialize components of state: state_io, shift, circular_storage and circular_storage_mover — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `is_leaf(x)` — [`L622`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L622)
  - `iterations_to_complete_first_microbatch(self)` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L103)
  - `iterations_to_complete_first_microbatch_one_repeat(self)` — [`L99`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L99)
  - `need_circ_storage(self)` — [`L93`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L93) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `permute_output_micro_per_stage_dim(self, output)` — [`L473`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L473) — Permutes the output microbatches to match the input order. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `realign_output_microbatches(self, output)` — [`L489`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L489) — Reorders the output tensor to reverse the circular shifts applied during execution. — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `select_state_or_input(first_stage_in, shift)` — [`L164`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L164)
  - `shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False)` — [`L283`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L283) — Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `vmap_gather(self, xs, ids, ids_dim)` — [`L339`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L339) — Use vmap to implement a stage-wise sharded gather.
  - `vmap_parallel_gather(self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights)` — [`L307`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L307) — Use vmap to implement a sharded parallel gather.
  - `batch_axis_name` — [`L59`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L59)
  - `config` — [`L650`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L650)
  - `forwarding_delay` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L54) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `input_sharding` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L74) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `layers` — [`L695`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L695)
  - `mesh` — [`L651`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L651)
  - `microbatches_per_stage` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L56)
  - `num_stages` — [`L53`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L53) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `output_sharding` — [`L83`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L83)
  - `pipeline_microbatch_size` — [`L55`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L55)
  - `remat_policy` — [`L652`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L652)
  - `seq_len_axis_name` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L60)
  - `spmd_axis_name` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L61) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `stages_in_logical` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L63) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `stages_in_sharding` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L65)
  - `stages_in_spec` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L64) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `state_io_logical` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L69) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `state_io_sharding` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L71) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `state_io_spec` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L70) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
  - `use_circ_storage` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L57) — documented in [maxtext-layers-pipeline](../../../../concepts/maxtext-layers-pipeline.md)
- protocol/private: `__init__`[`L641`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L641), `_gather_leaf`[`L608`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L608), `_gather_one`[`L324`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L324), `_identity_slice`[`L578`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L578), `_remove_from_spec`[`L632`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L632), `_rotate_left`[`L440`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L440), `_rotate_right`[`L385`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L385), `_rotate_right_and_update`[`L420`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L420), `_shift_left`[`L445`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L445), `_shift_right`[`L392`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L392), `_update_shift`[`L403`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L403), `_update_state_io`[`L455`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L455)
- uses (calls/refs, reference-scoped): [`get_current_stage_weights`](pipeline.md#NNXPipeline.get_current_stage_weights), [`get_main_vmap_func_for_iterations`](pipeline.md#NNXCircularPipeline.get_main_vmap_func_for_iterations), [`NNXCircularPipeline`](pipeline.md#NNXCircularPipeline), [`NNXPipeline`](pipeline.md#NNXPipeline)
- used by: [`__call__`](pipeline.md#NNXCircularPipeline.__call__), [`__call__`](pipeline.md#NNXPipeline.__call__), [`run_one_iteration`](pipeline.md#NNXCircularPipeline.run_one_iteration), [`run_one_iteration`](pipeline.md#NNXPipeline.run_one_iteration), [`get_current_weights_from_bsw`](pipeline.md#NNXCircularPipeline.get_current_weights_from_bsw), [`from_all_variables_to_repeat_weights`](pipeline.md#NNXCircularPipeline.from_all_variables_to_repeat_weights), [`get_current_stage_weights`](pipeline.md#NNXPipeline.get_current_stage_weights), [`_apply_sharding_hint`](pipeline.md#NNXCircularPipeline._apply_sharding_hint), [`_gather_one`](pipeline.md#NNXCircularPipeline._gather_one), [`_scatter_update`](pipeline.md#NNXPipeline._scatter_update), [`_scatter_update_mutables`](pipeline.md#NNXCircularPipeline._scatter_update_mutables), [`_shard_map_gather_weights`](pipeline.md#NNXCircularPipeline._shard_map_gather_weights), [`get_current_repeat_from_stages`](pipeline.md#NNXPipeline.get_current_repeat_from_stages), [`get_main_vmap_func_for_iterations`](pipeline.md#NNXCircularPipeline.get_main_vmap_func_for_iterations), [`NNXCircularPipeline`](pipeline.md#NNXCircularPipeline), [`NNXPipeline`](pipeline.md#NNXPipeline), [`gather_weights_for_stages_in`](pipeline.md#NNXPipeline.gather_weights_for_stages_in), [`select_weights_from_bsw`](pipeline.md#NNXCircularPipeline.select_weights_from_bsw)

## Functions
- `create_nnx_pipeline(config: Config, stage_factory: Any, mesh: Mesh, remat_policy: Any = None, *, rngs: nnx.Rngs)` — [`L1814`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1814) — Factory function to instantiate the NNX Pipeline module.
- `create_pipeline(config: Config, layers=None, mesh: Mesh = None, remat_policy: Any = None)` — [`L1835`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1835) — Returns the ToLinen-wrapped NNX pipeline appropriate for the config.

## Module values
- `CircularPipeline` — [`L1829`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1829)
- `Pipeline` — [`L1825`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline.py#L1825)

