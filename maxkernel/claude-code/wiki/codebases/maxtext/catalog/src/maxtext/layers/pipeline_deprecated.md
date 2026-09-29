---
title: 'Module: src/maxtext/layers/pipeline_deprecated.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/pipeline_deprecated.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.pipeline_deprecated`/Pipeline#
symbols:
  Pipeline.config: config.
  Pipeline.__call__: __call__().
  Pipeline.run_one_iteration: run_one_iteration().
  Pipeline.init_states: init_states().
  Pipeline.num_stages: num_stages.
  Pipeline.mesh: mesh.
  Pipeline.get_iteration_inputs: get_iteration_inputs().
  Pipeline.get_new_loop_state: get_new_loop_state().
  Pipeline.shard_dim_by_stages: shard_dim_by_stages().
  Pipeline.stages_in_spec: stages_in_spec.
  Pipeline.output_sharding: output_sharding.
  Pipeline._maybe_shard_with_logical: _maybe_shard_with_logical().
  Pipeline._update_shift: _update_shift().
  Pipeline.stages_in_logical: stages_in_logical.
  Pipeline.input_sharding: input_sharding.
  Pipeline._update_state_io: _update_state_io().
  Pipeline.get_current_repeat_from_stages: get_current_repeat_from_stages().
  Pipeline.get_vmap_func_for_init: get_vmap_func_for_init().
  Pipeline.get_main_vmap_func_for_iterations: get_main_vmap_func_for_iterations().
  Pipeline.all_gather_over_fsdp: all_gather_over_fsdp().
  Pipeline.state_io_spec: state_io_spec.
  Pipeline.get_microbatch_and_repeat_ids: get_microbatch_and_repeat_ids().
  Pipeline.state_io_sharding: state_io_sharding.
  Pipeline.pipeline_microbatch_size: pipeline_microbatch_size.
  Pipeline.stages_in_sharding: stages_in_sharding.
  Pipeline.need_circ_storage: need_circ_storage().
  Pipeline._rotate_right_and_update: _rotate_right_and_update().
  Pipeline.prepare_vars_for_main_vmap: prepare_vars_for_main_vmap().
  Pipeline.use_circ_storage: use_circ_storage.
  Pipeline.state_io_logical: state_io_logical.
  Pipeline.iterations_to_complete_first_microbatch_one_repeat: iterations_to_complete_first_microbatch_one_repeat().
  Pipeline.vmap_parallel_gather: vmap_parallel_gather().
  Pipeline.vmap_gather: vmap_gather().
  Pipeline._rotate_right: _rotate_right().
  Pipeline.forwarding_delay: forwarding_delay.
  Pipeline._maybe_shard_with_name: _maybe_shard_with_name().
  Pipeline.gather_weights_for_stages_in: gather_weights_for_stages_in().
  Pipeline.microbatches_per_stage: microbatches_per_stage.
  Pipeline.iterations_to_complete_first_microbatch: iterations_to_complete_first_microbatch().
  Pipeline._shift_right: _shift_right().
  Pipeline.permute_output_micro_per_stage_dim: permute_output_micro_per_stage_dim().
  Pipeline.get_current_stage_weights: get_current_stage_weights().
  Pipeline.get_pipeline_remat_policy: get_pipeline_remat_policy().
  Pipeline.get_partition_spec: get_partition_spec().
  Pipeline._remove_logically_partition: _remove_logically_partition().
  Pipeline.spmd_axis_name: spmd_axis_name.
  Pipeline._gather_one: _gather_one().
  Pipeline.func_to_vmap: func_to_vmap().
  Pipeline.batch_axis_name: batch_axis_name.
  Pipeline.seq_len_axis_name: seq_len_axis_name.
  Pipeline.run_iteration_scannable: run_iteration_scannable().
  Pipeline.select_state_or_input: select_state_or_input().
  Pipeline._shift_left: _shift_left().
  Pipeline.get_logical_spec_repeats_removed: get_logical_spec_repeats_removed().
  Pipeline.layers: layers.
  Pipeline.remat_policy: remat_policy.
  Pipeline.get_weight_sharding: get_weight_sharding().
  Pipeline._remove_fsdp_from_physical_partition_spec: _remove_fsdp_from_physical_partition_spec().
  Pipeline._rotate_left: _rotate_left().
  Pipeline._is_leaf: _is_leaf().
  Pipeline.get_partition_spec_leaf: get_partition_spec_leaf().
  Pipeline._remove_from_spec: _remove_from_spec().
  Pipeline._remove_logically_partition_leaf: _remove_logically_partition_leaf().
  Pipeline: ''
  Pipeline.setup: setup().
---
# Module: [`src/maxtext/layers/pipeline_deprecated.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py)

## Classes
### `Pipeline`  ·  implements/extends Module
- def: [`src/maxtext/layers/pipeline_deprecated.py:41`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L41)
- doc: Module that implements pipelining across stages.
- signature: `class Pipeline(nn.Module):`
- members:
  - `__call__(self, inputs: jnp.ndarray, segment_ids: jnp.ndarray, positions: jnp.ndarray, deterministic: bool, model_mode=MODEL_MODE_TRAIN, logical_partition_spec=None)` — [`L775`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L775) — The main method that maps the series of decoder layer inputs to final layer outputs. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `_maybe_shard_with_logical(self, inputs, logical_axes)` — [`L123`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L123) — Wrapper of maybe_shard_with_logical
  - `_maybe_shard_with_name(self, inputs, sharding_name)` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L135) — Wrapper of maybe_shard_with_name
  - `_remove_fsdp_from_physical_partition_spec(pps)` — [`L737`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L737) — Removes 'fsdp' and 'fsdp_transpose' from a physical PartitionSpec.
  - `all_gather_over_fsdp(self, variables, logical_partition_spec)` — [`L761`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L761) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `func_to_vmap(body_instance, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)` — [`L532`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L532) — nn.vmap requires either a nn.module class or a function whose first argument is a nn.module instance.
  - `gather_weights_for_stages_in(w, spec=None)` — [`L514`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L514) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None)` — [`L497`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L497) — get current repeat from stages — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None)` — [`L483`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L483) — Gets the current weights used for one iteration. Outputs a pytree whose arrays have leading dimension of stages, e.g. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift)` — [`L209`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L209) — Construct stages_in: the global array that is operated on for this iteration, shape same as — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_logical_spec_repeats_removed(full_logical)` — [`L715`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L715)
  - `get_main_vmap_func_for_iterations(self)` — [`L551`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L551) — Returns main stage function vmapped by number of stages. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_microbatch_and_repeat_ids(self, loop_iteration)` — [`L280`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L280) — Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_new_loop_state(self, output, loop_state)` — [`L350`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L350) — Update the various buffers given the output of the most recent iteration — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_partition_spec(pytree)` — [`L700`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L700)
  - `get_partition_spec_leaf(leaf)` — [`L704`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L704)
  - `get_pipeline_remat_policy(self)` — [`L679`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L679) — Returns the pipeline remat policy for this pipeline. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_vmap_func_for_init(self)` — [`L529`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L529) — This vmap func is used to initialize the weights only on init. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `get_weight_sharding(self, *init_args)` — [`L693`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L693) — get weight sharding function for this pipeline.
  - `init_states(self, inputs)` — [`L144`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L144) — Initialize components of state: state_io, shift, circular_storage and circular_storage_mover — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `iterations_to_complete_first_microbatch(self)` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L116) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `iterations_to_complete_first_microbatch_one_repeat(self)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L112)
  - `need_circ_storage(self)` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L106) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `permute_output_micro_per_stage_dim(self, output)` — [`L472`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L472) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `prepare_vars_for_main_vmap(weights, physical_partition_spec=None)` — [`L624`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L624) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `run_iteration_scannable(model, loop_state, xs)` — [`L907`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L907) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `run_one_iteration(self, loop_state, pipeline_weights, positions, segment_ids, deterministic, model_mode, decoder_layer_instance, logical_partition_spec=None)` — [`L589`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L589) — Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel, — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `select_state_or_input(first_stage_in, shift)` — [`L243`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L243) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `setup(self)` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L62)
  - `shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False)` — [`L256`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L256) — Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `vmap_gather(self, xs, ids, ids_dim)` — [`L325`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L325) — Use vmap to implement a stage-wise sharded gather. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `vmap_parallel_gather(self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights)` — [`L289`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L289) — Use vmap to implement a sharded parallel gather. — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `batch_axis_name` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L70) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `config` — [`L52`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L52)
  - `forwarding_delay` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L64) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `input_sharding` — [`L87`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L87) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `layers` — [`L56`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L56)
  - `mesh` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L58)
  - `microbatches_per_stage` — [`L67`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L67)
  - `num_stages` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L63) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `output_sharding` — [`L96`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L96) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `pipeline_microbatch_size` — [`L65`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L65)
  - `remat_policy` — [`L60`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L60) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `seq_len_axis_name` — [`L71`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L71) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `spmd_axis_name` — [`L74`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L74) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `stages_in_logical` — [`L76`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L76) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `stages_in_sharding` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L78) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `stages_in_spec` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L77) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `state_io_logical` — [`L82`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L82) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `state_io_sharding` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L84)
  - `state_io_spec` — [`L83`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L83) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
  - `use_circ_storage` — [`L68`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L68) — documented in [maxtext-layers-pipeline_deprecated](../../../../concepts/maxtext-layers-pipeline_deprecated.md)
- protocol/private: `_gather_one`[`L307`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L307), `_is_leaf`[`L701`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L701), `_remove_from_spec`[`L719`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L719), `_remove_logically_partition`[`L726`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L726), `_remove_logically_partition_leaf`[`L727`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L727), `_rotate_left`[`L437`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L437), `_rotate_right`[`L375`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L375), `_rotate_right_and_update`[`L416`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L416), `_shift_left`[`L443`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L443), `_shift_right`[`L388`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L388), `_update_shift`[`L399`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L399), `_update_state_io`[`L453`](../../../../../../../raw/code/maxtext/src/maxtext/layers/pipeline_deprecated.py#L453)

