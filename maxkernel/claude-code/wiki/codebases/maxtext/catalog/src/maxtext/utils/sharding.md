---
title: 'Module: src/maxtext/utils/sharding.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/sharding.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.sharding`/
symbols:
  maybe_shard_with_name: maybe_shard_with_name().
  maybe_update_params_sharding_with_opt_nnx: maybe_update_params_sharding_with_opt_nnx().
  maybe_shard_with_logical: maybe_shard_with_logical().
  assert_params_sufficiently_sharded: assert_params_sufficiently_sharded().
  get_nnx_var_named_sharding_with_scan_axis: get_nnx_var_named_sharding_with_scan_axis().
  get_physical_spec_no_fsdp: get_physical_spec_no_fsdp().
  logical_to_mesh_axes: logical_to_mesh_axes().
  create_sharding: create_sharding().
  clear_input_shardings_dump: clear_input_shardings_dump().
  all_gather_over_fsdp: all_gather_over_fsdp().
  maybe_update_params_sharding_with_opt_nnx._extract_param_only: maybe_update_params_sharding_with_opt_nnx()._extract_param_only().
  maybe_update_params_sharding_with_opt_nnx.find_adam_mu: maybe_update_params_sharding_with_opt_nnx().find_adam_mu().
  build_zero1_input_state_mesh_shardings._overlay: build_zero1_input_state_mesh_shardings()._overlay().
  logical_to_mesh: logical_to_mesh().
  logical_to_mesh_sharding: logical_to_mesh_sharding().
  _analyze_sharding: _analyze_sharding().
  remove_fsdp_sharding: remove_fsdp_sharding().
  _LOGGED_ACTIVATION_SHARDINGS: _LOGGED_ACTIVATION_SHARDINGS.
  remove_size_one_mesh_axis: remove_size_one_mesh_axis().
  get_input_data_sharding: get_input_data_sharding().
  maybe_shard_with_pspec: maybe_shard_with_pspec().
  nnx_construct_named_sharding: nnx_construct_named_sharding().
  maybe_update_params_sharding_with_opt: maybe_update_params_sharding_with_opt().
  build_zero1_input_state_mesh_shardings: build_zero1_input_state_mesh_shardings().
  _ACTIVATION_SHARDINGS_DUMP: _ACTIVATION_SHARDINGS_DUMP.
  _get_sharding_desc: _get_sharding_desc().
  get_mesh_axes_used_by_tensor_spec: get_mesh_axes_used_by_tensor_spec().
  _get_nontrival_mesh_axes: _get_nontrival_mesh_axes().
  _raise_if_unsharded_exceeds_tolerance: _raise_if_unsharded_exceeds_tolerance().
  maybe_update_params_sharding_with_opt_nnx._update_model_var: maybe_update_params_sharding_with_opt_nnx()._update_model_var().
  remove_fsdp_sharding._remove_fsdp_from_partition_spec: remove_fsdp_sharding()._remove_fsdp_from_partition_spec().
  add_data_to_sharding: add_data_to_sharding().
  logical_axis_rules_pp_act_as_dp: logical_axis_rules_pp_act_as_dp().
  get_formatted_sharding_annotations: get_formatted_sharding_annotations().
  remove_expert_from_partition_spec: remove_expert_from_partition_spec().
---
# Module: [`src/maxtext/utils/sharding.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py)

## Functions
- `_analyze_sharding(params, mesh, valid_target_mesh_axes)` — [`L358`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L358) — Analyzes parameters to find which are unsharded on any valid mesh axis.
- `_extract_param_only(state)` — [`L612`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L612) — Recursively extract nnx.Param variables from an nnx.State into a nested plain dict.
- `_get_nontrival_mesh_axes(mesh)` — [`L324`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L324) — Returns mesh axes from config that are valid and have more than one shard.
- `_get_sharding_desc(inputs, extra_stack_level)` — [`L54`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L54) — Get the inputs sharding description using inspect module
- `_overlay(model_node, params_node)` — [`L729`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L729)
- `_raise_if_unsharded_exceeds_tolerance(unsharded_size, total_size, tolerance, problematic_tensors_details)` — [`L420`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L420) — Raises an AssertionError if the percentage of unsharded parameters exceeds the given tolerance.
- `_remove_fsdp_from_partition_spec(named_sharding)` — [`L825`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L825) — Removes 'fsdp' and 'fsdp_transpose' from a PartitionSpec.
- `_update_model_var(path, var)` — [`L683`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L683)
- `add_data_to_sharding(mesh, path, aval, sharding)` — [`L504`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L504) — Adds 'data' dimension to sharding spec if compatible and not already present.
- `all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode)` — [`L909`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L909) — Performs an all-gather on FSDP-sharded variables via a sharding constraint.
- `assert_params_sufficiently_sharded(params, mesh, tolerance)` — [`L470`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L470) — Asserts that the total size of replicated parameters is within a given tolerance.
- `build_zero1_input_state_mesh_shardings(config, state_mesh_shardings, params_shardings)` — [`L704`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L704) — Build the train-step input shardings under shard_optimizer_over_data (Zero-1).
- `clear_input_shardings_dump()` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L37) — Clear the input shardings dump
- `create_sharding(mesh, logical_names, rules=None)` — [`L291`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L291) — Create NamedSharding with given logical names.
- `find_adam_mu(obj)` — [`L645`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L645)
- `get_formatted_sharding_annotations(params, mesh=None)` — [`L758`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L758) — Generates a readable string report of sharding annotations for all parameters.
- `get_input_data_sharding(config, mesh)` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L43) — Get the input data sharding for the model
- `get_mesh_axes_used_by_tensor_spec(tensor_sharding_spec)` — [`L296`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L296) — Extracts the set of mesh axis names that a tensor's PartitionSpec uses.
- `get_nnx_var_named_sharding_with_scan_axis(v: nnx.Variable, mesh)` — [`L179`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L179) — Compute NamedSharding for an NNX variable, correctly handling the scan axis.
- `get_physical_spec_no_fsdp(full_logical, mesh, logical_axis_rules)` — [`L878`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L878) — Generates a physical sharding spec for fully replicated weights.
- `logical_axis_rules_pp_act_as_dp(logical_rules)` — [`L740`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L740) — Add stage as a physical axes before data for each rule, so stage acts just like data instead of PP.
- `logical_to_mesh(tree, mesh, rules=None)` — [`L269`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L269) — Remove size one mesh axes given logical pspec pytree.
- `logical_to_mesh_axes(logical_names, mesh, rules=None)` — [`L263`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L263) — Remove size one mesh axes given logical names.
- `logical_to_mesh_sharding(tree, mesh, rules=None)` — [`L280`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L280) — Return sharding pytree given logical specs pytree
- `maybe_shard_with_logical(inputs, logical_axes, mesh, shard_mode, rules=None, debug_sharding=False, extra_stack_level=0, sharding_desc="")` — [`L131`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L131) — A wrapper of maybe_shard_with_name when logical axes are inputs
- `maybe_shard_with_name(inputs, named_sharding, shard_mode, debug_sharding=False, extra_stack_level=0, sharding_desc="", logical_axes=None)` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L73) — In auto shardmode, this function hints inputs follow given named_sharding.
- `maybe_shard_with_pspec(inputs, pspec: jax.sharding.PartitionSpec | None, mesh, shard_mode, debug_sharding=False, extra_stack_level=0)` — [`L116`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L116)
- `maybe_update_params_sharding_with_opt(config, state_mesh_shardings)` — [`L550`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L550) — Updates parameter sharding configuration when optimizer state sharding is enabled.
- `maybe_update_params_sharding_with_opt_nnx(config: pyconfig.HyperParameters, state_mesh_shardings: nnx.State)` — [`L588`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L588) — NNX version of parameter sharding update. Updates parameter sharding configuration
- `nnx_construct_named_sharding(abs_var_state: nnx.State, mesh)` — [`L230`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L230) — Compute NamedSharding for each NNX variable, correctly handling the scan (stacked layers) axis.
- `remove_expert_from_partition_spec(pspec, dims_to_peel)` — [`L852`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L852) — Return `pspec` with the 'expert' mesh axis removed from the given dim indices.
- `remove_fsdp_sharding(sharding_tree)` — [`L822`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L822) — Recursively traverses the sharding tree to remove fsdp axes.
- `remove_size_one_mesh_axis(spec, mesh)` — [`L158`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L158) — Removes mesh axes from a PartitionSpec (P) where the axis size is 1.

## Module values
- `_ACTIVATION_SHARDINGS_DUMP` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L34)
- `_LOGGED_ACTIVATION_SHARDINGS` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/utils/sharding.py#L33)

