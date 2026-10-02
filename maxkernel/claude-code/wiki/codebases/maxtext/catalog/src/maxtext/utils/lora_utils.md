---
title: 'Module: src/maxtext/utils/lora_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/lora_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.lora_utils`/
symbols:
  get_lora_abstract_state_nnx.add_lora: get_lora_abstract_state_nnx().add_lora().
  get_lora_abstract_state.add_lora_params: get_lora_abstract_state().add_lora_params().
  setup_initial_lora_state: setup_initial_lora_state().
  apply_lora_to_model: apply_lora_to_model().
  restore_lora_from_path: restore_lora_from_path().
  apply_lora_on_base_params.apply_lora_recursively: apply_lora_on_base_params().apply_lora_recursively().
  unapply_lora_from_base_params.unapply_lora_recursively: unapply_lora_from_base_params().unapply_lora_recursively().
  get_lora_abstract_state: get_lora_abstract_state().
  apply_lora_on_base_params_nnx.recurse: apply_lora_on_base_params_nnx().recurse().
  unapply_lora_from_base_params_nnx.recurse: unapply_lora_from_base_params_nnx().recurse().
  _verify_lora_parameters: _verify_lora_parameters().
  load_adapter: load_adapter().
  get_lora_abstract_state_nnx: get_lora_abstract_state_nnx().
  _get_lora_module_path: _get_lora_module_path().
  _build_lora_provider: _build_lora_provider().
  _is_nnx_branch: _is_nnx_branch().
  apply_lora_on_base_params: apply_lora_on_base_params().
  unapply_lora_from_base_params: unapply_lora_from_base_params().
  apply_lora_on_base_params_nnx: apply_lora_on_base_params_nnx().
  unapply_lora_from_base_params_nnx: unapply_lora_from_base_params_nnx().
  is_lora_enabled: is_lora_enabled().
  apply_lora_on_base_params.lora_update_or_base: apply_lora_on_base_params().lora_update_or_base().
  unapply_lora_from_base_params.lora_update_or_base: unapply_lora_from_base_params().lora_update_or_base().
  setup_initial_lora_state.create_train_state_fn: setup_initial_lora_state().create_train_state_fn().
  get_lora_abstract_state.get_lora_param_shape: get_lora_abstract_state().get_lora_param_shape().
  get_lora_abstract_state.get_lora_param_sharding: get_lora_abstract_state().get_lora_param_sharding().
  get_lora_abstract_state.module_is_target_module: get_lora_abstract_state().module_is_target_module().
  get_lora_abstract_state.get_lora_annotations: get_lora_abstract_state().get_lora_annotations().
  _prepare_dummy_inputs: _prepare_dummy_inputs().
  sync_lora_metadata: sync_lora_metadata().
  apply_lora_to_model._safe_reshard: apply_lora_to_model()._safe_reshard().
  restore_lora_from_path._map_to_state: restore_lora_from_path()._map_to_state().
  _nnx_param_subtree: _nnx_param_subtree().
  get_lora_abstract_state_nnx.get_lora_param_shape: get_lora_abstract_state_nnx().get_lora_param_shape().
  get_lora_abstract_state_nnx.get_lora_param_sharding: get_lora_abstract_state_nnx().get_lora_param_sharding().
  get_lora_abstract_state_nnx.module_is_target: get_lora_abstract_state_nnx().module_is_target().
---
# Module: [`src/maxtext/utils/lora_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py)

## Functions
- `_build_lora_provider(mt_config: pyconfig.HyperParameters)` — [`L445`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L445) — Builds a Qwix LoRA provider from MaxText LoRA settings.
- `_get_lora_module_path(mt_config: pyconfig.HyperParameters)` — [`L417`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L417) — Gets the regex for modules to apply LoRA on from config, architecture map, or fallback.
- `_is_nnx_branch(x)` — [`L746`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L746) — Return True if `x` should be recursed into as a sub-tree.
- `_map_to_state(path, variable)` — [`L703`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L703)
- `_nnx_param_subtree(unboxed_abstract_state)` — [`L751`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L751) — Return the `model` substate, peeling off the outer `TrainStateNNX` wrapping.
- `_prepare_dummy_inputs(dummy_bs: int = 1)` — [`L468`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L468) — Builds dummy decoder inputs used to materialize LoRA parameters.
- `_safe_reshard(var, sharding_spec)` — [`L626`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L626)
- `_verify_lora_parameters(lora_model: nnx.Module, mt_config: pyconfig.HyperParameters)` — [`L485`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L485) — Validates that LoRA is active or that target modules were matched.
- `add_lora(out_node, base_node, path)` — [`L885`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L885)
- `add_lora_params(lora_params, module_name, base_params, lora_rank, lora_target_modules)` — [`L364`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L364)
- `apply_lora_on_base_params(base_params, lora_params, lora_scale_factor=1)` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L44) — Apply the LoRA weights on the base weights of the model using formula:
- `apply_lora_on_base_params_nnx(base_params, lora_params, lora_scale_factor=1)` — [`L756`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L756) — Apply LoRA deltas to `base_params` on the NNX path.
- `apply_lora_recursively(base_params, lora_params, module_name)` — [`L64`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L64)
- `apply_lora_to_model(model: nnx.Module, mesh: Optional[jax.sharding.Mesh], mt_config: pyconfig.HyperParameters)` — [`L576`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L576) — Optionally applies LoRA/QLoRA to a MaxText model using Qwix.
- `create_train_state_fn()` — [`L209`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L209)
- `get_lora_abstract_state(base_abstract_params, lora_config)` — [`L254`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L254) — Generates an abstract state representing only the LoRA parameters,
- `get_lora_abstract_state_nnx(base_abstract_params, lora_config)` — [`L812`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L812) — Build an abstract LoRA state from an NNX-shaped base abstract state.
- `get_lora_annotations(lora_abstract_params)` — [`L398`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L398)
- `get_lora_param_shape(base_array_shape, lora_rank, lora_module)` — [`L284`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L284)
- `get_lora_param_shape(base_array_shape, lora_module)` — [`L839`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L839)
- `get_lora_param_sharding(base_param_sharding, lora_module)` — [`L310`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L310)
- `get_lora_param_sharding(base_param_sharding, lora_module)` — [`L855`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L855)
- `is_lora_enabled(model: nnx.Module)` — [`L477`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L477) — Checks if the model has LoRA parameters.
- `load_adapter(config, base_abstract_state_params, adapter_config_path, adapter_weights_path)` — [`L122`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L122) — Load a LoRA adapter from disk and return its parameters.
- `lora_update_or_base(base_weight, lora_a, lora_b)` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L58)
- `lora_update_or_base(base_weight, lora_a, lora_b)` — [`L97`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L97)
- `module_is_target(module_path)` — [`L879`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L879)
- `module_is_target_module(module, target_modules)` — [`L349`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L349) — Checks if any of the target_modules is part of the current module which represents an array.
- `recurse(base_node, lora_node, path)` — [`L770`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L770)
- `recurse(base_node, lora_node, path)` — [`L795`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L795)
- `restore_lora_from_path(model: nnx.Module, mt_config: pyconfig.HyperParameters)` — [`L647`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L647) — Restores LoRA parameter weights from an external Orbax checkpoint.
- `setup_initial_lora_state(model, data_iterator, tx, config, rng, mesh, checkpoint_manager, lora_adapter_path)` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L172) — Initialize the LoRA train state and optionally restore it from a checkpoint.
- `sync_lora_metadata(config: pyconfig.HyperParameters)` — [`L537`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L537) — Syncs LoRA parameters (rank, alpha) from the checkpoint sidecar metadata if present.
- `unapply_lora_from_base_params(base_params, lora_params, lora_scale_factor=1)` — [`L83`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L83) — Unapply the LoRA weights from the base weights of the model using formula:
- `unapply_lora_from_base_params_nnx(base_params, lora_params, lora_scale_factor=1)` — [`L788`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L788) — Unapply LoRA deltas from `base_params` on the NNX path.
- `unapply_lora_recursively(base_params, lora_params, module_name)` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/utils/lora_utils.py#L103)

