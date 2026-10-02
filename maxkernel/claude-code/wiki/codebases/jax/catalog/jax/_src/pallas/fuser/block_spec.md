---
title: 'Module: jax/_src/pallas/fuser/block_spec.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/fuser/block_spec.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.fuser.block_spec`/
symbols:
  _pull_block_transform: _pull_block_transform().
  make_kernel_function.kernel_fn: make_kernel_function().kernel_fn().
  _push_block_spec_jaxpr: _push_block_spec_jaxpr().
  BlockIndexTransform: BlockIndexTransform#
  Usage: Usage#
  _get_pull_rule: _get_pull_rule().
  compute_usage: compute_usage().
  _get_eval_rule: _get_eval_rule().
  _reshape_pull_rule: _reshape_pull_rule().
  register_binop_push_rule: register_binop_push_rule.
  BlockIndexTransform.block_shape: BlockIndexTransform#block_shape.
  _swap_eval_rule: _swap_eval_rule().
  _concatenate_eval_rule: _concatenate_eval_rule().
  _custom_jvp_call_eval_rule: _custom_jvp_call_eval_rule().
  _custom_vjp_call_eval_rule: _custom_vjp_call_eval_rule().
  register_eval_rule: register_eval_rule().
  register_pull_block_spec_rule: register_pull_block_spec_rule().
  PullRuleContext: PullRuleContext#
  _reshape_push_rule: _reshape_push_rule().
  _apply_block_transform: _apply_block_transform().
  register_binop_rule: register_binop_rule().
  register_eltwise_rule: register_eltwise_rule().
  _get_fusion_values: _get_fusion_values().
  _concatenate_rule: _concatenate_rule().
  _jit_eval_rule: _jit_eval_rule().
  _init_block_transforms: _init_block_transforms().
  _tile_pull_rule: _tile_pull_rule().
  _offset_indexer: _offset_indexer().
  KernelEvalContext: KernelEvalContext#
  _slice_rule: _slice_rule().
  _reduce_sum_pull_rule: _reduce_sum_pull_rule().
  _reduce_sum_push_rule: _reduce_sum_push_rule().
  _broadcast_in_dim_push_rule: _broadcast_in_dim_push_rule().
  _unstack_push_rule: _unstack_push_rule().
  PullRuleContext.avals_in: PullRuleContext#avals_in.
  _block_size: _block_size().
  register_push_block_spec_rule: register_push_block_spec_rule().
  _dot_general_pull_rule: _dot_general_pull_rule().
  _broadcast_in_dim_pull_rule: _broadcast_in_dim_pull_rule().
  _transpose_pull_rule: _transpose_pull_rule().
  _reduce_sum_eval_rule: _reduce_sum_eval_rule().
  _concatenate_push_rule: _concatenate_push_rule().
  _pull_block_spec: _pull_block_spec().
  no_block_index_transform: no_block_index_transform.
  _squeeze_block_spec: _squeeze_block_spec().
  _stack_pull_rule: _stack_pull_rule().
  _unstack_pull_rule: _unstack_pull_rule().
  make_kernel_function: make_kernel_function().
  KernelEvalContext.get_out_block_indices: KernelEvalContext#get_out_block_indices().
  pull_block_spec.wrapped: pull_block_spec().wrapped().
  BlockIndexTransform.block_index_transform: BlockIndexTransform#block_index_transform.
  _select_n_pull_block_spec_rule: _select_n_pull_block_spec_rule().
  _dynamic_slice_rule: _dynamic_slice_rule().
  _broadcast_in_dim_eval_rule: _broadcast_in_dim_eval_rule().
  _random_fold_in_pull_rule: _random_fold_in_pull_rule().
  _iota_eval_rule: _iota_eval_rule().
  _jit_pull_block_spec_rule: _jit_pull_block_spec_rule().
  _custom_jvp_call_pull_block_spec_rule: _custom_jvp_call_pull_block_spec_rule().
  _custom_vjp_call_pull_block_spec_rule: _custom_vjp_call_pull_block_spec_rule().
  Usage.REGULAR: Usage#REGULAR.
  _block_transforms_equal: _block_transforms_equal().
  _binop_push_rule: _binop_push_rule().
  _transpose_eval_rule: _transpose_eval_rule().
  _tile_eval_rule: _tile_eval_rule().
  _random_bits_pull_rule: _random_bits_pull_rule().
  _iota_pull_rule: _iota_pull_rule().
  _jit_usage_rule: _jit_usage_rule().
  _custom_jvp_call_usage_rule: _custom_jvp_call_usage_rule().
  _custom_vjp_call_usage_rule: _custom_vjp_call_usage_rule().
  _stack_push_rule: _stack_push_rule().
  _pull_bcast_block_spec: _pull_bcast_block_spec().
  PushRuleContext: PushRuleContext#
  KernelEvalContext.out_block_specs: KernelEvalContext#out_block_specs.
  register_usage_rule: register_usage_rule().
  BlockIndexTransform.replace: BlockIndexTransform#replace().
  _unwrap_block_spec_scalar_prefetch: _unwrap_block_spec_scalar_prefetch().
  _binop_pull_rule: _binop_pull_rule().
  KernelEvalContext.get_in_block_indices: KernelEvalContext#get_in_block_indices().
  _convert_element_type_pull_rule: _convert_element_type_pull_rule().
  _bitcast_convert_type_pull_rule: _bitcast_convert_type_pull_rule().
  _random_bits_eval_rule: _random_bits_eval_rule().
  _reshape_eval_rule: _reshape_eval_rule().
  _transpose_push_rule: _transpose_push_rule().
  _convert_element_type_push_rule: _convert_element_type_push_rule().
  _custom_jvp_call_push_rule: _custom_jvp_call_push_rule().
  _pjit_push_rule: _pjit_push_rule().
  KernelEvalContext.avals_in: KernelEvalContext#avals_in.
  _dot_general_pull_rule.make_transform: _dot_general_pull_rule().make_transform().
  _wrap_block_spec_scalar_prefetch: _wrap_block_spec_scalar_prefetch().
  _pull_block_transform._get_in_block_transforms: _pull_block_transform()._get_in_block_transforms().
  _get_fusion_values.new_kernel_fn: _get_fusion_values().new_kernel_fn().
  Usage.SCALAR_PREFETCH: Usage#SCALAR_PREFETCH.
  _push_bcast_block_spec: _push_bcast_block_spec().
  _dynamic_slice_usage_rule: _dynamic_slice_usage_rule().
  _dot_general_usage_rule: _dot_general_usage_rule().
  _broadcast_in_dim_usage_rule: _broadcast_in_dim_usage_rule().
  _select_n_push_rule: _select_n_push_rule().
  _dot_general_push_rule: _dot_general_push_rule().
  _custom_vjp_call_push_rule: _custom_vjp_call_push_rule().
  _push_block_spec_jaxpr._read_block_spec: _push_block_spec_jaxpr()._read_block_spec().
  _dynamic_slice_rule.new_block_index_transform: _dynamic_slice_rule().new_block_index_transform().
  push_block_spec.wrapper: push_block_spec().wrapper().
  PushRuleContext.avals_in: PushRuleContext#avals_in.
  _select_n_eval_rule: _select_n_eval_rule().
  _squeeze_eval_rule: _squeeze_eval_rule().
  _dot_general_eval_rule: _dot_general_eval_rule().
  _swap_pull_rule: _swap_pull_rule().
  _stack_eval_rule: _stack_eval_rule().
  _unstack_eval_rule: _unstack_eval_rule().
  _convert_element_type_eval_rule: _convert_element_type_eval_rule().
  _bitcast_convert_type_eval_rule: _bitcast_convert_type_eval_rule().
  _random_wrap_pull_rule: _random_wrap_pull_rule().
  compute_usage.read_usage_env: compute_usage().read_usage_env().
  KernelEvalContext.in_block_specs: KernelEvalContext#in_block_specs.
  pull_block_spec: pull_block_spec().
  make_kernel_function._read_block_spec: make_kernel_function()._read_block_spec().
  get_fusion_values: get_fusion_values().
  _pull_bcast_block_spec.new_block_index_transform: _pull_bcast_block_spec().new_block_index_transform().
  register_default_eval_rule: register_default_eval_rule().
  _jit_eval_rule.read_usage_env: _jit_eval_rule().read_usage_env().
  _custom_jvp_call_eval_rule.read_usage_env: _custom_jvp_call_eval_rule().read_usage_env().
  _custom_vjp_call_eval_rule.read_usage_env: _custom_vjp_call_eval_rule().read_usage_env().
  _push_block_spec_jaxpr._write_block_spec: _push_block_spec_jaxpr()._write_block_spec().
  _sp_context: _sp_context().
  _wrap_block_spec_scalar_prefetch.new_index_map: _wrap_block_spec_scalar_prefetch().new_index_map().
  _block_dim_equal: _block_dim_equal().
  _pull_block_transform._read_block_spec: _pull_block_transform()._read_block_spec().
  make_kernel_function._get_block_aval: make_kernel_function()._get_block_aval().
  _eltwise_pull_rule: _eltwise_pull_rule().
  _eltwise_usage_rule: _eltwise_usage_rule().
  _binop_usage_rule: _binop_usage_rule().
  _squeeze_block_spec.new_block_index_transform: _squeeze_block_spec().new_block_index_transform().
  _maybe_static_check: _maybe_static_check().
  _slice_rule.new_block_index_transform: _slice_rule().new_block_index_transform().
  _get_eval_rule._slice: _get_eval_rule()._slice().
  _concatenate_rule.make_block_transform.new_block_index_transform: _concatenate_rule().make_block_transform().new_block_index_transform().
  _jit_pull_block_spec_rule.read_usage_env: _jit_pull_block_spec_rule().read_usage_env().
  _custom_jvp_call_pull_block_spec_rule.read_usage_env: _custom_jvp_call_pull_block_spec_rule().read_usage_env().
  _custom_vjp_call_pull_block_spec_rule.read_usage_env: _custom_vjp_call_pull_block_spec_rule().read_usage_env().
  _eltwise_push_rule: _eltwise_push_rule().
  _reshape_pull_rule.new_block_index_transform: _reshape_pull_rule().new_block_index_transform().
  KernelEvalContext.program_ids: KernelEvalContext#program_ids.
  _slice_eval_rule: _slice_eval_rule().
  _random_wrap_eval_rule: _random_wrap_eval_rule().
  _custom_call_hi_primitive_pull_block_spec_rule: _custom_call_hi_primitive_pull_block_spec_rule().
  _custom_call_hi_primitive_eval_rule: _custom_call_hi_primitive_eval_rule().
  _custom_call_hi_primitive_push_block_spec_rule: _custom_call_hi_primitive_push_block_spec_rule().
  NoBlockIndexTransform: NoBlockIndexTransform#
  compute_usage.write_usage_env: compute_usage().write_usage_env().
  KernelEvalContext.grid_len: KernelEvalContext#grid_len.
  KernelEvalContext.scalar_prefetch_handler: KernelEvalContext#scalar_prefetch_handler.
  _get_scalar_prefetch: _get_scalar_prefetch().
  _block_shapes_equal: _block_shapes_equal().
  make_kernel_function.kernel_fn.read_env: make_kernel_function().kernel_fn().read_env().
  _pattern_match_lanes_to_sublanes_reshape: _pattern_match_lanes_to_sublanes_reshape().
  push_block_spec: push_block_spec().
  pull_block_spec_rules.pull_block_spec_rules: pull_block_spec_rules.pull_block_spec_rules.
  usage_rules.usage_rules: usage_rules.usage_rules.
  eval_rules.eval_rules: eval_rules.eval_rules.
  push_block_spec_rules.push_block_spec_rules: push_block_spec_rules.push_block_spec_rules.
  PullRuleContext.avals_out: PullRuleContext#avals_out.
  _sp_env: _sp_env.
  _init_block_transforms.compare: _init_block_transforms().compare().
  _apply_block_transform.make_new_idx_map.new_idx_map: _apply_block_transform().make_new_idx_map().new_idx_map().
  KernelEvalContext.scalar_prefetch: KernelEvalContext#scalar_prefetch.
  _unwrap_block_spec_scalar_prefetch.new_index_map: _unwrap_block_spec_scalar_prefetch().new_index_map().
  _compare_index_transforms: _compare_index_transforms().
  _pull_block_transform._write_block_spec: _pull_block_transform()._write_block_spec().
  _pull_block_transform._scalar_prefetch_fn: _pull_block_transform()._scalar_prefetch_fn().
  make_kernel_function._remove_nones: make_kernel_function()._remove_nones().
  get_stateful_input_fusion_values: get_stateful_input_fusion_values().
  get_stateful_output_fusion_values: get_stateful_output_fusion_values().
  register_usage_rule.wrapper: register_usage_rule().wrapper().
  register_eval_rule.wrapper: register_eval_rule().wrapper().
  register_pull_block_spec_rule.wrapper: register_pull_block_spec_rule().wrapper().
  register_default_eval_rule.default_rule: register_default_eval_rule().default_rule().
  _get_pull_rule.new_block_index_transform: _get_pull_rule().new_block_index_transform().
  _concatenate_rule.make_block_transform: _concatenate_rule().make_block_transform().
  _stack_pull_rule.make_block_transform: _stack_pull_rule().make_block_transform().
  _tile_pull_rule.new_block_index_transform: _tile_pull_rule().new_block_index_transform().
  _reduce_sum_pull_rule.new_block_index_transform: _reduce_sum_pull_rule().new_block_index_transform().
  register_push_block_spec_rule.wrapper: register_push_block_spec_rule().wrapper().
  BlockIndexTransform.pipeline_mode: BlockIndexTransform#pipeline_mode.
  KernelEvalContext.get_program_ids: KernelEvalContext#get_program_ids().
  _unwrap_cache._unwrap_cache: _unwrap_cache._unwrap_cache.
  UsageRuleContext: UsageRuleContext#
  UsageRuleFn.__call__: UsageRuleFn#__call__().
  PullBlockSpecRuleFn.__call__: PullBlockSpecRuleFn#__call__().
  _dynamic_slice_eval_rule: _dynamic_slice_eval_rule().
  _random_fold_in_eval_rule: _random_fold_in_eval_rule().
  PushBlockSpecRuleFn.__call__: PushBlockSpecRuleFn#__call__().
  make_scalar_prefetch_handler: make_scalar_prefetch_handler().
  PullRuleContext.scalar_prefetch_handler: PullRuleContext#scalar_prefetch_handler.
  PullRuleContext.grid_len: PullRuleContext#grid_len.
  PullRuleContext.strict_mode: PullRuleContext#strict_mode.
  _SpEnv.scalar_prefetch: _SpEnv#scalar_prefetch.
  UsageRuleFn: UsageRuleFn#
  EvalRuleFn: EvalRuleFn#
  PullBlockSpecRuleFn: PullBlockSpecRuleFn#
  PushBlockSpecRuleFn: PushBlockSpecRuleFn#
  _select_block_indices: _select_block_indices().
  _apply_block_transform.make_new_idx_map: _apply_block_transform().make_new_idx_map().
  PullRuleContext.out_usages: PullRuleContext#out_usages.
  PushRuleContext.avals_out: PushRuleContext#avals_out.
  UsageRuleContext.avals_in: UsageRuleContext#avals_in.
  UsageRuleContext.avals_out: UsageRuleContext#avals_out.
  KernelEvalContext.avals_out: KernelEvalContext#avals_out.
  KernelEvalContext.out_usages: KernelEvalContext#out_usages.
  _swap_eval_rule._slice: _swap_eval_rule()._slice().
  _stack_pull_rule.make_block_transform.new_block_index_transform: _stack_pull_rule().make_block_transform().new_block_index_transform().
  _unstack_pull_rule.new_block_index_transform: _unstack_pull_rule().new_block_index_transform().
  _broadcast_in_dim_pull_rule.new_block_index_transform: _broadcast_in_dim_pull_rule().new_block_index_transform().
  _transpose_pull_rule.new_block_index_transform: _transpose_pull_rule().new_block_index_transform().
  _transpose_push_rule.new_index_map: _transpose_push_rule().new_index_map().
  _reshape_push_rule.new_index_map: _reshape_push_rule().new_index_map().
  _reduce_sum_push_rule.new_index_map: _reduce_sum_push_rule().new_index_map().
  _broadcast_in_dim_push_rule.new_index_map: _broadcast_in_dim_push_rule().new_index_map().
  _concatenate_push_rule._new_index_map: _concatenate_push_rule()._new_index_map().
  _stack_push_rule._new_index_map: _stack_push_rule()._new_index_map().
  _unstack_push_rule._new_index_map: _unstack_push_rule()._new_index_map().
  BlockIndexTransform.memory_space: BlockIndexTransform#memory_space.
  _illegal: _illegal.
  EvalRuleFn.__call__: EvalRuleFn#__call__().
  PullRuleContext.scalar_prefetch_fn: PullRuleContext#scalar_prefetch_fn.
  _null_block_index_trafo: _null_block_index_trafo().
  _select_block_indices.block_transform: _select_block_indices().block_transform().
  make_scalar_prefetch_handler.scalar_prefetch_getter: make_scalar_prefetch_handler().scalar_prefetch_getter().
  _SpEnv: _SpEnv#
  make_kernel_function.kernel_fn.write_env: make_kernel_function().kernel_fn().write_env().
  _eltwise_eval_rule: _eltwise_eval_rule().
  _binop_eval_rule: _binop_eval_rule().
  _dot_general_pull_rule.make_transform.transform: _dot_general_pull_rule().make_transform().transform().
  NoBlockIndexTransform.__repr__: NoBlockIndexTransform#__repr__().
  _SpEnv.__init__: _SpEnv#__init__().
---
# Module: [`jax/_src/pallas/fuser/block_spec.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py)

## Classes
### `BlockIndexTransform`
- def: [`jax/_src/pallas/fuser/block_spec.py:67`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L67) — documented in [jax-_src-pallas-fuser-block_spec](../../../../../concepts/jax-_src-pallas-fuser-block_spec.md)
- signature: `class BlockIndexTransform:`
- members:
  - `replace(self, **kwargs)` — [`L77`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L77)
  - `block_index_transform` — [`L69`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L69)
  - `block_shape` — [`L68`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L68)
  - `memory_space` — [`L72`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L72)
  - `pipeline_mode` — [`L74`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L74)
- uses (calls/refs, reference-scoped): [`BlockDim`](../core.md#BlockDim.BlockDim), [`Buffered`](../core.md#Buffered), [`_null_block_index_trafo`](block_spec.md#_null_block_index_trafo)
- used by: [`_pull_block_transform`](block_spec.md#_pull_block_transform), [`_get_pull_rule`](block_spec.md#_get_pull_rule), [`_reshape_pull_rule`](block_spec.md#_reshape_pull_rule), [`_apply_block_transform`](block_spec.md#_apply_block_transform), [`_concatenate_rule`](block_spec.md#_concatenate_rule), [`_init_block_transforms`](block_spec.md#_init_block_transforms), [`_tile_pull_rule`](block_spec.md#_tile_pull_rule), [`_reduce_sum_pull_rule`](block_spec.md#_reduce_sum_pull_rule), [`_slice_rule`](block_spec.md#_slice_rule), [`_broadcast_in_dim_pull_rule`](block_spec.md#_broadcast_in_dim_pull_rule), [`_dot_general_pull_rule`](block_spec.md#_dot_general_pull_rule), [`_transpose_pull_rule`](block_spec.md#_transpose_pull_rule), [`_squeeze_block_spec`](block_spec.md#_squeeze_block_spec), [`_stack_pull_rule`](block_spec.md#_stack_pull_rule), [`_unstack_pull_rule`](block_spec.md#_unstack_pull_rule), [`_dynamic_slice_rule`](block_spec.md#_dynamic_slice_rule), [`_random_fold_in_pull_rule`](block_spec.md#_random_fold_in_pull_rule), [`_select_n_pull_block_spec_rule`](block_spec.md#_select_n_pull_block_spec_rule), [`_block_transforms_equal`](block_spec.md#_block_transforms_equal), [`_iota_pull_rule`](block_spec.md#_iota_pull_rule), [`_random_bits_pull_rule`](block_spec.md#_random_bits_pull_rule), [`_pull_bcast_block_spec`](block_spec.md#_pull_bcast_block_spec), [`_bitcast_convert_type_pull_rule`](block_spec.md#_bitcast_convert_type_pull_rule), [`_convert_element_type_pull_rule`](block_spec.md#_convert_element_type_pull_rule), [`make_transform`](block_spec.md#_dot_general_pull_rule.make_transform), [`_get_in_block_transforms`](block_spec.md#_pull_block_transform._get_in_block_transforms), [`_custom_fusion_pull_block_spec_rule`](custom_fusion_lib.md#_custom_fusion_pull_block_spec_rule), [`new_block_index_transform`](block_spec.md#_dynamic_slice_rule.new_block_index_transform), [`_random_wrap_pull_rule`](block_spec.md#_random_wrap_pull_rule), [`_swap_pull_rule`](block_spec.md#_swap_pull_rule), [`new_block_index_transform`](block_spec.md#_pull_bcast_block_spec.new_block_index_transform), [`_eltwise_pull_rule`](block_spec.md#_eltwise_pull_rule), [`_read_block_spec`](block_spec.md#_pull_block_transform._read_block_spec), [`new_block_index_transform`](block_spec.md#_concatenate_rule.make_block_transform.new_block_index_transform), [`new_block_index_transform`](block_spec.md#_slice_rule.new_block_index_transform), [`new_block_index_transform`](block_spec.md#_squeeze_block_spec.new_block_index_transform), [`new_block_index_transform`](block_spec.md#_reshape_pull_rule.new_block_index_transform), [`_write_block_spec`](block_spec.md#_pull_block_transform._write_block_spec), [`make_block_transform`](block_spec.md#_concatenate_rule.make_block_transform), [`make_block_transform`](block_spec.md#_stack_pull_rule.make_block_transform)  (+9 more)

### `EvalRuleFn`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/fuser/block_spec.py:885`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L885)
- signature: `class EvalRuleFn(Protocol):`
- protocol/private: `__call__`[`L887`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L887)
- uses (calls/refs, reference-scoped): [`KernelEvalContext`](block_spec.md#KernelEvalContext)
- used by: [`register_eval_rule`](block_spec.md#register_eval_rule), [`eval_rules`](block_spec.md#eval_rules.eval_rules), [`wrapper`](block_spec.md#register_eval_rule.wrapper)

### `KernelEvalContext`
- def: [`jax/_src/pallas/fuser/block_spec.py:249`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L249)
- signature: `class KernelEvalContext:`
- members:
  - `get_in_block_indices(self)` — [`L265`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L265)
  - `get_out_block_indices(self)` — [`L274`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L274)
  - `get_program_ids(self)` — [`L260`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L260)
  - `avals_in` — [`L252`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L252)
  - `avals_out` — [`L253`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L253)
  - `grid_len` — [`L256`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L256)
  - `in_block_specs` — [`L254`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L254)
  - `out_block_specs` — [`L255`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L255)
  - `out_usages` — [`L258`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L258)
  - `program_ids` — [`L251`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L251)
  - `scalar_prefetch` — [`L250`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L250)
  - `scalar_prefetch_handler` — [`L257`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L257)
- uses (calls/refs, reference-scoped): [`Array`](../../basearray.md#Array), [`AbstractValue`](../../core.md#AbstractValue), [`BlockSpec`](../core.md#BlockSpec), [`Usage`](block_spec.md#Usage), [`no_block_spec`](../core.md#no_block_spec), [`index_map`](../core.md#BlockSpec.index_map), [`_sp_context`](block_spec.md#_sp_context)
- used by: [`kernel_fn`](block_spec.md#make_kernel_function.kernel_fn), [`_get_eval_rule`](block_spec.md#_get_eval_rule), [`_concatenate_eval_rule`](block_spec.md#_concatenate_eval_rule), [`_custom_jvp_call_eval_rule`](block_spec.md#_custom_jvp_call_eval_rule), [`_custom_vjp_call_eval_rule`](block_spec.md#_custom_vjp_call_eval_rule), [`_swap_eval_rule`](block_spec.md#_swap_eval_rule), [`_jit_eval_rule`](block_spec.md#_jit_eval_rule), [`_reduce_sum_eval_rule`](block_spec.md#_reduce_sum_eval_rule), [`_custom_fusion_eval_rule`](custom_fusion_lib.md#_custom_fusion_eval_rule), [`_broadcast_in_dim_eval_rule`](block_spec.md#_broadcast_in_dim_eval_rule), [`_iota_eval_rule`](block_spec.md#_iota_eval_rule), [`_tile_eval_rule`](block_spec.md#_tile_eval_rule), [`_transpose_eval_rule`](block_spec.md#_transpose_eval_rule), [`_unpack_dtype_eval_rule`](fusible_dtype.md#_unpack_dtype_eval_rule), [`_random_bits_eval_rule`](block_spec.md#_random_bits_eval_rule), [`_reshape_eval_rule`](block_spec.md#_reshape_eval_rule), [`_bitcast_convert_type_eval_rule`](block_spec.md#_bitcast_convert_type_eval_rule), [`_convert_element_type_eval_rule`](block_spec.md#_convert_element_type_eval_rule), [`_dot_general_eval_rule`](block_spec.md#_dot_general_eval_rule), [`_select_n_eval_rule`](block_spec.md#_select_n_eval_rule), [`_squeeze_eval_rule`](block_spec.md#_squeeze_eval_rule), [`_stack_eval_rule`](block_spec.md#_stack_eval_rule), [`_unstack_eval_rule`](block_spec.md#_unstack_eval_rule), [`_custom_call_hi_primitive_eval_rule`](block_spec.md#_custom_call_hi_primitive_eval_rule), [`_pack_dtype_eval_rule`](fusible_dtype.md#_pack_dtype_eval_rule), [`_random_wrap_eval_rule`](block_spec.md#_random_wrap_eval_rule), [`_random_fold_in_eval_rule`](block_spec.md#_random_fold_in_eval_rule), [`__call__`](block_spec.md#EvalRuleFn.__call__)

### `NoBlockIndexTransform`
- def: [`jax/_src/pallas/fuser/block_spec.py:81`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L81)
- signature: `class NoBlockIndexTransform:`
- protocol/private: `__repr__`[`L82`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L82)
- used by: [`_pull_block_transform`](block_spec.md#_pull_block_transform), [`_apply_block_transform`](block_spec.md#_apply_block_transform), [`_init_block_transforms`](block_spec.md#_init_block_transforms), [`no_block_index_transform`](block_spec.md#no_block_index_transform), [`_block_transforms_equal`](block_spec.md#_block_transforms_equal)

### `PullBlockSpecRuleFn`
- def: [`jax/_src/pallas/fuser/block_spec.py:914`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L914)
- protocol/private: `__call__`[`L916`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L916)
- uses (calls/refs, reference-scoped): [`BlockIndexTransform`](block_spec.md#BlockIndexTransform), [`PullRuleContext`](block_spec.md#PullRuleContext)
- used by: [`register_pull_block_spec_rule`](block_spec.md#register_pull_block_spec_rule), [`pull_block_spec_rules`](block_spec.md#pull_block_spec_rules.pull_block_spec_rules), [`wrapper`](block_spec.md#register_pull_block_spec_rule.wrapper)

### `PullRuleContext`
- def: [`jax/_src/pallas/fuser/block_spec.py:163`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L163)
- signature: `class PullRuleContext:`
- members:
  - `avals_in` — [`L164`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L164)
  - `avals_out` — [`L165`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L165)
  - `grid_len` — [`L169`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L169)
  - `out_usages` — [`L166`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L166)
  - `scalar_prefetch_fn` — [`L167`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L167)
  - `scalar_prefetch_handler` — [`L168`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L168)
  - `strict_mode` — [`L170`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L170)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`Usage`](block_spec.md#Usage)
- used by: [`_pull_block_transform`](block_spec.md#_pull_block_transform), [`_get_pull_rule`](block_spec.md#_get_pull_rule), [`_reshape_pull_rule`](block_spec.md#_reshape_pull_rule), [`_reshape_push_rule`](block_spec.md#_reshape_push_rule), [`_concatenate_rule`](block_spec.md#_concatenate_rule), [`_tile_pull_rule`](block_spec.md#_tile_pull_rule), [`_reduce_sum_pull_rule`](block_spec.md#_reduce_sum_pull_rule), [`_slice_rule`](block_spec.md#_slice_rule), [`_broadcast_in_dim_pull_rule`](block_spec.md#_broadcast_in_dim_pull_rule), [`_dot_general_pull_rule`](block_spec.md#_dot_general_pull_rule), [`_transpose_pull_rule`](block_spec.md#_transpose_pull_rule), [`_squeeze_block_spec`](block_spec.md#_squeeze_block_spec), [`_stack_pull_rule`](block_spec.md#_stack_pull_rule), [`_unstack_pull_rule`](block_spec.md#_unstack_pull_rule), [`_custom_jvp_call_pull_block_spec_rule`](block_spec.md#_custom_jvp_call_pull_block_spec_rule), [`_custom_vjp_call_pull_block_spec_rule`](block_spec.md#_custom_vjp_call_pull_block_spec_rule), [`_dynamic_slice_rule`](block_spec.md#_dynamic_slice_rule), [`_jit_pull_block_spec_rule`](block_spec.md#_jit_pull_block_spec_rule), [`_random_fold_in_pull_rule`](block_spec.md#_random_fold_in_pull_rule), [`_select_n_pull_block_spec_rule`](block_spec.md#_select_n_pull_block_spec_rule), [`_binop_push_rule`](block_spec.md#_binop_push_rule), [`_iota_pull_rule`](block_spec.md#_iota_pull_rule), [`_pack_dtype_pull_rule`](fusible_dtype.md#_pack_dtype_pull_rule), [`_random_bits_pull_rule`](block_spec.md#_random_bits_pull_rule), [`_binop_pull_rule`](block_spec.md#_binop_pull_rule), [`_bitcast_convert_type_pull_rule`](block_spec.md#_bitcast_convert_type_pull_rule), [`_convert_element_type_pull_rule`](block_spec.md#_convert_element_type_pull_rule), [`_custom_fusion_pull_block_spec_rule`](custom_fusion_lib.md#_custom_fusion_pull_block_spec_rule), [`new_block_index_transform`](block_spec.md#_dynamic_slice_rule.new_block_index_transform), [`_random_wrap_pull_rule`](block_spec.md#_random_wrap_pull_rule), [`_swap_pull_rule`](block_spec.md#_swap_pull_rule), [`_eltwise_pull_rule`](block_spec.md#_eltwise_pull_rule), [`_eltwise_push_rule`](block_spec.md#_eltwise_push_rule), [`_custom_call_hi_primitive_pull_block_spec_rule`](block_spec.md#_custom_call_hi_primitive_pull_block_spec_rule), [`_custom_call_hi_primitive_push_block_spec_rule`](block_spec.md#_custom_call_hi_primitive_push_block_spec_rule), [`__call__`](block_spec.md#PullBlockSpecRuleFn.__call__)

### `PushBlockSpecRuleFn`
- def: [`jax/_src/pallas/fuser/block_spec.py:2625`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2625)
- protocol/private: `__call__`[`L2627`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2627)
- uses (calls/refs, reference-scoped): [`BlockSpec`](../core.md#BlockSpec), [`PushRuleContext`](block_spec.md#PushRuleContext)
- used by: [`register_push_block_spec_rule`](block_spec.md#register_push_block_spec_rule), [`push_block_spec_rules`](block_spec.md#push_block_spec_rules.push_block_spec_rules), [`wrapper`](block_spec.md#register_push_block_spec_rule.wrapper)

### `PushRuleContext`
- def: [`jax/_src/pallas/fuser/block_spec.py:174`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L174)
- signature: `class PushRuleContext:`
- members:
  - `avals_in` — [`L175`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L175)
  - `avals_out` — [`L176`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L176)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue), [`api_boundary`](../../traceback_util.md#api_boundary)
- used by: [`_push_block_spec_jaxpr`](block_spec.md#_push_block_spec_jaxpr), [`_broadcast_in_dim_push_rule`](block_spec.md#_broadcast_in_dim_push_rule), [`_reduce_sum_push_rule`](block_spec.md#_reduce_sum_push_rule), [`_unstack_push_rule`](block_spec.md#_unstack_push_rule), [`_concatenate_push_rule`](block_spec.md#_concatenate_push_rule), [`_unpack_dtype_pull_rule`](fusible_dtype.md#_unpack_dtype_pull_rule), [`_unpack_dtype_push_rule`](fusible_dtype.md#_unpack_dtype_push_rule), [`_stack_push_rule`](block_spec.md#_stack_push_rule), [`_convert_element_type_push_rule`](block_spec.md#_convert_element_type_push_rule), [`_transpose_push_rule`](block_spec.md#_transpose_push_rule), [`_custom_fusion_push_block_spec_rule`](custom_fusion_lib.md#_custom_fusion_push_block_spec_rule), [`_dot_general_push_rule`](block_spec.md#_dot_general_push_rule), [`_select_n_push_rule`](block_spec.md#_select_n_push_rule), [`__call__`](block_spec.md#PushBlockSpecRuleFn.__call__)

### `Usage`
- def: [`jax/_src/pallas/fuser/block_spec.py:850`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L850)
- members:
  - `REGULAR` — [`L851`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L851)
  - `SCALAR_PREFETCH` — [`L852`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L852)
- used by: [`_pull_block_transform`](block_spec.md#_pull_block_transform), [`kernel_fn`](block_spec.md#make_kernel_function.kernel_fn), [`compute_usage`](block_spec.md#compute_usage), [`_get_fusion_values`](block_spec.md#_get_fusion_values), [`_pull_block_spec`](block_spec.md#_pull_block_spec), [`wrapped`](block_spec.md#pull_block_spec.wrapped), [`_custom_fusion_usage_rule`](custom_fusion_lib.md#_custom_fusion_usage_rule), [`_custom_jvp_call_usage_rule`](block_spec.md#_custom_jvp_call_usage_rule), [`_custom_vjp_call_usage_rule`](block_spec.md#_custom_vjp_call_usage_rule), [`_jit_usage_rule`](block_spec.md#_jit_usage_rule), [`_get_in_block_transforms`](block_spec.md#_pull_block_transform._get_in_block_transforms), [`_broadcast_in_dim_usage_rule`](block_spec.md#_broadcast_in_dim_usage_rule), [`_dot_general_usage_rule`](block_spec.md#_dot_general_usage_rule), [`_dynamic_slice_usage_rule`](block_spec.md#_dynamic_slice_usage_rule), [`read_usage_env`](block_spec.md#_custom_jvp_call_eval_rule.read_usage_env), [`read_usage_env`](block_spec.md#_custom_vjp_call_eval_rule.read_usage_env), [`read_usage_env`](block_spec.md#_jit_eval_rule.read_usage_env), [`read_usage_env`](block_spec.md#compute_usage.read_usage_env), [`_binop_usage_rule`](block_spec.md#_binop_usage_rule), [`_eltwise_usage_rule`](block_spec.md#_eltwise_usage_rule), [`read_usage_env`](block_spec.md#_custom_jvp_call_pull_block_spec_rule.read_usage_env), [`read_usage_env`](block_spec.md#_custom_vjp_call_pull_block_spec_rule.read_usage_env), [`read_usage_env`](block_spec.md#_jit_pull_block_spec_rule.read_usage_env), [`write_usage_env`](block_spec.md#compute_usage.write_usage_env), [`__call__`](block_spec.md#UsageRuleFn.__call__), [`out_usages`](block_spec.md#KernelEvalContext.out_usages), [`out_usages`](block_spec.md#PullRuleContext.out_usages)

### `UsageRuleContext`
- def: [`jax/_src/pallas/fuser/block_spec.py:205`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L205)
- signature: `class UsageRuleContext:`
- members:
  - `avals_in` — [`L206`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L206)
  - `avals_out` — [`L207`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L207)
- uses (calls/refs, reference-scoped): [`AbstractValue`](../../core.md#AbstractValue)
- used by: [`compute_usage`](block_spec.md#compute_usage), [`_custom_fusion_usage_rule`](custom_fusion_lib.md#_custom_fusion_usage_rule), [`_dot_general_usage_rule`](block_spec.md#_dot_general_usage_rule), [`_eltwise_usage_rule`](block_spec.md#_eltwise_usage_rule), [`__call__`](block_spec.md#UsageRuleFn.__call__)

### `UsageRuleFn`  ·  implements/extends Protocol
- def: [`jax/_src/pallas/fuser/block_spec.py:855`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L855)
- signature: `class UsageRuleFn(Protocol):`
- protocol/private: `__call__`[`L857`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L857)
- uses (calls/refs, reference-scoped): [`Usage`](block_spec.md#Usage), [`UsageRuleContext`](block_spec.md#UsageRuleContext)
- used by: [`register_usage_rule`](block_spec.md#register_usage_rule), [`usage_rules`](block_spec.md#usage_rules.usage_rules), [`wrapper`](block_spec.md#register_usage_rule.wrapper)

### `_SpEnv`  ·  implements/extends local
- def: [`jax/_src/pallas/fuser/block_spec.py:287`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L287)
- signature: `class _SpEnv(threading.local):`
- members:
  - `scalar_prefetch` — [`L290`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L290)
- protocol/private: `__init__`[`L289`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L289)
- used by: [`_sp_context`](block_spec.md#_sp_context), [`_get_scalar_prefetch`](block_spec.md#_get_scalar_prefetch), [`_sp_env`](block_spec.md#_sp_env)

## Functions
- `_apply_block_transform(block_specs: tuple[pallas_core.BlockSpec, ...], block_index_transform: BlockIndexTransform | NoBlockIndexTransform)` — [`L131`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L131)
- `_binop_eval_rule(prim, ctx, x, y, **params)` — [`L1017`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1017)
- `_binop_pull_rule(prim, ctx: PullRuleContext, block_transform, **params)` — [`L1022`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1022)
- `_binop_push_rule(prim: core.Primitive, ctx: PullRuleContext, left_block_spec: pallas_core.BlockSpec, right_block_spec: pallas_core.BlockSpec, **params: Any)` — [`L2649`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2649)
- `_binop_usage_rule(prim, ctx, used_out: set[Usage], **params)` — [`L1005`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1005)
- `_bitcast_convert_type_eval_rule(eval_ctx: KernelEvalContext, x, new_dtype)` — [`L1980`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1980)
- `_bitcast_convert_type_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, new_dtype: jnp.dtype)` — [`L1986`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1986)
- `_block_dim_equal(b1: int | pallas_core.BlockDim | None, b2: int | pallas_core.BlockDim | None)` — [`L410`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L410)
- `_block_shapes_equal(bs1: tuple[int | pallas_core.BlockDim | None] | None, bs2: tuple[int | pallas_core.BlockDim | None] | None)` — [`L429`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L429)
- `_block_size(dim: pallas_core.Element | int | None)` — [`L190`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L190)
- `_block_transforms_equal(bs1: BlockIndexTransform | NoBlockIndexTransform, bs2: BlockIndexTransform | NoBlockIndexTransform, block_idxs_avals: tuple[tuple[core.AbstractValue, ...], ...], strict_mode: bool = True)` — [`L446`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L446)
- `_broadcast_in_dim_eval_rule(eval_ctx: KernelEvalContext, x, broadcast_dimensions, shape, **params)` — [`L1784`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1784)
- `_broadcast_in_dim_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, shape: tuple[int, ...], broadcast_dimensions: tuple[int, ...], sharding: jax.sharding.Sharding)` — [`L1803`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1803)
- `_broadcast_in_dim_push_rule(ctx: PushRuleContext, block_spec: pallas_core.BlockSpec, *, shape: tuple[int, ...], broadcast_dimensions: tuple[int, ...], sharding: jax.sharding.Sharding)` — [`L2913`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2913)
- `_broadcast_in_dim_usage_rule(ctx, used_out: set[Usage], **params)` — [`L1769`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1769)
- `_compare_index_transforms(idx_map1, idx_map2, block_idxs_avals)` — [`L438`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L438)
- `_concatenate_eval_rule(ctx: KernelEvalContext, *args, dimension)` — [`L1575`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1575)
- `_concatenate_push_rule(ctx: PushRuleContext, *block_specs: pallas_core.BlockSpec, dimension: int)` — [`L2953`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2953)
- `_concatenate_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, dimension: int)` — [`L1631`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1631)
- `_convert_element_type_eval_rule(eval_ctx: KernelEvalContext, x, new_dtype, **params)` — [`L1959`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1959)
- `_convert_element_type_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, new_dtype: jnp.dtype, weak_type: bool, sharding: jax.sharding.Sharding)` — [`L1967`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1967)
- `_convert_element_type_push_rule(ctx: PushRuleContext, block_spec: pallas_core.BlockSpec, *, new_dtype: jnp.dtype, weak_type: bool, sharding: jax.sharding.Sharding)` — [`L2736`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2736)
- `_custom_call_hi_primitive_eval_rule(ctx: KernelEvalContext, *args, _prim)` — [`L2533`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2533)
- `_custom_call_hi_primitive_pull_block_spec_rule(ctx: PullRuleContext, out_block_specs, *, _prim)` — [`L2527`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2527)
- `_custom_call_hi_primitive_push_block_spec_rule(ctx: PullRuleContext, *block_specs, _prim)` — [`L2800`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2800)
- `_custom_jvp_call_eval_rule(ctx: KernelEvalContext, *args, call_jaxpr: core.ClosedJaxpr, **kwargs)` — [`L2403`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2403)
- `_custom_jvp_call_pull_block_spec_rule(ctx: PullRuleContext, out_block_specs, *, call_jaxpr, **kwargs)` — [`L2437`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2437)
- `_custom_jvp_call_push_rule(ctx, *block_specs, call_jaxpr: core.ClosedJaxpr, **_)` — [`L2778`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2778)
- `_custom_jvp_call_usage_rule(ctx, used_out: list[set[Usage]], *, call_jaxpr: core.ClosedJaxpr, **_)` — [`L2393`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2393)
- `_custom_vjp_call_eval_rule(ctx: KernelEvalContext, *args, call_jaxpr: core.ClosedJaxpr, **kwargs)` — [`L2470`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2470)
- `_custom_vjp_call_pull_block_spec_rule(ctx: PullRuleContext, out_block_specs, *, call_jaxpr, **kwargs)` — [`L2504`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2504)
- `_custom_vjp_call_push_rule(ctx, *block_specs, call_jaxpr: core.ClosedJaxpr, num_consts, fwd_jaxpr_thunk, bwd, out_trees, symbolic_zeros)` — [`L2786`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2786)
- `_custom_vjp_call_usage_rule(ctx, used_out: list[set[Usage]], *, call_jaxpr: core.ClosedJaxpr, **_)` — [`L2460`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2460)
- `_dot_general_eval_rule(ctx: KernelEvalContext, x, y, **params)` — [`L1389`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1389)
- `_dot_general_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, dimension_numbers, **_)` — [`L1324`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1324)
- `_dot_general_push_rule(ctx: PushRuleContext, lhs_block_spec: pallas_core.BlockSpec | pallas_core.NoBlockSpec, rhs_block_spec: pallas_core.BlockSpec | pallas_core.NoBlockSpec, *, dimension_numbers, **_)` — [`L2766`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2766)
- `_dot_general_usage_rule(ctx: UsageRuleContext, used_out: set[Usage], **params)` — [`L1379`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1379)
- `_dynamic_slice_eval_rule(ctx, x, *args, **params)` — [`L1275`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1275)
- `_dynamic_slice_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, slice_sizes: tuple[int, ...])` — [`L1281`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1281)
- `_dynamic_slice_usage_rule(ctx, used_out: set[Usage], **params)` — [`L1263`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1263)
- `_eltwise_eval_rule(prim, ctx, x, **params)` — [`L941`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L941)
- `_eltwise_pull_rule(prim: core.Primitive, ctx: PullRuleContext, block_transform: BlockIndexTransform, **params)` — [`L946`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L946)
- `_eltwise_push_rule(prim: core.Primitive, ctx: PullRuleContext, block_spec: pallas_core.BlockSpec, **params)` — [`L2703`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2703)
- `_eltwise_usage_rule(prim: core.Primitive, ctx: UsageRuleContext, used_out: set[Usage], **params)` — [`L956`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L956)
- `_get_block_aval(bs, aval)` — [`L665`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L665)
- `_get_eval_rule(ctx: KernelEvalContext, ref, *idx, tree)` — [`L1512`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1512)
- `_get_fusion_values(fusion: Callable, args, kwargs, discharge_refs: bool = False, allow_additional_outputs: bool = False)` — [`L791`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L791)
- `_get_in_block_transforms(v, usage)` — [`L583`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L583)
- `_get_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, tree)` — [`L1448`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1448)
- `_get_scalar_prefetch()` — [`L306`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L306)
- `_init_block_transforms(block_specs: tuple[pallas_core.BlockSpec, ...])` — [`L93`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L93)
- `_iota_eval_rule(eval_ctx: KernelEvalContext, *, dimension, shape, dtype, sharding)` — [`L2061`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2061)
- `_iota_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, dtype: jnp.dtype, dimension: int, shape: tuple[int, ...], sharding: jax.sharding.Sharding)` — [`L2081`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2081)
- `_jit_eval_rule(ctx: KernelEvalContext, *args, jaxpr, **kwargs)` — [`L2338`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2338)
- `_jit_pull_block_spec_rule(ctx: PullRuleContext, out_block_specs, *, jaxpr, **kwargs)` — [`L2370`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2370)
- `_jit_usage_rule(ctx, used_out: list[set[Usage]], *, jaxpr: core.ClosedJaxpr, **_)` — [`L2328`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2328)
- `_maybe_static_check(pred: bool, msg: str)` — [`L1193`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1193)
- `_new_index_map(*args)` — [`L2972`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2972)
- `_new_index_map(*args)` — [`L3013`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L3013)
- `_new_index_map(*args)` — [`L3054`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L3054)
- `_null_block_index_trafo(*block_indices)` — [`L61`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L61)
- `_offset_indexer(bs: pallas_core.BlockDim | int | None, indexer, slice_start, slice_size)` — [`L1144`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1144)
- `_pattern_match_lanes_to_sublanes_reshape(aval_in: core.ShapedArray, aval_out: core.ShapedArray)` — [`L2098`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2098)
- `_pjit_push_rule(ctx, *block_specs, jaxpr: core.ClosedJaxpr, **_)` — [`L2807`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2807)
- `_pull_bcast_block_spec(block_transform: BlockIndexTransform, i: int)` — [`L963`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L963)
- `_pull_block_spec(jaxpr: core.Jaxpr, out_block_specs: tuple[pallas_core.BlockSpec, ...], *, read_usage_env: Callable[[core.Var], set[Usage]], scalar_prefetch_handler: Any | None = None, grid_len: int, strict_mode: bool = True)` — [`L605`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L605)
- `_pull_block_transform(jaxpr: core.Jaxpr, out_block_transforms: tuple[BlockIndexTransform, ...], *, read_usage_env: Callable[[core.Var], set[Usage]], scalar_prefetch_handler: Any | None = None, grid_len: int, strict_mode: bool = True)` — [`L468`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L468) — documented in [jax-_src-pallas-fuser-block_spec](../../../../../concepts/jax-_src-pallas-fuser-block_spec.md)
- `_push_bcast_block_spec(block_spec: pallas_core.BlockSpec, i: int, size: int)` — [`L990`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L990)
- `_push_block_spec_jaxpr(jaxpr: core.Jaxpr, *flat_block_specs)` — [`L2558`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2558)
- `_random_bits_eval_rule(eval_ctx: KernelEvalContext, key, bit_width, shape)` — [`L2002`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2002)
- `_random_bits_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, **_)` — [`L2015`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2015)
- `_random_fold_in_eval_rule(eval_ctx: KernelEvalContext, key, msg)` — [`L2042`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2042)
- `_random_fold_in_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, **_)` — [`L2048`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2048)
- `_random_wrap_eval_rule(eval_ctx: KernelEvalContext, arr, *, impl)` — [`L2028`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2028)
- `_random_wrap_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, impl)` — [`L2034`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2034)
- `_read_block_spec(atom: core.Atom)` — [`L498`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L498)
- `_read_block_spec(atom: core.Atom)` — [`L684`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L684)
- `_read_block_spec(atom: core.Atom)` — [`L2575`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2575)
- `_reduce_sum_eval_rule(ctx: KernelEvalContext, x, *, axes: tuple[int, ...], out_sharding)` — [`L2304`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2304)
- `_reduce_sum_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, axes: tuple[int, ...], out_sharding)` — [`L2265`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2265)
- `_reduce_sum_push_rule(ctx: PushRuleContext, block_spec: pallas_core.BlockSpec, *, axes: tuple[int, ...], out_sharding)` — [`L2881`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2881)
- `_remove_nones(shape: tuple[pallas_core.BlockDim | int | None, ...] | None)` — [`L657`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L657)
- `_reshape_eval_rule(eval_ctx: KernelEvalContext, x, *, dimensions, new_sizes, sharding)` — [`L2250`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2250)
- `_reshape_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, dimensions: tuple[int, ...] | None, new_sizes: tuple[int, ...], sharding: jax.sharding.Sharding)` — [`L2117`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2117)
- `_reshape_push_rule(ctx: PullRuleContext, block_spec: pallas_core.BlockSpec, *, dimensions: tuple[int, ...] | None, new_sizes: tuple[int, ...], sharding: jax.sharding.Sharding)` — [`L2838`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2838)
- `_scalar_prefetch_fn(jaxpr)` — [`L559`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L559)
- `_select_block_indices(i)` — [`L87`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L87)
- `_select_n_eval_rule(ctx: KernelEvalContext, *args)` — [`L1083`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1083)
- `_select_n_pull_block_spec_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform)` — [`L1088`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1088)
- `_select_n_push_rule(ctx: PushRuleContext, *args: pallas_core.BlockSpec)` — [`L2749`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2749)
- `_slice(i, b)` — [`L1435`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1435)
- `_slice(i, b)` — [`L1528`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1528)
- `_slice_eval_rule(ctx, x, **params)` — [`L1133`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1133)
- `_slice_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, start_indices: tuple[int, ...], limit_indices: tuple[int, ...], strides: tuple[int, ...] | None)` — [`L1205`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1205)
- `_sp_context(*scalar_prefetch)` — [`L297`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L297)
- `_squeeze_block_spec(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, dimensions: tuple[int, ...])` — [`L1105`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1105)
- `_squeeze_eval_rule(ctx: KernelEvalContext, x: jax.Array, **params: Any)` — [`L1099`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1099)
- `_stack_eval_rule(ctx: KernelEvalContext, *args, axis)` — [`L1621`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1621)
- `_stack_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, axis: int)` — [`L1697`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1697)
- `_stack_push_rule(ctx: PushRuleContext, *block_specs: pallas_core.BlockSpec, axis: int)` — [`L3005`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L3005)
- `_swap_eval_rule(ctx: KernelEvalContext, ref, val, *idx, tree)` — [`L1406`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1406)
- `_swap_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, **kwargs)` — [`L1395`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1395)
- `_tile_eval_rule(eval_ctx: KernelEvalContext, x, reps: tuple[int, ...])` — [`L1885`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1885)
- `_tile_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, reps: tuple[int, ...])` — [`L1910`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1910)
- `_transpose_eval_rule(eval_ctx: KernelEvalContext, x, permutation: tuple[int, ...])` — [`L1834`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1834)
- `_transpose_pull_rule(ctx: PullRuleContext, block_transform: BlockIndexTransform, *, permutation: tuple[int, ...])` — [`L1858`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1858)
- `_transpose_push_rule(ctx: PushRuleContext, block_spec: pallas_core.BlockSpec, *, permutation: tuple[int, ...])` — [`L2714`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2714)
- `_unstack_eval_rule(ctx: KernelEvalContext, x, *, axis)` — [`L1626`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1626)
- `_unstack_pull_rule(ctx: PullRuleContext, out_block_transforms: tuple[BlockIndexTransform, ...], *, axis: int)` — [`L1738`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1738)
- `_unstack_push_rule(ctx: PushRuleContext, block_spec: pallas_core.BlockSpec, *, axis: int)` — [`L3035`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L3035)
- `_unwrap_block_spec_scalar_prefetch(block_spec: pallas_core.BlockSpec)` — [`L331`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L331)
- `_wrap_block_spec_scalar_prefetch(block_spec: pallas_core.BlockSpec, num_grid_args: int)` — [`L310`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L310)
- `_write_block_spec(atom: core.Atom, block_transform: BlockIndexTransform)` — [`L503`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L503)
- `_write_block_spec(atom: core.Atom, block_spec: pallas_core.BlockSpec | pallas_core.NoBlockSpec)` — [`L2582`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2582)
- `block_transform(*block_indices)` — [`L88`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L88)
- `compare(x, y)` — [`L100`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L100)
- `compute_usage(jaxpr: core.Jaxpr, jaxpr_out_usages)` — [`L210`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L210)
- `default_rule(ctx, *args, **params)` — [`L1047`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1047)
- `get_fusion_values(fusion: Callable, *args, **kwargs)` — [`L762`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L762)
- `get_stateful_input_fusion_values(fusion: Callable, *args, **kwargs)` — [`L772`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L772)
- `get_stateful_output_fusion_values(fusion: Callable, *args, **kwargs)` — [`L782`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L782)
- `kernel_fn(program_ids, scalar_prefetch, *args, **kwargs)` — [`L689`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L689) — documented in [jax-_src-pallas-fuser-block_spec](../../../../../concepts/jax-_src-pallas-fuser-block_spec.md)
- `make_block_transform(child_index: int)` — [`L1675`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1675)
- `make_block_transform(child_index: int)` — [`L1723`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1723)
- `make_kernel_function(jaxpr: core.Jaxpr, consts, in_tree, out_tree, read_usage_env, in_block_specs, block_spec_env, scalar_prefetch_handler, grid_len)` — [`L642`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L642)
- `make_new_idx_map(block_index_transform)` — [`L136`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L136)
- `make_scalar_prefetch_handler(*args)` — [`L180`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L180)
- `make_transform(block_transform, contraction_index, out_index)` — [`L1352`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1352)
- `new_block_index_transform(*idxs)` — [`L966`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L966)
- `new_block_index_transform(*idxs)` — [`L1115`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1115)
- `new_block_index_transform(*idxs)` — [`L1245`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1245)
- `new_block_index_transform(*idxs)` — [`L1288`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1288)
- `new_block_index_transform(*idxs)` — [`L1491`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1491)
- `new_block_index_transform(*idxs)` — [`L1676`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1676)
- `new_block_index_transform(*idxs)` — [`L1724`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1724)
- `new_block_index_transform(*idxs)` — [`L1756`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1756)
- `new_block_index_transform(*idxs)` — [`L1817`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1817)
- `new_block_index_transform(*idxs)` — [`L1875`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1875)
- `new_block_index_transform(*idxs)` — [`L1943`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1943)
- `new_block_index_transform(*idxs)` — [`L2193`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2193)
- `new_block_index_transform(*idxs)` — [`L2284`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2284)
- `new_idx_map(*args)` — [`L140`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L140)
- `new_index_map(*args_and_scalar_prefetch)` — [`L317`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L317)
- `new_index_map(*args)` — [`L340`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L340)
- `new_index_map(*args)` — [`L2728`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2728)
- `new_index_map(*args)` — [`L2872`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2872)
- `new_index_map(*args)` — [`L2903`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2903)
- `new_index_map(*args)` — [`L2943`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2943)
- `new_kernel_fn(values, *args, **kwargs)` — [`L824`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L824)
- `pull_block_spec(f: Callable, out_block_specs: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...], *, scalar_prefetch_handler: Any | None = None, grid_len: int | None = None, strict_mode: bool = True)` — [`L351`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L351)
- `push_block_spec(f: Callable, *in_spec_args, **in_spec_kwargs)` — [`L2540`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2540)
- `read_env(atom)` — [`L695`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L695)
- `read_usage_env(atom: core.Atom)` — [`L215`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L215)
- `read_usage_env(_: core.Var)` — [`L2345`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2345)
- `read_usage_env(_: core.Var)` — [`L2378`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2378)
- `read_usage_env(_: core.Var)` — [`L2412`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2412)
- `read_usage_env(_: core.Var)` — [`L2445`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2445)
- `read_usage_env(_: core.Var)` — [`L2479`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2479)
- `read_usage_env(_: core.Var)` — [`L2512`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2512)
- `register_binop_rule(prim: core.Primitive)` — [`L1054`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1054)
- `register_default_eval_rule(prim: core.Primitive)` — [`L1046`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1046)
- `register_eltwise_rule(prim: core.Primitive)` — [`L2812`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2812)
- `register_eval_rule(prim: core.Primitive)` — [`L899`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L899)
- `register_pull_block_spec_rule(prim: core.Primitive)` — [`L925`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L925)
- `register_push_block_spec_rule(prim: core.Primitive)` — [`L2636`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2636)
- `register_usage_rule(prim: core.Primitive)` — [`L869`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L869)
- `scalar_prefetch_getter(*sp_inputs)` — [`L181`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L181)
- `transform(*idxs)` — [`L1355`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L1355)
- `wrapped(*args, **kwargs)` — [`L359`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L359)
- `wrapper(f: UsageRuleFn)` — [`L873`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L873)
- `wrapper(f: EvalRuleFn)` — [`L902`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L902)
- `wrapper(f: PullBlockSpecRuleFn)` — [`L929`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L929)
- `wrapper(*args, **kwargs)` — [`L2545`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2545)
- `wrapper(f: PushBlockSpecRuleFn)` — [`L2640`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2640)
- `write_env(var, val)` — [`L702`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L702)
- `write_usage_env(atom: core.Atom, usage: set[Usage])` — [`L219`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L219)

## Module values
- `_illegal` — [`L284`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L284)
- `_sp_env` — [`L293`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L293)
- `_unwrap_cache` — [`L328`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L328)
- `eval_rules` — [`L896`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L896)
- `no_block_index_transform` — [`L84`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L84)
- `pull_block_spec_rules` — [`L58`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L58)
- `push_block_spec_rules` — [`L2622`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2622)
- `register_binop_push_rule` — [`L2686`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L2686)
- `usage_rules` — [`L866`](../../../../../../../../raw/code/jax/jax/_src/pallas/fuser/block_spec.py#L866)

