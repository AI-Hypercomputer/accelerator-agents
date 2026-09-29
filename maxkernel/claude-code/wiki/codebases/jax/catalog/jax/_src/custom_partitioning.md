---
title: 'Module: jax/_src/custom_partitioning.py'
type: catalog
provenance: extracted
module: jax/_src/custom_partitioning.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.custom_partitioning`/
symbols:
  custom_partitioning.__call__: custom_partitioning#__call__().
  _custom_partitioning_lowering_rule: _custom_partitioning_lowering_rule().
  _custom_partitioning_partition: _custom_partitioning_partition().
  custom_partitioning_p: custom_partitioning_p.
  _custom_partitioning_lowering_rule.to_mesh_pspec_sharding: _custom_partitioning_lowering_rule().to_mesh_pspec_sharding().
  custom_partitioning.def_partition: custom_partitioning#def_partition().
  _custom_partitioning_infer_sharding_from_operands: _custom_partitioning_infer_sharding_from_operands().
  _custom_partitioning_propagate_user_sharding: _custom_partitioning_propagate_user_sharding().
  _check_for_tracers: _check_for_tracers().
  custom_partitioning.decode_shardings: custom_partitioning#decode_shardings.
  custom_partitioning: custom_partitioning#
  _flatten_sharding: _flatten_sharding().
  _ShardingCallbackInfo.unflatten_arg_shape: _ShardingCallbackInfo#unflatten_arg_shape().
  _to_hlo_sharding: _to_hlo_sharding().
  _to_jax_shape: _to_jax_shape().
  _ShardingCallbackInfo.unflatten_arg_shapes: _ShardingCallbackInfo#unflatten_arg_shapes().
  _sharding_callbacks: _sharding_callbacks.
  custom_partitioning.fun: custom_partitioning#fun.
  custom_partitioning.static_argnums: custom_partitioning#static_argnums.
  custom_partitioning.sharding_rule: custom_partitioning#sharding_rule.
  _to_jax_sharded_shape: _to_jax_sharded_shape().
  _custom_partitioning_impl: _custom_partitioning_impl().
  _CUSTOM_PARTITIONING_CALL_NAME: _CUSTOM_PARTITIONING_CALL_NAME.
  _pack_result_sharding: _pack_result_sharding().
  custom_partitioning.propagate_user_sharding: custom_partitioning#propagate_user_sharding.
  custom_partitioning.infer_sharding_from_operands: custom_partitioning#infer_sharding_from_operands.
  custom_partitioning.partition: custom_partitioning#partition.
  _resolve_kwargs: _resolve_kwargs().
  _ShardingCallbackInfo: _ShardingCallbackInfo#
  _ShardingCallbackInfo.to_mesh_pspec_sharding: _ShardingCallbackInfo#to_mesh_pspec_sharding.
  _ShardingCallbackInfo.in_tree: _ShardingCallbackInfo#in_tree.
  _custom_partitioning_abstract_eval: _custom_partitioning_abstract_eval().
  _ShardingCallbackInfo.__init__: _ShardingCallbackInfo#__init__().
  _ShardingCallbackInfo.propagate_user_sharding: _ShardingCallbackInfo#propagate_user_sharding.
  _ShardingCallbackInfo.partition: _ShardingCallbackInfo#partition.
  _ShardingCallbackInfo.out_tree: _ShardingCallbackInfo#out_tree.
  _ShardingCallbackInfo.infer_sharding_from_operands: _ShardingCallbackInfo#infer_sharding_from_operands.
  _ShardingCallbackInfo.module_context: _ShardingCallbackInfo#module_context.
  _ShardingCallbackInfo.mesh: _ShardingCallbackInfo#mesh.
  _ShardingCallbackInfo.static_args: _ShardingCallbackInfo#static_args.
  custom_partitioning.__init__: custom_partitioning#__init__().
  custom_partitioning.__getattr__: custom_partitioning#__getattr__.
---
# Module: [`jax/_src/custom_partitioning.py`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py)

## Classes
### `_ShardingCallbackInfo`
- def: [`jax/_src/custom_partitioning.py:60`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L60)
- signature: `class _ShardingCallbackInfo:`
- members:
  - `unflatten_arg_shape(self, s, sharding)` — [`L75`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L75)
  - `unflatten_arg_shapes(self, arg_shapes, arg_shardings)` — [`L80`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L80)
  - `in_tree` — [`L68`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L68)
  - `infer_sharding_from_operands` — [`L70`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L70)
  - `mesh` — [`L72`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L72)
  - `module_context` — [`L71`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L71)
  - `out_tree` — [`L69`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L69)
  - `partition` — [`L66`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L66)
  - `propagate_user_sharding` — [`L65`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L65)
  - `static_args` — [`L73`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L73)
  - `to_mesh_pspec_sharding` — [`L67`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L67)
- protocol/private: `__init__`[`L62`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L62)
- uses (calls/refs, reference-scoped): [`_to_jax_sharded_shape`](custom_partitioning.md#_to_jax_sharded_shape)
- used by: [`_custom_partitioning_lowering_rule`](custom_partitioning.md#_custom_partitioning_lowering_rule)

### `custom_partitioning`
- def: [`jax/_src/custom_partitioning.py:265`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L265)
- doc: Inserts a CustomCallOp into the XLA graph with custom SPMD lowering rules.
- signature: `class custom_partitioning:`
- members:
  - `def_partition(self, partition, infer_sharding_from_operands=None, propagate_user_sharding=None, decode_shardings=True, sharding_rule=None, *, reduction_factors=(), need_replication_factors=(), permutation_factors=(), **factor_sizes)` — [`L478`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L478)
  - `decode_shardings` — [`L486`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L486)
  - `fun` — [`L469`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L469)
  - `infer_sharding_from_operands` — [`L473`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L473)
  - `partition` — [`L470`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L470)
  - `propagate_user_sharding` — [`L472`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L472)
  - `sharding_rule` — [`L474`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L474)
  - `static_argnums` — [`L471`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L471)
- protocol/private: `__call__`[`L508`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L508), `__getattr__`[`L476`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L476), `__init__`[`L468`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L468)
- uses (calls/refs, reference-scoped): [`bind`](core.md#Primitive.bind), [`typeof`](core.md#typeof), [`tree_unflatten`](tree_util.md#tree_unflatten), [`tree_flatten`](tree_util.md#tree_flatten), [`ClosedJaxpr`](core.md#ClosedJaxpr), [`debug_info`](api_util.md#debug_info), [`flatten_fun_nokwargs`](api_util.md#flatten_fun_nokwargs), [`shape`](mesh.md#Mesh.shape), [`argnums_partial`](api_util.md#argnums_partial), [`physical_mesh`](mesh.md#ResourceEnv.physical_mesh), [`thread_resources`](mesh.md#thread_resources), [`extend_axis_env_nd`](core.md#extend_axis_env_nd), [`use_shardy_partitioner`](config.md#use_shardy_partitioner), [`custom_partitioning_p`](custom_partitioning.md#custom_partitioning_p), [`env`](mesh.md#_ThreadResourcesLocalState.env), [`str_to_sdy_sharding_rule`](custom_partitioning_sharding_rule.md#str_to_sdy_sharding_rule), [`SdyShardingRule`](custom_partitioning_sharding_rule.md#SdyShardingRule), [`_check_for_tracers`](custom_partitioning.md#_check_for_tracers), [`_resolve_kwargs`](custom_partitioning.md#_resolve_kwargs)
- used by: [`_dot_product_attention_bwd_p_wrapper`](cudnn/fused_attention_stablehlo.md#_dot_product_attention_bwd_p_wrapper), [`_dot_product_attention_fp8_bwd_p_wrapper`](cudnn/fused_attention_stablehlo.md#_dot_product_attention_fp8_bwd_p_wrapper), [`_scaled_matmul_lower`](cudnn/scaled_matmul_stablehlo.md#_scaled_matmul_lower), [`_dot_product_attention_bwd_lower`](cudnn/fused_attention_stablehlo.md#_dot_product_attention_bwd_lower), [`_dot_product_attention_fp8_bwd_lower`](cudnn/fused_attention_stablehlo.md#_dot_product_attention_fp8_bwd_lower), [`_dot_product_attention_fp8_fwd_lower`](cudnn/fused_attention_stablehlo.md#_dot_product_attention_fp8_fwd_lower), [`_dot_product_attention_fwd_lower`](cudnn/fused_attention_stablehlo.md#_dot_product_attention_fwd_lower)

## Functions
- `_check_for_tracers(x)` — [`L255`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L255)
- `_custom_partitioning_abstract_eval(*avals, call, in_tree, out_tree, propagate_user_sharding, partition, infer_sharding_from_operands, decode_shardings, sharding_rule, static_args)` — [`L230`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L230)
- `_custom_partitioning_impl(*args, call, in_tree, out_tree, propagate_user_sharding, partition, infer_sharding_from_operands, decode_shardings, sharding_rule, static_args)` — [`L242`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L242)
- `_custom_partitioning_infer_sharding_from_operands(arg_shapes, arg_shardings, result_shape, backend_string)` — [`L206`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L206)
- `_custom_partitioning_lowering_rule(ctx: mlir.LoweringRuleContext, *values, call, in_tree, out_tree, propagate_user_sharding, partition, infer_sharding_from_operands, decode_shardings, sharding_rule, static_args)` — [`L570`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L570)
- `_custom_partitioning_partition(arg_shapes, arg_shardings, result_shape, result_sharding, backend_string)` — [`L151`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L151)
- `_custom_partitioning_propagate_user_sharding(user_sharding, shape, backend_string)` — [`L120`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L120)
- `_flatten_sharding(tree, shardings, shapes)` — [`L111`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L111)
- `_pack_result_sharding(shape, result_shardings)` — [`L104`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L104)
- `_resolve_kwargs(fun, args, kwargs)` — [`L51`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L51)
- `_to_hlo_sharding(sharding, num_dimensions)` — [`L145`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L145)
- `_to_jax_shape(s)` — [`L94`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L94)
- `_to_jax_sharded_shape(s, sharding)` — [`L98`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L98)
- `to_mesh_pspec_sharding(hlo_sharding: xc.HloSharding | None, ndim)` — [`L601`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L601)

## Module values
- `_CUSTOM_PARTITIONING_CALL_NAME` — [`L91`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L91)
- `_sharding_callbacks` — [`L89`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L89)
- `custom_partitioning_p` — [`L225`](../../../../../../raw/code/jax/jax/_src/custom_partitioning.py#L225)

