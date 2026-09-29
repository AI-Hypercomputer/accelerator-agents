---
title: 'Module: easydel/layers/linears/_linear_quantized.py'
type: catalog
provenance: extracted
module: easydel/layers/linears/_linear_quantized.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.linears._linear_quantized`/
symbols:
  ParallelLinearQuantized.from_quantized: ParallelLinearQuantized#from_quantized().
  ParallelLinearQuantized.__call__: ParallelLinearQuantized#__call__().
  ParallelLinearQuantized._qmm_runtime_kwargs: ParallelLinearQuantized#_qmm_runtime_kwargs().
  ParallelLinearQuantized._resolve_shard_specs: ParallelLinearQuantized#_resolve_shard_specs().
  ParallelLinearQuantized._distributed_quantized_matmul: ParallelLinearQuantized#_distributed_quantized_matmul().
  Array: Array.
  ParallelLinearQuantized.restage: ParallelLinearQuantized#restage().
  ParallelLinearQuantized.craft_sharding: ParallelLinearQuantized#craft_sharding().
  ParallelLinearQuantized._direction: ParallelLinearQuantized#_direction.
  ParallelLinearQuantized._resolve_ejkernel_params: ParallelLinearQuantized#_resolve_ejkernel_params().
  RowParallelLinearQuantized: RowParallelLinearQuantized#
  ColumnParallelLinearQuantized: ColumnParallelLinearQuantized#
  ParallelLinearQuantized: ParallelLinearQuantized#
  ParallelLinearQuantized._quantize_runtime: ParallelLinearQuantized#_quantize_runtime().
  ParallelLinearQuantized._quantize_array: ParallelLinearQuantized#_quantize_array().
  ParallelLinearQuantized.dtype: ParallelLinearQuantized#dtype.
  _lookup_qmm_policy_entry: _lookup_qmm_policy_entry().
  ParallelLinearQuantized.wqdtype: ParallelLinearQuantized#wqdtype().
  ParallelLinearQuantized.__repr__: ParallelLinearQuantized#__repr__().
  _effective_ejkernel_group_size: _effective_ejkernel_group_size().
  _sanitize_spec_for_shape: _sanitize_spec_for_shape().
  ParallelLinearQuantized.__init__: ParallelLinearQuantized#__init__().
  ParallelLinearQuantized._local_matmul: ParallelLinearQuantized#_local_matmul().
  _spec_for_mesh: _spec_for_mesh().
  ParallelLinearQuantized.use_bias: ParallelLinearQuantized#use_bias.
  ParallelLinearQuantized._dequantize_array: ParallelLinearQuantized#_dequantize_array().
  ParallelLinearQuantized._ej_group_size: ParallelLinearQuantized#_ej_group_size.
  ParallelLinearQuantized._mapped: ParallelLinearQuantized#_mapped().
  ParallelLinearQuantized.config: ParallelLinearQuantized#config.
  ParallelLinearQuantized.quant_kernel: ParallelLinearQuantized#quant_kernel.
  ParallelLinearQuantized.quant_scales: ParallelLinearQuantized#quant_scales.
  ParallelLinearQuantized.quant_biases: ParallelLinearQuantized#quant_biases.
  logger: logger.
  _spec_matches_kernel_parallel_layout: _spec_matches_kernel_parallel_layout().
  ParallelLinearQuantized._direct_matmul: ParallelLinearQuantized#_direct_matmul().
  _axis_names: _axis_names().
  ParallelLinearQuantized.qmm_tpu_path: ParallelLinearQuantized#qmm_tpu_path.
  _policy_mode_key: _policy_mode_key().
  _extract_tp_axis_name: _extract_tp_axis_name().
  _quantized_linear_craft_spec: _quantized_linear_craft_spec().
  _reconcile_input_k_dim: _reconcile_input_k_dim().
  _spec_is_sharded: _spec_is_sharded().
  ParallelLinearQuantized.out_features: ParallelLinearQuantized#out_features.
  ParallelLinearQuantized.qmm_strict_fuse: ParallelLinearQuantized#qmm_strict_fuse.
  ParallelLinearQuantized.qmm_allow_dense_fallback: ParallelLinearQuantized#qmm_allow_dense_fallback.
  ParallelLinearQuantized.bias: ParallelLinearQuantized#bias.
  ParallelLinearQuantized.__str__: ParallelLinearQuantized#__str__().
  _EJKERNEL_GROUP_SIZES: _EJKERNEL_GROUP_SIZES.
  _quantized_linear_sharding_fn: _quantized_linear_sharding_fn().
  ParallelLinearQuantized.in_features: ParallelLinearQuantized#in_features.
  ParallelLinearQuantized.param_dtype: ParallelLinearQuantized#param_dtype.
  ParallelLinearQuantized.precision: ParallelLinearQuantized#precision.
  ParallelLinearQuantized.qmm_fuse: ParallelLinearQuantized#qmm_fuse.
  default_kernel_init: default_kernel_init.
  default_bias_init: default_bias_init.
  _QMM_NON_AFFINE_MODES: _QMM_NON_AFFINE_MODES.
  _policy_size_key: _policy_size_key().
  _mesh_matches: _mesh_matches().
  _pick_mesh_from_arrays: _pick_mesh_from_arrays().
  _mesh_partition_product: _mesh_partition_product().
  _pick_tensor_axis_name: _pick_tensor_axis_name().
  ParallelLinearQuantized.kernel_init: ParallelLinearQuantized#kernel_init.
  ParallelLinearQuantized.bias_init: ParallelLinearQuantized#bias_init.
  ParallelLinearQuantized.qmm_platform: ParallelLinearQuantized#qmm_platform.
  ParallelLinearQuantized.qmm_use_best_config: ParallelLinearQuantized#qmm_use_best_config.
  ParallelLinearQuantized.qmm_tpu_auto_xla_max_m: ParallelLinearQuantized#qmm_tpu_auto_xla_max_m.
  ParallelLinearQuantized.qmm_policy_table: ParallelLinearQuantized#qmm_policy_table.
  ParallelLinearQuantized.qmm_allow_input_all_gather: ParallelLinearQuantized#qmm_allow_input_all_gather.
  ParallelLinearQuantized._shape_of: ParallelLinearQuantized#_shape_of().
  _QMM_DEFAULT_POLICY_TABLE._QMM_DEFAULT_POLICY_TABLE: _QMM_DEFAULT_POLICY_TABLE._QMM_DEFAULT_POLICY_TABLE.
  Axis: Axis.
  Size: Size.
  ParallelLinearQuantized.rngs: ParallelLinearQuantized#rngs.
  RowParallelLinearQuantized._direction: RowParallelLinearQuantized#_direction.
  ColumnParallelLinearQuantized._direction: ColumnParallelLinearQuantized#_direction.
---
# Module: [`easydel/layers/linears/_linear_quantized.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py)

## Classes
### `ColumnParallelLinearQuantized`  ·  implements/extends ParallelLinearQuantized
- def: [`easydel/layers/linears/_linear_quantized.py:1266`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1266)
- doc: Column-parallel variant of quantized linear layer.
- signature: `class ColumnParallelLinearQuantized(ParallelLinearQuantized):`
- protocol/private: `_direction`[`L1286`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1286)
- uses (calls/refs, reference-scoped): [`ParallelLinearQuantized`](_linear_quantized.md#ParallelLinearQuantized)
- used by: [`to_quantized`](_linear.md#ParallelLinear.to_quantized), [`ParallelLinearQuantized`](_linear_quantized.md#ParallelLinearQuantized), [`_quantized_friend`](_linear.md#ParallelLinear._quantized_friend), [`is_quantized`](../../infra/base_module.md#EasyDeLBaseModule.is_quantized)

### `ParallelLinearQuantized`  ·  implements/extends Module
- def: [`easydel/layers/linears/_linear_quantized.py:498`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L498)
- doc: A quantized linear transformation layer with parallel execution support.
- signature: `class ParallelLinearQuantized(nn.Module):`
- members:
  - `__call__(self, inputs: Array, w: Array | None = None)` — [`L1111`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1111) — Apply the quantized linear transformation to inputs.
  - `__init__(self, in_features: int, out_features: int, *, use_bias: bool = True, dtype: Dtype | None = None, param_dtype: Dtype = jnp.float32, precision: PrecisionLike = None, kernel_init: Initializer = default_kernel_init, bias_init: Initializer = default_bias_init, config: QuantizationConfig, qmm_platform: tp.Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None, qmm_use_best_config: bool | None = None, qmm_fuse: bool | None = None, qmm_strict_fuse: bool | None = None, qmm_allow_dense_fallback: bool | None = None, qmm_tpu_path: tp.Literal["hybrid", "packed", "predecode"] | None = None, qmm_tpu_auto_xla_max_m: int | None = 1024, qmm_policy_table: dict[str, tp.Any] | None = None, qmm_allow_input_all_gather: bool = False, rngs: rnglib.Rngs)` — [`L545`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L545) — Initialize a quantized parallel linear layer.
  - `__repr__(self)` — [`L1219`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1219) — Return a string representation of the quantized layer.
  - `__str__(self)` — [`L1234`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1234) — Return a string representation of the quantized layer.
  - `_dequantize_array(self, wq: jax.Array, scale: jax.Array, bias: jax.Array | None)` — [`L822`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L822) — Dequantize weights back to full precision for computation.
  - `_distributed_quantized_matmul(self, inputs_2d: Array, kernel_value: Array, scale_value: Array, bias_value: Array | None, *, group_size: int, bits: int, mode: str)` — [`L1026`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1026) — Run quantized matmul under shard_map with explicit TP communication.
  - `_qmm_runtime_kwargs(self, backend: str, *, m_tokens: int | None = None, quant_mode: str | None = None)` — [`L652`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L652) — Resolve per-call ejkernel quantized_matmul controls.
  - `_quantize_array(self, array: jax.Array)` — [`L757`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L757) — Quantize an array according to the configured quantization type.
  - `_quantize_runtime(self, array: jax.Array)` — [`L799`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L799) — Quantize activations at runtime if configured.
  - `_resolve_ejkernel_params(self)` — [`L750`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L750) — Resolve ejkernel quantization parameters from config.
  - `_resolve_shard_specs(self, mesh: jax.sharding.Mesh, inputs_2d: Array, kernel_value: Array, scale_value: Array, bias_value: Array | None, is_tpu: bool)` — [`L940`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L940) — Resolve partition specs for distributed quantized matmul.
  - `craft_sharding(self, *, partition_manager=None, **_kwargs)` — [`L1170`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1170) — Return dynamic partition specs for quantized parameters.
  - `from_quantized(self, rngs: rnglib.Rngs | None = None)` — [`L852`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L852) — Convert this quantized module back to a regular Linear module. — documented in [easydel-layers-linears-_linear](../../../../concepts/easydel-layers-linears-_linear.md)
  - `restage(self, kernel: jax.Array, bias: jax.Array | None)` — [`L909`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L909) — Update the layer's weights by quantizing new kernel values.
  - `wqdtype(self)` — [`L1210`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1210) — Get the weight quantization data type.
  - `bias` — [`L650`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L650)
  - `bias_init` — [`L627`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L627)
  - `config` — [`L628`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L628)
  - `dtype` — [`L623`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L623)
  - `in_features` — [`L620`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L620)
  - `kernel_init` — [`L626`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L626)
  - `out_features` — [`L621`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L621)
  - `param_dtype` — [`L624`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L624)
  - `precision` — [`L625`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L625)
  - `qmm_allow_dense_fallback` — [`L633`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L633)
  - `qmm_allow_input_all_gather` — [`L637`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L637)
  - `qmm_fuse` — [`L631`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L631)
  - `qmm_platform` — [`L629`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L629)
  - `qmm_policy_table` — [`L636`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L636)
  - `qmm_strict_fuse` — [`L632`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L632)
  - `qmm_tpu_auto_xla_max_m` — [`L635`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L635)
  - `qmm_tpu_path` — [`L634`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L634)
  - `qmm_use_best_config` — [`L630`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L630)
  - `quant_biases` — [`L647`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L647)
  - `quant_kernel` — [`L645`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L645)
  - `quant_scales` — [`L646`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L646)
  - `rngs` — [`L638`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L638)
  - `use_bias` — [`L622`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L622)
- protocol/private: `_direct_matmul`[`L1047`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1047), `_direction`[`L543`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L543), `_ej_group_size`[`L640`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L640), `_local_matmul`[`L1071`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1071), `_mapped`[`L1099`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1099), `_shape_of`[`L1187`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1187)
- uses (calls/refs, reference-scoped): [`ColumnParallelLinear`](_linear.md#ColumnParallelLinear), [`RowParallelLinear`](_linear.md#RowParallelLinear), [`QuantizationConfig`](../quantization/_configs.md#QuantizationConfig), [`QuantizationType`](../quantization/_configs.md#QuantizationType), [`resolve_ejkernel_quant_params`](../quantization/_configs.md#resolve_ejkernel_quant_params), [`quantize`](../quantization/_quants.md#quantize), [`group_size`](../quantization/_configs.md#QuantizationConfig.group_size), [`dtype`](../quantization/_configs.md#QuantizationConfig.dtype), [`kernel`](_linear.md#ParallelLinear.kernel), [`Array`](_linear_quantized.md#Array), [`resolve_safe_sharding`](../_sharding.md#resolve_safe_sharding), [`ColumnParallelLinearQuantized`](_linear_quantized.md#ColumnParallelLinearQuantized), [`RowParallelLinearQuantized`](_linear_quantized.md#RowParallelLinearQuantized), [`bias`](_linear.md#ParallelLinear.bias), [`_lookup_qmm_policy_entry`](_linear_quantized.md#_lookup_qmm_policy_entry), [`_effective_ejkernel_group_size`](_linear_quantized.md#_effective_ejkernel_group_size), [`_sanitize_spec_for_shape`](_linear_quantized.md#_sanitize_spec_for_shape), [`runtime_dtype`](../quantization/_configs.md#QuantizationConfig.runtime_dtype), [`_spec_for_mesh`](_linear_quantized.md#_spec_for_mesh), [`_spec_matches_kernel_parallel_layout`](_linear_quantized.md#_spec_matches_kernel_parallel_layout), [`_axis_names`](_linear_quantized.md#_axis_names), [`_extract_tp_axis_name`](_linear_quantized.md#_extract_tp_axis_name), [`_quantized_linear_craft_spec`](_linear_quantized.md#_quantized_linear_craft_spec), [`_reconcile_input_k_dim`](_linear_quantized.md#_reconcile_input_k_dim), [`_spec_is_sharded`](_linear_quantized.md#_spec_is_sharded), [`_pick_mesh_from_arrays`](_linear_quantized.md#_pick_mesh_from_arrays), [`_pick_tensor_axis_name`](_linear_quantized.md#_pick_tensor_axis_name), [`default_bias_init`](_linear_quantized.md#default_bias_init), [`default_kernel_init`](_linear_quantized.md#default_kernel_init)
- used by: [`ColumnParallelLinearQuantized`](_linear_quantized.md#ColumnParallelLinearQuantized), [`RowParallelLinearQuantized`](_linear_quantized.md#RowParallelLinearQuantized), [`is_quantized`](../../infra/base_module.md#EasyDeLBaseModule.is_quantized)

### `RowParallelLinearQuantized`  ·  implements/extends ParallelLinearQuantized
- def: [`easydel/layers/linears/_linear_quantized.py:1243`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1243)
- doc: Row-parallel variant of quantized linear layer.
- signature: `class RowParallelLinearQuantized(ParallelLinearQuantized):`
- protocol/private: `_direction`[`L1263`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L1263)
- uses (calls/refs, reference-scoped): [`ParallelLinearQuantized`](_linear_quantized.md#ParallelLinearQuantized)
- used by: [`to_quantized`](_linear.md#ParallelLinear.to_quantized), [`ParallelLinearQuantized`](_linear_quantized.md#ParallelLinearQuantized), [`_quantized_friend`](_linear.md#ParallelLinear._quantized_friend), [`is_quantized`](../../infra/base_module.md#EasyDeLBaseModule.is_quantized)

## Functions
- `_axis_names(axis_spec: tp.Any)` — [`L333`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L333)
- `_effective_ejkernel_group_size(mode: str, requested_group_size: int, array_shape: tuple[int, ...])` — [`L169`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L169) — Clamp ejkernel group size to the local packed weight layout when possible.
- `_extract_tp_axis_name(kernel_spec: jax.sharding.PartitionSpec, direction: tp.Literal["row", "column"] | None, mesh: jax.sharding.Mesh)` — [`L343`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L343)
- `_lookup_qmm_policy_entry(*, backend: str, mode: str | None, m_tokens: int | None, tpu_small_threshold: int | None, policy_table: dict[str, tp.Any] | None)` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L154)
- `_mesh_matches(lhs: jax.sharding.Mesh, rhs: jax.sharding.Mesh)` — [`L205`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L205)
- `_mesh_partition_product(mesh: jax.sharding.Mesh, axis_spec: tp.Any)` — [`L259`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L259)
- `_pick_mesh_from_arrays(*arrays: jax.Array | None)` — [`L232`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L232)
- `_pick_tensor_axis_name(mesh: jax.sharding.Mesh)` — [`L368`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L368) — Pick a multi-device tensor axis from mesh, preferring canonical names.
- `_policy_mode_key(mode: str | None)` — [`L139`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L139)
- `_policy_size_key(m_tokens: int | None, threshold: int | None)` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L148)
- `_quantized_linear_craft_spec(*, direction: tp.Literal["row", "column"] | None, use_bias: bool, mode: str, group_size: int, needs_biases: bool)` — [`L418`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L418) — Craft dynamic sharding specs for quantized linear parameters.
- `_quantized_linear_sharding_fn(*, direction: tp.Literal["row", "column"] | None, param_name: str, mode: str, group_size: int, needs_biases: bool)` — [`L380`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L380) — Return sharding dynamic-axes for a quantized linear parameter.
- `_reconcile_input_k_dim(local_inputs: Array, local_kernel: Array, direction: tp.Literal["row", "column"] | None, tp_axis_name: str | None, *, allow_column_all_gather: bool = False)` — [`L454`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L454) — Align the K dimension of local inputs with the local kernel shard.
- `_sanitize_spec_for_shape(spec: jax.sharding.PartitionSpec, shape: tuple[int, ...], mesh: jax.sharding.Mesh)` — [`L272`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L272)
- `_spec_for_mesh(array: jax.Array | None, mesh: jax.sharding.Mesh)` — [`L250`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L250)
- `_spec_is_sharded(spec: jax.sharding.PartitionSpec)` — [`L299`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L299)
- `_spec_matches_kernel_parallel_layout(kernel_spec: jax.sharding.PartitionSpec, aux_spec: jax.sharding.PartitionSpec, direction: tp.Literal["row", "column"] | None)` — [`L303`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L303) — Return whether aux tensor sharding is compatible with kernel sharding.

## Module values
- `Array` — [`L79`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L79)
- `Axis` — [`L80`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L80)
- `Size` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L81)
- `_EJKERNEL_GROUP_SIZES` — [`L87`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L87)
- `_QMM_DEFAULT_POLICY_TABLE` — [`L88`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L88)
- `_QMM_NON_AFFINE_MODES` — [`L86`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L86)
- `default_bias_init` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L84)
- `default_kernel_init` — [`L83`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L83)
- `logger` — [`L73`](../../../../../../../raw/code/EasyDeL/easydel/layers/linears/_linear_quantized.py#L73)

