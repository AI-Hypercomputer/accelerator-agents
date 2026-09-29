---
title: 'Module: tokamax/_src/ops/op.py'
type: catalog
provenance: extracted
module: tokamax/_src/ops/op.py
status: fresh
symbol_base: scip-python python tokamax 0.0.0 `tokamax._src.ops.op`/
symbols:
  BoundArguments: BoundArguments#
  Op: Op#
  Op._get_heuristics_config: Op#_get_heuristics_config().
  Op.bind: Op#bind().
  Op._get_autotuning_configs: Op#_get_autotuning_configs().
  Op.supported_on: Op#supported_on().
  BoundArguments.op: BoundArguments#op.
  Op._fwd: Op#_fwd().
  Op.fwd: Op#fwd().
  BoundArguments.args: BoundArguments#args().
  BoundArguments.autotune: BoundArguments#autotune().
  BoundArguments.arguments: BoundArguments#arguments.
  BoundArguments.get_config: BoundArguments#get_config().
  Op.vjp: Op#vjp.
  BoundArguments.cached_autotuning_data: BoundArguments#cached_autotuning_data().
  Op._get_autotuning_cache_key: Op#_get_autotuning_cache_key().
  BoundArguments.kwargs: BoundArguments#kwargs().
  Op.get_autotuning_cache: Op#get_autotuning_cache().
  Op.config: Op#config.
  BoundArguments.benchmark: BoundArguments#benchmark().
  _Config: _Config.
  BoundArguments.autotuning_configs: BoundArguments#autotuning_configs().
  Op.replace: Op#replace().
  BoundArguments.heuristics_config: BoundArguments#heuristics_config().
  BoundArguments.autotuning_cache_key: BoundArguments#autotuning_cache_key().
  BoundArguments.vjp_arg_spec: BoundArguments#vjp_arg_spec().
  BoundArguments.default_config: BoundArguments#default_config().
  _P: _P.
  Op.__call__: Op#__call__().
  _T: _T.
  NullConfig: NullConfig#
  Residuals.tree_flatten: Residuals#tree_flatten().
  infer_device_kind: infer_device_kind().
  _AUTOTUNING_CACHE._AUTOTUNING_CACHE: _AUTOTUNING_CACHE._AUTOTUNING_CACHE.
  Op._capture_batched_args: Op#_capture_batched_args().
  BoundArguments.vmap_axis_sizes: BoundArguments#vmap_axis_sizes().
  _get_arg_spec_adapter: _get_arg_spec_adapter().
  infer_devices: infer_devices().
  Op.f: Op#f().
  Op.signature: Op#signature().
  Op._fwd_signature: Op#_fwd_signature().
  BoundArguments.__get_pydantic_core_schema__: BoundArguments#__get_pydantic_core_schema__().
  BOUND_ARGS_ADAPTER: BOUND_ARGS_ADAPTER.
  _R: _R.
  BoundArguments.signature: BoundArguments#signature().
  BoundArguments._bound_args: BoundArguments#_bound_args().
  Op.config_cls: Op#config_cls.
  _abstractify: _abstractify().
  _Key: _Key.
  Residuals: Residuals#
  AUTO: AUTO#
  BoundArguments.validate: BoundArguments#validate().
  get_autotuning_cache_overlay_state: get_autotuning_cache_overlay_state().
  BoundArguments.replace: BoundArguments#replace.
  Op.fwd_flat: Op#fwd_flat().
  _NULL_CONFIG._NULL_CONFIG: _NULL_CONFIG._NULL_CONFIG.
  AutotuningData: AutotuningData.
  _AUTOTUNING_CACHE_OVERLAY: _AUTOTUNING_CACHE_OVERLAY.
  Residuals.out: Residuals#out.
  Residuals.residuals: Residuals#residuals.
  BoundArguments.serialize: BoundArguments#serialize().
  DeviceKind: DeviceKind.
  _AlwaysEqual: _AlwaysEqual#
  Op.__init_subclass__: Op#__init_subclass__().
  BoundArguments.__post_init__: BoundArguments#__post_init__().
  _AlwaysEqual.__call__: _AlwaysEqual#__call__().
  _AlwaysEqual.__eq__: _AlwaysEqual#__eq__().
  _T2: _T2.
  Op.supports_batched_args_capture: Op#supports_batched_args_capture.
  _FlatTree: _FlatTree#
  Residuals.args: Residuals#args.
  Residuals.kwargs: Residuals#kwargs.
  Op.supports_symbolic_shapes: Op#supports_symbolic_shapes.
  _AlwaysEqual.value: _AlwaysEqual#value.
  _abstractify.abstractify_leaf: _abstractify().abstractify_leaf().
  _FlatTree.values: _FlatTree#values.
  _FlatTree.tree: _FlatTree#tree.
  Residuals.tree_unflatten: Residuals#tree_unflatten().
  Op.__pydantic_config__: Op#__pydantic_config__.
---
# Module: [`tokamax/_src/ops/op.py`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py)

## Classes
### `AUTO`
- def: [`tokamax/_src/ops/op.py:412`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L412)
- signature: `class AUTO:`
- used by: [`autotune`](op.md#BoundArguments.autotune), [`get_config`](op.md#BoundArguments.get_config), [`test_get_config`](op_test.md#BoundArgumentsTest.test_get_config), [`default_config`](op.md#BoundArguments.default_config)

### `BoundArguments`
- def: [`tokamax/_src/ops/op.py:417`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L417) — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
- members:
  - `args(self)` — [`L432`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L432) — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
  - `autotune(self, configs: set[_Config] | type[AUTO] = AUTO, autotuner: autotuner_lib.Autotuner = autotuner_lib.Autotuner(), cache_results: bool = True, event_filter_regex: str | None = None)` — [`L572`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L572) — Autotunes the op with the bound arguments.
  - `autotuning_cache_key(self)` — [`L518`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L518) — Returns a key for autotuning cache lookup.
  - `autotuning_configs(self)` — [`L552`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L552) — Returns the configs used for autotuning when `AUTO` is specified.
  - `benchmark(self, mode: benchmarking.BenchmarkMode = "forward")` — [`L556`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L556) — Benchmarks the op with the bound arguments.
  - `cached_autotuning_data(self)` — [`L523`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L523) — Returns autotuning data from the cache, if available.
  - `default_config(self)` — [`L449`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L449) — Returns the default config for the op.
  - `get_config(self, check_autotuning_cache: bool = True, autotune_configs: set[_Config] | type[AUTO] | None = None, cache_autotuning_results: bool = True, allow_heuristics: bool = True)` — [`L471`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L471) — Returns a config.
  - `heuristics_config(self)` — [`L513`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L513) — Returns a config based on heuristics.
  - `kwargs(self)` — [`L436`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L436)
  - `serialize(value, handler, info)` — [`L622`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L622)
  - `signature(self)` — [`L428`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L428)
  - `validate(value: dict[str, Any])` — [`L627`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L627)
  - `vjp_arg_spec(self)` — [`L599`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L599) — Returns VJP arg specification for this op and arguments.
  - `vmap_axis_sizes(self)` — [`L445`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L445)
  - `arguments` — [`L421`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L421) — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
  - `op` — [`L420`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L420) — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
  - `replace` — [`L641`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L641)
- protocol/private: `__get_pydantic_core_schema__`[`L613`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L613), `__post_init__`[`L423`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L423), `_bound_args`[`L440`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L440)
- uses (calls/refs, reference-scoped): [`Op`](op.md#Op), [`_get_heuristics_config`](op.md#Op._get_heuristics_config), [`_get_autotuning_configs`](op.md#Op._get_autotuning_configs), [`_get_autotuning_cache_key`](op.md#Op._get_autotuning_cache_key), [`get_autotuning_cache`](op.md#Op.get_autotuning_cache), [`standardize_function`](../benchmarking.md#standardize_function), [`config`](op.md#Op.config), [`_Config`](op.md#_Config), [`replace`](op.md#Op.replace), [`compile_benchmark`](../benchmarking.md#compile_benchmark), [`get_vmap_axis_sizes`](../batching.md#get_vmap_axis_sizes), [`infer_device_kind`](op.md#infer_device_kind), [`_get_arg_spec_adapter`](op.md#_get_arg_spec_adapter), [`_fwd_signature`](op.md#Op._fwd_signature), [`AUTO`](op.md#AUTO), [`BenchmarkData`](../benchmarking.md#BenchmarkData), [`_Key`](op.md#_Key), [`get_autotuning_cache_overlay_state`](op.md#get_autotuning_cache_overlay_state), [`_NULL_CONFIG`](op.md#_NULL_CONFIG._NULL_CONFIG), [`AutotuningData`](op.md#AutotuningData), [`BenchmarkMode`](../benchmarking.md#BenchmarkMode.BenchmarkMode)
- used by: [`DotProductAttention`](attention/base.md#DotProductAttention), [`_get_heuristics_config`](op.md#Op._get_heuristics_config), [`bind`](op.md#Op.bind), [`_get_autotuning_configs`](op.md#Op._get_autotuning_configs), [`get_autotuning_configs`](attention/pallas_mosaic_gpu_vjp_kernel_sm100.md#get_autotuning_configs), [`test_roundtrip`](op_test.md#BoundArgumentsTest.test_roundtrip), [`_get_heuristics_config`](ragged_dot/pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_heuristics_config), [`_get_sm90_autotuning_configs`](ragged_dot/pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_sm90_autotuning_configs), [`get_autotuning_configs`](gated_linear_unit/pallas_mosaic_gpu_kernel_sm90.md#get_autotuning_configs), [`fwd`](op.md#Op.fwd), [`_get_heuristics_config`](ragged_dot/pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_heuristics_config), [`_get_autotuning_configs`](attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_autotuning_configs), [`test_heuristics_config`](ragged_dot/pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_heuristics_config), [`get_autotuning_configs`](attention/pallas_mosaic_gpu_kernel_sm100.md#get_autotuning_configs), [`_get_autotuning_configs`](ragged_dot/pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_autotuning_configs), [`_get_heuristics_config`](attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_heuristics_config), [`_get_heuristics_config`](attention/pallas_triton.md#PallasTritonFlashAttention._get_heuristics_config), [`_tgmm_heuristics_config`](ragged_dot/pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._tgmm_heuristics_config), [`autotune`](../autotuning/api.md#autotune), [`get_heuristics_config`](attention/pallas_mosaic_gpu_kernel_sm100.md#get_heuristics_config), [`get_heuristics_config`](attention/pallas_mosaic_gpu_vjp_kernel_sm90.md#get_heuristics_config), [`_get_autotuning_configs`](ragged_dot/pallas_triton.md#PallasTritonRaggedDot._get_autotuning_configs), [`_get_heuristics_config`](flex_attention/pallas_triton.md#PallasTritonFlexAttention._get_heuristics_config), [`_get_autotuning_cache_key`](op.md#Op._get_autotuning_cache_key), [`get_bound_args`](../autotuning/api.md#get_bound_args), [`bind`](attention/base.md#DotProductAttention.bind), [`_get_autotuning_configs`](normalization/pallas_triton_vjp.md#PallasTritonNormalizationVjp._get_autotuning_configs), [`_get_autotuning_configs`](normalization/pallas_triton.md#PallasTritonNormalization._get_autotuning_configs), [`_get_heuristics_config`](gated_linear_unit/pallas_triton.md#PallasTritonGatedLinearUnit._get_heuristics_config), [`bind`](ragged_dot/base.md#RaggedDot.bind), [`test_basic_api`](normalization/api_test.md#LayerNormTest.test_basic_api), [`test_get_config`](op_test.md#BoundArgumentsTest.test_get_config), [`_get_sm100_autotuning_configs`](ragged_dot/pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_sm100_autotuning_configs), [`get_autotuning_configs`](attention/pallas_mosaic_gpu_vjp_kernel_sm90.md#get_autotuning_configs), [`_get_autotuning_configs`](attention/pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._get_autotuning_configs), [`_get_autotuning_configs`](gated_linear_unit/pallas_triton.md#PallasTritonGatedLinearUnit._get_autotuning_configs), [`_get_heuristics_config`](attention/pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._get_heuristics_config), [`_get_heuristics_config`](ragged_dot/pallas_triton.md#PallasTritonRaggedDot._get_heuristics_config), [`test_basic_api`](gated_linear_unit/api_test.md#GatedLinearUnitTest.test_basic_api), [`test_basic_api`](ragged_dot/api_test.md#RaggedDotTest.test_basic_api)  (+65 more)

### `NullConfig`
- def: [`tokamax/_src/ops/op.py:54`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L54)
- signature: `class NullConfig:`
- used by: [`_get_heuristics_config`](op.md#Op._get_heuristics_config), [`_fwd`](attention/jax_nn.md#JaxNnDotProductAttention._fwd), [`_fwd`](flex_attention/wrapper.md#WrappedFlexAttention._fwd), [`_fwd`](attention/xla_chunked.md#XlaChunkedDotProductAttention._fwd), [`XlaChunkedDotProductAttention`](attention/xla_chunked.md#XlaChunkedDotProductAttention), [`JaxNnDotProductAttention`](attention/jax_nn.md#JaxNnDotProductAttention), [`WrappedFlexAttention`](flex_attention/wrapper.md#WrappedFlexAttention), [`config_cls`](op.md#Op.config_cls), [`_NULL_CONFIG`](op.md#_NULL_CONFIG._NULL_CONFIG)

### `Op`  ·  implements/extends ABC, Generic
- def: [`tokamax/_src/ops/op.py:105`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L105) — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
- doc: Base class for operations.
- signature: `class Op(abc.ABC, Generic[_P, _T, _R, _Config, _Key]):`
- members:
  - `_get_autotuning_cache_key(self, ba: BoundArguments)` — [`L351`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L351) — Returns a key for autotuning cache lookup.
  - `_get_autotuning_configs(self, ba: BoundArguments)` — [`L363`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L363) — Returns configs to autotune.
  - `_get_heuristics_config(self, ba: BoundArguments)` — [`L344`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L344) — Returns a config based on heuristics. — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
  - `bind(self, *args: _P.args, return_residuals: bool = False, **kwargs: _P.kwargs)` — [`L295`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L295) — Binds the op to the given arguments. — documented in [tokamax-_src-ops-op](../../../../concepts/tokamax-_src-ops-op.md)
  - `f(*arrays)` — [`L249`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L249)
  - `fwd(*arrays, batched_args, fwd_res=True)` — [`L204`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L204)
  - `fwd_flat(*arrays)` — [`L224`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L224)
  - `get_autotuning_cache(self, device_kind: DeviceKind | None = None)` — [`L312`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L312)
  - `replace(self, **kwargs)` — [`L304`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L304)
  - `signature(self)` — [`L374`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L374) — Infers signature of the op.
  - `supported_on(self, device: jax.Device)` — [`L391`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L391) — Returns whether this op runs on the given device.
  - `config` — [`L150`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L150)
  - `config_cls` — [`L144`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L144)
  - `supports_batched_args_capture` — [`L148`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L148)
  - `supports_symbolic_shapes` — [`L147`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L147)
  - `vjp` — [`L157`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L157)
- protocol/private: `__call__`[`L165`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L165), `__init_subclass__`[`L159`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L159), `__pydantic_config__`[`L389`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L389), `_capture_batched_args`[`L368`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L368), `_fwd`[`L341`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L341), `_fwd_signature`[`L383`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L383)
- uses (calls/refs, reference-scoped): [`BoundArguments`](op.md#BoundArguments), [`DotProductAttention`](attention/base.md#DotProductAttention), [`_get_heuristics_config`](ragged_dot/pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_heuristics_config), [`_fwd`](attention/base.md#DotProductAttention._fwd), [`_fwd`](attention/base.md#DotProductAttentionVjp._fwd), [`_get_heuristics_config`](ragged_dot/pallas_mosaic_gpu.md#PallasMosaicGpuRaggedDot._get_heuristics_config), [`FlexAttention`](flex_attention/base.md#FlexAttention), [`RaggedDot`](ragged_dot/base.md#RaggedDot), [`_get_autotuning_configs`](attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_autotuning_configs), [`_fwd`](ragged_dot/base.md#RaggedDot._fwd), [`args`](op.md#BoundArguments.args), [`arguments`](op.md#BoundArguments.arguments), [`_get_autotuning_configs`](ragged_dot/pallas_mosaic_tpu.md#PallasMosaicTpuRaggedDot._get_autotuning_configs), [`_get_heuristics_config`](attention/pallas_mosaic_tpu.md#PallasMosaicTpuFlashAttention._get_heuristics_config), [`GatedLinearUnit`](gated_linear_unit/base.md#GatedLinearUnit), [`_get_heuristics_config`](attention/pallas_triton.md#PallasTritonFlashAttention._get_heuristics_config), [`DotProductAttentionVjp`](attention/base.md#DotProductAttentionVjp), [`_get_autotuning_configs`](ragged_dot/pallas_triton.md#PallasTritonRaggedDot._get_autotuning_configs), [`_get_heuristics_config`](flex_attention/pallas_triton.md#PallasTritonFlexAttention._get_heuristics_config), [`Normalization`](normalization/base.md#Normalization), [`_FakeOp`](op_test.md#_FakeOp), [`bind`](attention/base.md#DotProductAttention.bind), [`_fwd`](flex_attention/base.md#FlexAttention._fwd), [`_get_autotuning_configs`](normalization/pallas_triton_vjp.md#PallasTritonNormalizationVjp._get_autotuning_configs), [`_get_autotuning_configs`](normalization/pallas_triton.md#PallasTritonNormalization._get_autotuning_configs), [`_get_heuristics_config`](gated_linear_unit/pallas_triton.md#PallasTritonGatedLinearUnit._get_heuristics_config), [`bind`](ragged_dot/base.md#RaggedDot.bind), [`kwargs`](op.md#BoundArguments.kwargs), [`_fwd`](gated_linear_unit/base.md#GatedLinearUnit._fwd), [`_get_autotuning_configs`](attention/pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._get_autotuning_configs), [`_get_autotuning_configs`](gated_linear_unit/pallas_triton.md#PallasTritonGatedLinearUnit._get_autotuning_configs), [`_get_heuristics_config`](attention/pallas_triton_vjp.md#PallasTritonFlashAttentionVjp._get_heuristics_config), [`_get_heuristics_config`](ragged_dot/pallas_triton.md#PallasTritonRaggedDot._get_heuristics_config), [`LinearSoftmaxCrossEntropyLoss`](linear_softmax_cross_entropy_loss/base.md#LinearSoftmaxCrossEntropyLoss), [`_get_autotuning_configs`](attention/pallas_triton.md#PallasTritonFlashAttention._get_autotuning_configs), [`_get_autotuning_configs`](flex_attention/pallas_triton.md#PallasTritonFlexAttention._get_autotuning_configs), [`MultiHeadLatentAttention`](experimental/mla/base.md#MultiHeadLatentAttention), [`_Config`](op.md#_Config), [`_get_autotuning_configs`](attention/pallas_mosaic_tpu_vjp.md#PallasMosaicTpuFlashAttentionVjp._get_autotuning_configs), [`_fwd`](gated_linear_unit/base.md#GatedLinearUnitVjp._fwd)  (+80 more)
- used by: [`DotProductAttention`](attention/base.md#DotProductAttention), [`op`](op.md#BoundArguments.op), [`test_roundtrip`](op_test.md#BoundArgumentsTest.test_roundtrip), [`FlexAttention`](flex_attention/base.md#FlexAttention), [`RaggedDot`](ragged_dot/base.md#RaggedDot), [`autotune`](op.md#BoundArguments.autotune), [`test_tgmm_drhs_with_tile_info_pipes`](ragged_dot/pallas_mosaic_tpu_v2_test.md#PallasMosaicTpuV2OpParameterPipingTest.test_tgmm_drhs_with_tile_info_pipes), [`_API_IMPLEMENTATIONS`](../autotuning/api.md#_API_IMPLEMENTATIONS._API_IMPLEMENTATIONS), [`GatedLinearUnit`](gated_linear_unit/base.md#GatedLinearUnit), [`get_config`](op.md#BoundArguments.get_config), [`autotune`](../autotuning/api.md#autotune), [`DotProductAttentionVjp`](attention/base.md#DotProductAttentionVjp), [`cached_autotuning_data`](op.md#BoundArguments.cached_autotuning_data), [`test_split_k`](ragged_dot/pallas_triton_test.md#PallasTritonRaggedDotTest.test_split_k), [`test_split_k_quantized`](ragged_dot/pallas_triton_test.md#PallasTritonRaggedDotTest.test_split_k_quantized), [`get_bound_args`](../autotuning/api.md#get_bound_args), [`Normalization`](normalization/base.md#Normalization), [`_FakeOp`](op_test.md#_FakeOp), [`bind`](attention/base.md#DotProductAttention.bind), [`bind`](ragged_dot/base.md#RaggedDot.bind), [`test_get_config`](op_test.md#BoundArgumentsTest.test_get_config), [`test_small_block_q`](attention/pallas_triton_test.md#PallasTritonFlashAttentionTest.test_small_block_q), [`fn`](ragged_dot/pallas_mosaic_gpu_test.md#PallasMosaicGpuRaggedDotTest.fn), [`LinearSoftmaxCrossEntropyLoss`](linear_softmax_cross_entropy_loss/base.md#LinearSoftmaxCrossEntropyLoss), [`_test_attention`](attention/pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest._test_attention), [`test_autotune_vjp`](attention/pallas_mosaic_tpu_test.md#PallasMosaicTpuFlashAttentionTest.test_autotune_vjp), [`test_shard_map_manual_axis_type_unreduced`](ragged_dot/pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_unreduced), [`test_shard_map_manual_axis_type_varying`](ragged_dot/pallas_mosaic_tpu_test.md#PallasMosaicTpuRaggedDotTest.test_shard_map_manual_axis_type_varying), [`MultiHeadLatentAttention`](experimental/mla/base.md#MultiHeadLatentAttention), [`_serialize_bound_args_autotuning_data`](../autotuning/api.md#_serialize_bound_args_autotuning_data), [`autotuning_configs`](op.md#BoundArguments.autotuning_configs), [`test_autotune_configs`](attention/pallas_mosaic_gpu_test.md#PallasMosaicGpuFlashAttentionTest.test_autotune_configs), [`test_default_config`](op_test.md#BoundArgumentsTest.test_default_config), [`GatedLinearUnitVjp`](gated_linear_unit/base.md#GatedLinearUnitVjp), [`_validate_bound_args_autotuning_data`](../autotuning/api.md#_validate_bound_args_autotuning_data), [`autotuning_cache_key`](op.md#BoundArguments.autotuning_cache_key), [`heuristics_config`](op.md#BoundArguments.heuristics_config), [`TriangleMultiplication`](triangle_multiplication/base.md#TriangleMultiplication), [`__init__`](attention/pallas_mosaic_gpu_test.md#PallasMosaicGpuFlashAttentionTest.__init__), [`__init__`](attention/pallas_triton_test.md#PallasTritonFlashAttentionWithPallasTritonVjpTest.__init__)  (+43 more; 1 test-only)

### `Residuals`  ·  implements/extends Generic
- def: [`tokamax/_src/ops/op.py:70`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L70)
- doc: Complete arguments and residual values passed to VJP.
- signature: `class Residuals(Generic[_T, _R]):`
- members:
  - `tree_flatten(self)` — [`L78`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L78)
  - `tree_unflatten(cls, aux_data, children)` — [`L93`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L93)
  - `args` — [`L73`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L73)
  - `kwargs` — [`L74`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L74)
  - `out` — [`L75`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L75)
  - `residuals` — [`L76`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L76)
- uses (calls/refs, reference-scoped): [`_T`](op.md#_T), [`_R`](op.md#_R)
- used by: [`fwd`](op.md#Op.fwd)

### `_AlwaysEqual`
- def: [`tokamax/_src/ops/op.py:645`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L645)
- members:
  - `value` — [`L648`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L648)
- protocol/private: `__call__`[`L650`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L650), `__eq__`[`L653`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L653)
- used by: [`fwd`](op.md#Op.fwd)

### `_FlatTree`
- def: [`tokamax/_src/ops/op.py:63`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L63)
- signature: `class _FlatTree:`
- members:
  - `tree` — [`L65`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L65)
  - `values` — [`L64`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L64)
- used by: [`fwd`](op.md#Op.fwd)

## Functions
- `_abstractify(pytree)` — [`L688`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L688)
- `_get_arg_spec_adapter(op: Op)` — [`L658`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L658)
- `abstractify_leaf(x)` — [`L689`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L689)
- `get_autotuning_cache_overlay_state()` — [`L404`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L404)
- `infer_device_kind(ba: BoundArguments)` — [`L678`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L678) — Infers the device kind from bound array arguments.
- `infer_devices(ba: BoundArguments)` — [`L666`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L666) — Infers the devices from bound array arguments.

## Module values
- `AutotuningData` — [`L49`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L49)
- `BOUND_ARGS_ADAPTER` — [`L663`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L663)
- `DeviceKind` — [`L50`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L50)
- `_AUTOTUNING_CACHE` — [`L397`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L397)
- `_AUTOTUNING_CACHE_OVERLAY` — [`L401`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L401)
- `_Config` — [`L47`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L47)
- `_Key` — [`L48`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L48)
- `_NULL_CONFIG` — [`L58`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L58)
- `_P` — [`L43`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L43)
- `_R` — [`L46`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L46)
- `_T` — [`L44`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L44)
- `_T2` — [`L45`](../../../../../../../raw/code/tokamax/tokamax/_src/ops/op.py#L45)

