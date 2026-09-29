---
title: 'Module: src/maxtext/layers/linears.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/linears.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.linears`/
symbols:
  DenseGeneral.__call__: DenseGeneral#__call__().
  MlpBlock.__call__: MlpBlock#__call__().
  MlpBlock.wi: MlpBlock#wi.
  MlpBlock.wo: MlpBlock#wo.
  DenseGeneral.kernel: DenseGeneral#kernel.
  DenseGeneral._quant_dot_general_name: DenseGeneral#_quant_dot_general_name.
  DenseGeneral.parameter_memory_host_offload: DenseGeneral#parameter_memory_host_offload.
  MlpBlock.mlp_layer_norm: MlpBlock#mlp_layer_norm.
  MlpBlock.config: MlpBlock#config.
  DeepSeekV4GroupedLinear.__call__: DeepSeekV4GroupedLinear#__call__().
  DeepSeekV4GroupedLinear.parameter_memory_host_offload: DeepSeekV4GroupedLinear#parameter_memory_host_offload.
  DeepSeekV4GroupedLinear.kernel: DeepSeekV4GroupedLinear#kernel.
  DenseGeneral.bias: DenseGeneral#bias.
  DenseGeneral.in_features_shape: DenseGeneral#in_features_shape.
  DenseGeneral.out_features_shape: DenseGeneral#out_features_shape.
  DenseGeneral.axis: DenseGeneral#axis.
  dense_general: dense_general().
  MlpBlock.activations: MlpBlock#activations.
  MlpBlock.dropout: MlpBlock#dropout.
  MlpBlock.get_norm_layer: MlpBlock#get_norm_layer().
  DenseGeneral.dtype: DenseGeneral#dtype.
  MlpBlock.intermediate_logical: MlpBlock#intermediate_logical.
  canonicalize_tuple: canonicalize_tuple().
  DenseGeneral: DenseGeneral#
  MlpBlock.dtype: MlpBlock#dtype.
  MlpBlock.use_bias: MlpBlock#use_bias.
  DenseGeneral.quant_dot_general: DenseGeneral#quant_dot_general().
  MlpBlock.model_mode: MlpBlock#model_mode.
  MlpBlock.intermediate_dim: MlpBlock#intermediate_dim.
  MlpBlock.kernel_init: MlpBlock#kernel_init.
  MlpBlock.weight_dtype: MlpBlock#weight_dtype.
  MlpBlock.quant: MlpBlock#quant.
  mlp_block: mlp_block().
  deepseek_v4_grouped_linear: deepseek_v4_grouped_linear().
  _convert_to_activation_function: _convert_to_activation_function().
  normalize_axes: normalize_axes().
  DenseGeneral.weight_dtype: DenseGeneral#weight_dtype.
  DenseGeneral.kernel_axes: DenseGeneral#kernel_axes.
  DenseGeneral.quant: DenseGeneral#quant.
  DeepSeekV4GroupedLinear.dtype: DeepSeekV4GroupedLinear#dtype.
  _compute_dot_general_nnx: _compute_dot_general_nnx().
  DenseGeneral.kernel_init: DenseGeneral#kernel_init.
  DenseGeneral.use_bias: DenseGeneral#use_bias.
  DenseGeneral.shard_mode: DenseGeneral#shard_mode.
  DenseGeneral.matmul_precision: DenseGeneral#matmul_precision.
  Dropout: Dropout#
  MlpBlock: MlpBlock#
  MlpBlock.intermediate_dropout_rate: MlpBlock#intermediate_dropout_rate.
  MlpBlock.use_pre_norm: MlpBlock#use_pre_norm.
  MlpBlock._maybe_shard_with_logical: MlpBlock#_maybe_shard_with_logical.
  DeepSeekV4GroupedLinear: DeepSeekV4GroupedLinear#
  DeepSeekV4GroupedLinear.in_features_per_group: DeepSeekV4GroupedLinear#in_features_per_group.
  DeepSeekV4GroupedLinear.n_groups: DeepSeekV4GroupedLinear#n_groups.
  DeepSeekV4GroupedLinear.out_features_per_group: DeepSeekV4GroupedLinear#out_features_per_group.
  DeepSeekV4GroupedLinear.weight_dtype: DeepSeekV4GroupedLinear#weight_dtype.
  DeepSeekV4GroupedLinear.kernel_init: DeepSeekV4GroupedLinear#kernel_init.
  DeepSeekV4GroupedLinear.kernel_axes: DeepSeekV4GroupedLinear#kernel_axes.
  DeepSeekV4GroupedLinear.matmul_precision: DeepSeekV4GroupedLinear#matmul_precision.
  _compute_dot_general: _compute_dot_general().
  DenseGeneral.__init__: DenseGeneral#__init__().
  Dropout.__init__: Dropout#__init__().
  Dropout.rate: Dropout#rate.
  Dropout.broadcast_dims: Dropout#broadcast_dims.
  Dropout.deterministic: Dropout#deterministic.
  Dropout.rng_collection: Dropout#rng_collection.
  Dropout.rngs: Dropout#rngs.
  MlpBlock.__init__: MlpBlock#__init__().
  MlpBlock.mesh: MlpBlock#mesh.
  MlpBlock.in_features: MlpBlock#in_features.
  DeepSeekV4GroupedLinear.__init__: DeepSeekV4GroupedLinear#__init__().
  DeepSeekV4GroupedLinear.out_features: DeepSeekV4GroupedLinear#out_features.
---
# Module: [`src/maxtext/layers/linears.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py)

## Classes
### `DeepSeekV4GroupedLinear`  ·  implements/extends Module
- def: [`src/maxtext/layers/linears.py:575`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L575)
- doc: Block-diagonal grouped linear used by the grouped output projection in DeepSeek-V4.
- signature: `class DeepSeekV4GroupedLinear(nnx.Module):`
- members:
  - `__call__(self, inputs: Array)` — [`L647`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L647) — Applies a batched grouped linear transformation to the inputs.
  - `__init__(self, in_features_per_group: int, out_features: int, n_groups: int, weight_dtype: DType = jnp.float32, dtype: DType = jnp.float32, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), kernel_axes: tuple[None | str, ...] = ("groups", "embed", "mlp"), matmul_precision: str = "default", parameter_memory_host_offload: bool = False, *, rngs: nnx.Rngs)` — [`L584`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L584) — Initializes the DeepSeekV4GroupedLinear module.
  - `dtype` — [`L621`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L621)
  - `in_features_per_group` — [`L615`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L615)
  - `kernel` — [`L636`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L636)
  - `kernel_axes` — [`L623`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L623)
  - `kernel_init` — [`L622`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L622)
  - `matmul_precision` — [`L624`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L624)
  - `n_groups` — [`L617`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L617)
  - `out_features` — [`L616`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L616)
  - `out_features_per_group` — [`L618`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L618)
  - `parameter_memory_host_offload` — [`L625`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L625)
  - `weight_dtype` — [`L620`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L620)
- used by: [`deepseek_v4_grouped_linear`](linears.md#deepseek_v4_grouped_linear)

### `DenseGeneral`  ·  implements/extends Module
- def: [`src/maxtext/layers/linears.py:107`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L107) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- doc: A linear transformation with flexible axes.
- signature: `class DenseGeneral(nnx.Module):`
- members:
  - `__call__(self, inputs: Array, _initializing: bool = False, out_sharding: NamedSharding | None = None)` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L203) — Applies a linear transformation to the inputs along multiple dimensions. — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `__init__(self, in_features_shape: Iterable[int] | int, out_features_shape: Iterable[int] | int, axis: Iterable[int] | int = -1, weight_dtype: DType = jnp.float32, dtype: DType = jnp.float32, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), kernel_axes: tuple[None | str, ...] = (), quant: None | Quant = None, use_bias: bool = False, shard_mode: ShardMode = ShardMode.AUTO, matmul_precision: str = "default", parameter_memory_host_offload: bool = False, *, rngs: nnx.Rngs = None)` — [`L110`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L110) — Initializes the DenseGeneral module.
  - `quant_dot_general(self)` — [`L198`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L198) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `axis` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L147) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `bias` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L178) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `dtype` — [`L149`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L149) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `in_features_shape` — [`L145`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L145) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `kernel` — [`L164`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L164) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `kernel_axes` — [`L151`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L151) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `kernel_init` — [`L150`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L150)
  - `matmul_precision` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L155) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `out_features_shape` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L146)
  - `parameter_memory_host_offload` — [`L156`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L156) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `quant` — [`L152`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L152) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `shard_mode` — [`L154`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L154) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `use_bias` — [`L153`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L153) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `weight_dtype` — [`L148`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L148) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- protocol/private: `_quant_dot_general_name`[`L189`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L189)
- uses (calls/refs, reference-scoped): [`canonicalize_tuple`](linears.md#canonicalize_tuple), [`normalize_axes`](linears.md#normalize_axes), [`_compute_dot_general_nnx`](linears.md#_compute_dot_general_nnx)
- used by: [`wi`](linears.md#MlpBlock.wi), [`wo`](linears.md#MlpBlock.wo), [`dense_general`](linears.md#dense_general)

### `Dropout`  ·  implements/extends Dropout
- def: [`src/maxtext/layers/linears.py:320`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L320) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- doc: Forked nnx.Dropout that is easier to use with bridge
- signature: `class Dropout(nnx.Dropout):`
- members:
  - `broadcast_dims` — [`L333`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L333)
  - `deterministic` — [`L334`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L334)
  - `rate` — [`L332`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L332)
  - `rng_collection` — [`L335`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L335)
  - `rngs` — [`L338`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L338)
- protocol/private: `__init__`[`L323`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L323)
- used by: [`dropout`](linears.md#MlpBlock.dropout)

### `MlpBlock`  ·  implements/extends Module
- def: [`src/maxtext/layers/linears.py:343`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L343)
- doc: Transformer MLP / feed-forward block.
- signature: `class MlpBlock(nnx.Module):`
- members:
  - `__call__(self, inputs, decode: bool = False, deterministic: bool = False, intermediate_sharding: NamedSharding | None = None, out_sharding: NamedSharding | None = None)` — [`L490`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L490) — Applies Transformer MlpBlock module. — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `__init__(self, config: Config, mesh: Mesh, in_features: int, intermediate_dim: int = 2048, activations: Sequence[str | Callable[..., Any]] = ("relu",), kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), intermediate_dropout_rate: float = 0.1, dtype: Any = jnp.float32, weight_dtype: Any = jnp.float32, use_bias: bool = False, use_pre_norm: bool = False, quant: None | Quant = None, model_mode: None | str = None, *, rngs: nnx.Rngs)` — [`L346`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L346) — A MlpBlock module.
  - `get_norm_layer(self, num_features: int)` — [`L466`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L466) — get normalization layer. — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `activations` — [`L387`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L387) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `config` — [`L383`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L383) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `dropout` — [`L444`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L444) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `dtype` — [`L390`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L390)
  - `in_features` — [`L385`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L385)
  - `intermediate_dim` — [`L386`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L386) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `intermediate_dropout_rate` — [`L389`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L389) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `intermediate_logical` — [`L409`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L409) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `kernel_init` — [`L388`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L388)
  - `mesh` — [`L384`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L384)
  - `mlp_layer_norm` — [`L398`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L398) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `model_mode` — [`L395`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L395) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `quant` — [`L394`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L394)
  - `use_bias` — [`L392`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L392)
  - `use_pre_norm` — [`L393`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L393) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `weight_dtype` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L391)
  - `wi` — [`L414`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L414) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
  - `wo` — [`L445`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L445) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- protocol/private: `_maybe_shard_with_logical`[`L459`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L459)
- uses (calls/refs, reference-scoped): [`DenseGeneral`](linears.md#DenseGeneral), [`_convert_to_activation_function`](linears.md#_convert_to_activation_function), [`Dropout`](linears.md#Dropout)
- used by: [`mlp_block`](linears.md#mlp_block)

## Functions
- `_compute_dot_general(inputs, kernel, kernel_axes, axis, contract_ind, matmul_precision, quant)` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L73) — Computes a dot_general operation that may be quantized.
- `_compute_dot_general_nnx(inputs, kernel, axis, contract_ind, matmul_precision, quant_dot_general: nnx_wrappers.ToNNX | None, initializing: bool, out_sharding: NamedSharding | None = None)` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L84) — Computes a dot_general operation that may be quantized. — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- `_convert_to_activation_function(fn_or_string: str | Callable[..., Any])` — [`L43`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L43) — Convert a string to an activation function. — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- `canonicalize_tuple(x)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L66) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)
- `deepseek_v4_grouped_linear(*, in_features_per_group: int, out_features: int, n_groups: int, weight_dtype: DType = jnp.float32, dtype: DType = jnp.float32, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), kernel_axes: tuple[None | str, ...] = ("groups", "embed", "mlp"), matmul_precision: str = "default", parameter_memory_host_offload: bool = False, name: None | str = None)` — [`L683`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L683) — Creates a DeepSeekV4GroupedLinear Linen module using nnx.bridge.to_linen.
- `dense_general(*, inputs_shape: tuple[int, ...] | None = None, in_features_shape: tuple[int, ...] | int | None = None, out_features_shape: Iterable[int] | int, axis: Iterable[int] | int = -1, weight_dtype: DType = jnp.float32, dtype: DType = jnp.float32, kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), kernel_axes: tuple[None | str, ...] = (), quant: None | Quant = None, use_bias: bool = False, shard_mode: ShardMode = ShardMode.AUTO, matmul_precision: str = "default", parameter_memory_host_offload: bool = False, name: None | str = None)` — [`L255`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L255) — Creates a DenseGeneral Linen module using nnx.bridge.to_linen.
- `mlp_block(*, config: Config, mesh: Mesh, in_features: int, intermediate_dim: int = 2048, activations: Sequence[str | Callable[..., Any]] = ("relu",), kernel_init: NdInitializer = nd_dense_init(1, "fan_in", "truncated_normal"), intermediate_dropout_rate: float = 0.1, dtype: Any = jnp.float32, weight_dtype: Any = jnp.float32, use_bias: bool = False, use_pre_norm: bool = False, quant: None | Quant = None, model_mode: None | str = None, name: None | str = None)` — [`L535`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L535) — Creates a MlpBlock Linen module using nnx.bridge.to_linen.
- `normalize_axes(axes: Iterable[int], ndim: int)` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/layers/linears.py#L61) — documented in [maxtext-layers-linears](../../../../concepts/maxtext-layers-linears.md)

