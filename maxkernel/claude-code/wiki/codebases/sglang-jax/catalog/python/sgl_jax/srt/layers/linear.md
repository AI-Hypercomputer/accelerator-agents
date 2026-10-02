---
title: 'Module: python/sgl_jax/srt/layers/linear.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/linear.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.linear`/
symbols:
  QuantizedLinear.__call__: QuantizedLinear#__call__().
  QuantizedLinear.from_linear: QuantizedLinear#from_linear().
  LinearBase: LinearBase#
  LinearBase.bias: LinearBase#bias.
  LinearBase.__call__: LinearBase#__call__().
  MergedColumnParallelLinear.output_sizes: MergedColumnParallelLinear#output_sizes.
  LinearBase.weight: LinearBase#weight.
  LinearBase.mesh: LinearBase#mesh.
  QuantizedLinear: QuantizedLinear#
  QuantizedLinear.bias: QuantizedLinear#bias.
  MergedColumnParallelLinear: MergedColumnParallelLinear#
  QuantizedLinear.__init__: QuantizedLinear#__init__().
  LinearBase.skip_bias_add: LinearBase#skip_bias_add.
  LinearBase.params_dtype: LinearBase#params_dtype.
  LinearBase.kernel_axes: LinearBase#kernel_axes.
  LinearBase.name: LinearBase#name.
  QuantizedLinear.weight_q: QuantizedLinear#weight_q.
  QuantizedLinear.activation_dtype: QuantizedLinear#activation_dtype.
  QuantizedLinear.mesh: QuantizedLinear#mesh.
  QuantizedLinear.kernel_axes: QuantizedLinear#kernel_axes.
  logger: logger.
  _shard_map_output_partition_dim: _shard_map_output_partition_dim().
  LinearBase.__init__: LinearBase#__init__().
  MergedColumnParallelLinear._mesh_tp_size: MergedColumnParallelLinear#_mesh_tp_size().
  QuantizedLinear.weight_scale: QuantizedLinear#weight_scale.
  QuantizedLinear.skip_bias_add: QuantizedLinear#skip_bias_add.
  QuantizedLinear.compute_dtype: QuantizedLinear#compute_dtype.
  QuantizedLinear.weight_block_size: QuantizedLinear#weight_block_size.
  QuantizedLinear.allow_narrow_n_blockwise: QuantizedLinear#allow_narrow_n_blockwise.
  MergedColumnParallelLinear.__init__: MergedColumnParallelLinear#__init__().
  QuantizedLinear.params_dtype: QuantizedLinear#params_dtype.
  QuantizedLinear.name: QuantizedLinear#name.
---
# Module: [`python/sgl_jax/srt/layers/linear.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py)

## Classes
### `LinearBase`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/linear.py:34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L34)
- doc: Base linear layer.
- signature: `class LinearBase(nnx.Module):`
- members:
  - `__call__(self, x: jax.Array, *, out_sharding: jax.sharding.Sharding | None = None)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L87) — Forward pass. If ``out_sharding`` is None, falls back to the
  - `__init__(self, input_size: int, output_size: int, mesh: jax.sharding.Mesh, use_bias: bool = True, skip_bias_add: bool = False, params_dtype: jnp.dtype | None = jnp.bfloat16, kernel_axes: Sequence[str | None] | None = None, scope_name: str = "linear_base")` — [`L48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L48) — Initialize parameters and quantization method.
  - `bias` — [`L75`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L75)
  - `kernel_axes` — [`L62`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L62)
  - `mesh` — [`L63`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L63)
  - `name` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L64)
  - `params_dtype` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L61)
  - `skip_bias_add` — [`L60`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L60)
  - `weight` — [`L66`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L66)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`MergedColumnParallelLinear`](linear.md#MergedColumnParallelLinear)
- used by: [`__call__`](attention/linear/kda_backend.md#KDAAttnBackend.__call__), [`__call__`](attention/linear/gdn_backend.md#GDNAttnBackend.__call__), [`from_linear`](linear.md#QuantizedLinear.from_linear), [`apply_linear_quantization`](../utils/quantization/quantization_utils.md#apply_linear_quantization), [`_replace_linear_recursive`](../utils/quantization/quantization_utils.md#apply_linear_quantization._replace_linear_recursive), [`create_bf16_linear`](../utils/weight_utils.md#WeightLoader.create_bf16_linear), [`__init__`](../lora/layers.md#LoRAEmbedding.__init__), [`__init__`](../lora/layers.md#LoRALinear.__init__), [`_replace_with_lora`](../lora/lora_manager.md#LoRAManager._replace_with_lora), [`output_sizes`](linear.md#MergedColumnParallelLinear.output_sizes), [`dequant_fp8_linear`](../utils/weight_utils.md#WeightLoader.dequant_fp8_linear), [`MergedColumnParallelLinear`](linear.md#MergedColumnParallelLinear), [`__init__`](radix_linear_attention.md#RadixLinearAttention.__init__)

### `MergedColumnParallelLinear`  ·  implements/extends LinearBase
- def: [`python/sgl_jax/srt/layers/linear.py:116`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L116)
- doc: Column-parallel linear with multiple logical outputs merged into one weight.
- signature: `class MergedColumnParallelLinear(LinearBase):`
- members:
  - `_mesh_tp_size(mesh: jax.sharding.Mesh | None)` — [`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L159) — TP size = mesh size on the ``"tensor"`` axis (1 if absent).
  - `output_sizes` — [`L178`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L178)
- protocol/private: `__init__`[`L168`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L168)
- uses (calls/refs, reference-scoped): [`LinearBase`](linear.md#LinearBase), [`__init__`](linear.md#LinearBase.__init__)
- used by: [`LinearBase`](linear.md#LinearBase)

### `QuantizedLinear`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/linear.py:198`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L198)
- doc: Quantized linear layer using native quantized matmul.
- signature: `class QuantizedLinear(nnx.Module):`
- members:
  - `__call__(self, x: jax.Array, *, out_sharding: jax.sharding.Sharding | None = None)` — [`L429`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L429) — Forward pass using quantized matmul. If ``out_sharding`` is None,
  - `__init__(self, weight_q: jax.Array, weight_scale: jax.Array, bias: jax.Array | None, activation_dtype: jnp.dtype | None, mesh: jax.sharding.Mesh, kernel_axes: Sequence[str | None] | None = None, skip_bias_add: bool = False, params_dtype: jnp.dtype | None = jnp.bfloat16, compute_dtype: jnp.dtype | None = None, weight_block_size: tuple[int, int] | None = None, allow_narrow_n_blockwise: bool = False, scope_name: str = "quantized_linear")` — [`L227`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L227) — Initialize the quantized linear layer with pre-quantized weights.
  - `from_linear(cls, linear: LinearBase, weight_dtype: jnp.dtype, activation_dtype: jnp.dtype | None = None, is_static_input: bool = False, weight_block_size: Sequence[int] | None = None, allow_narrow_n_blockwise: bool = False)` — [`L270`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L270) — Convert a LinearBase layer to a QuantizedLinear layer.
  - `activation_dtype` — [`L259`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L259)
  - `allow_narrow_n_blockwise` — [`L266`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L266)
  - `bias` — [`L258`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L258)
  - `compute_dtype` — [`L264`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L264)
  - `kernel_axes` — [`L261`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L261)
  - `mesh` — [`L260`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L260)
  - `name` — [`L267`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L267)
  - `params_dtype` — [`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L263)
  - `skip_bias_add` — [`L262`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L262)
  - `weight_block_size` — [`L265`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L265)
  - `weight_q` — [`L256`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L256)
  - `weight_scale` — [`L257`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L257)
- uses (calls/refs, reference-scoped): [`named_scope`](../utils/profiling_utils.md#named_scope), [`quantize_tensor`](../utils/quantization/quantization_utils.md#quantize_tensor), [`LinearBase`](linear.md#LinearBase), [`bias`](linear.md#LinearBase.bias), [`xla_quantized_matmul_local`](../kernels/quantized_matmul/kernel.md#xla_quantized_matmul_local), [`weight`](linear.md#LinearBase.weight), [`expand_block_scale`](../kernels/quantized_matmul/blockwise_utils.md#expand_block_scale), [`mesh`](linear.md#LinearBase.mesh), [`kernel_axes`](linear.md#LinearBase.kernel_axes), [`name`](linear.md#LinearBase.name), [`params_dtype`](linear.md#LinearBase.params_dtype), [`skip_bias_add`](linear.md#LinearBase.skip_bias_add), [`_shard_map_output_partition_dim`](linear.md#_shard_map_output_partition_dim), [`logger`](linear.md#logger)
- used by: [`apply_linear_quantization`](../utils/quantization/quantization_utils.md#apply_linear_quantization), [`_replace_linear_recursive`](../utils/quantization/quantization_utils.md#apply_linear_quantization._replace_linear_recursive), [`dequant_fp8_layers`](../utils/weight_utils.md#WeightLoader.dequant_fp8_layers)

## Functions
- `_shard_map_output_partition_dim(sharding: jax.sharding.Sharding, axis_name: str | None)` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L21)

## Module values
- `logger` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/linear.py#L18)

