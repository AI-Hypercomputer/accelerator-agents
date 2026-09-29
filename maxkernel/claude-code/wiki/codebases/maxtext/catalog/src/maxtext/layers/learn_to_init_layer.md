---
title: 'Module: src/maxtext/layers/learn_to_init_layer.py'
type: catalog
provenance: extracted
module: src/maxtext/layers/learn_to_init_layer.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.layers.learn_to_init_layer`/
symbols:
  LearnToInitDense.__call__: LearnToInitDense#__call__().
  LearnToInitDense.B: LearnToInitDense#B.
  apply_lti_model_update: apply_lti_model_update().
  LearnToInitDense.C: LearnToInitDense#C.
  LearnToInitDense: LearnToInitDense#
  LearnToInitDense.bias: LearnToInitDense#bias.
  LearnToInitDense.weight_dtype: LearnToInitDense#weight_dtype.
  _customize_attention_modules: _customize_attention_modules().
  apply_lti_modification: apply_lti_modification().
  LearnToInitDense.W: LearnToInitDense#W.
  LearnToInitDense.A: LearnToInitDense#A.
  calculate_attn_weight: calculate_attn_weight().
  LearnToInitDense.TENSOR_A: LearnToInitDense#TENSOR_A.
  LearnToInitDense.TENSOR_B: LearnToInitDense#TENSOR_B.
  LearnToInitDense.TENSOR_W: LearnToInitDense#TENSOR_W.
  LearnToInitDense.in_features_shape: LearnToInitDense#in_features_shape.
  LearnToInitDense.out_features_shape: LearnToInitDense#out_features_shape.
  LearnToInitDense.is_output_projection: LearnToInitDense#is_output_projection.
  LearnToInitDense.dtype: LearnToInitDense#dtype.
  LearnToInitDense.matmul_precision: LearnToInitDense#matmul_precision.
  LearnToInitDense.use_general_linear_map: LearnToInitDense#use_general_linear_map.
  _calc_attn_weight: _calc_attn_weight().
  LTI_MODIFIED_ATTENTION_PARAM_NAMES: LTI_MODIFIED_ATTENTION_PARAM_NAMES.
  LearnToInitDense.TENSOR_C: LearnToInitDense#TENSOR_C.
  LearnToInitDense.axis: LearnToInitDense#axis.
  LearnToInitDense.kernel_axes: LearnToInitDense#kernel_axes.
  LTI_ORIGINAL_ATTENTION_PARAMS_NAME: LTI_ORIGINAL_ATTENTION_PARAMS_NAME.
  LTI_LAYER_PATH_PREFIXES: LTI_LAYER_PATH_PREFIXES.
  LearnToInitDense.use_bias: LearnToInitDense#use_bias.
  LearnToInitDense.shard_mode: LearnToInitDense#shard_mode.
  LearnToInitDense.parameter_memory_host_offload: LearnToInitDense#parameter_memory_host_offload.
  LearnToInitDense.__init__: LearnToInitDense#__init__().
  LearnToInitDense.kernel_init: LearnToInitDense#kernel_init.
  LearnToInitDense.quant: LearnToInitDense#quant.
---
# Module: [`src/maxtext/layers/learn_to_init_layer.py`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py)

## Classes
### `LearnToInitDense`
- def: [`src/maxtext/layers/learn_to_init_layer.py:124`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L124)
- members:
  - `A` — [`L201`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L201)
  - `B` — [`L205`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L205)
  - `C` — [`L184`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L184)
  - `TENSOR_A` — [`L144`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L144)
  - `TENSOR_B` — [`L145`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L145)
  - `TENSOR_C` — [`L146`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L146)
  - `TENSOR_W` — [`L147`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L147)
  - `W` — [`L196`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L196)
  - `axis` — [`L171`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L171)
  - `bias` — [`L233`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L233)
  - `dtype` — [`L174`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L174)
  - `in_features_shape` — [`L169`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L169)
  - `is_output_projection` — [`L173`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L173)
  - `kernel_axes` — [`L176`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L176)
  - `kernel_init` — [`L175`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L175)
  - `matmul_precision` — [`L180`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L180)
  - `out_features_shape` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L170)
  - `parameter_memory_host_offload` — [`L181`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L181)
  - `quant` — [`L177`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L177)
  - `shard_mode` — [`L179`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L179)
  - `use_bias` — [`L178`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L178)
  - `use_general_linear_map` — [`L182`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L182)
  - `weight_dtype` — [`L172`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L172)
- protocol/private: `__call__`[`L240`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L240), `__init__`[`L149`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L149)
- uses (calls/refs, reference-scoped): [`_calc_attn_weight`](learn_to_init_layer.md#_calc_attn_weight)
- used by: [`apply_lti_model_update`](learn_to_init_layer.md#apply_lti_model_update), [`_customize_attention_modules`](learn_to_init_layer.md#_customize_attention_modules)

## Functions
- `_calc_attn_weight(A: jax.Array | nnx.Param | None, B: jax.Array | nnx.Param | None, C: jax.Array | nnx.Param | None, general_map: Optional[jax.Array | nnx.Param] = None, is_output_projection: bool = False, matmul_precision: str = "default", scan_dim: str = "")` — [`L298`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L298) — Computes the effective attention weights from teacher weight and learnable projection(s).
- `_customize_attention_modules(config: Config, attn_module_name: str, module: nnx.Module)` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L77) — Replaces specific DenseGeneral modules (q, k, v projections) in the attention module.
- `apply_lti_model_update(student_model, student_config)` — [`L369`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L369) — Applies the finalized learn-to-init weights to the student model and cleans up the NNX graph.
- `apply_lti_modification(module: nnx.Module, module_name: str | None = None)` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L37) — Applies Learn-To-Init structural modifications to an instantiated NNX module.
- `calculate_attn_weight(A: jax.Array | None, B: jax.Array | None, C: jax.Array, general_map: Optional[jax.Array] = None, is_output_projection: bool = False, matmul_precision: str = "default")` — [`L326`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L326) — Helper function to dynamically compute the effective attention weights using `jnp.einsum`.

## Module values
- `LTI_LAYER_PATH_PREFIXES` — [`L34`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L34)
- `LTI_MODIFIED_ATTENTION_PARAM_NAMES` — [`L32`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L32)
- `LTI_ORIGINAL_ATTENTION_PARAMS_NAME` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/layers/learn_to_init_layer.py#L33)

