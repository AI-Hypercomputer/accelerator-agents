---
title: 'Module: recurrentgemma/common.py'
type: catalog
provenance: extracted
module: recurrentgemma/common.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.common`/
symbols:
  TemporalBlockType: TemporalBlockType#
  ScanType: ScanType#
  TemporalBlockType.RECURRENT: TemporalBlockType#RECURRENT.
  Preset.config_dict: Preset#config_dict().
  TemporalBlockType.ATTENTION: TemporalBlockType#ATTENTION.
  GriffinConfig.from_flax_params_or_variables: GriffinConfig#from_flax_params_or_variables().
  GriffinConfig.vocab_size: GriffinConfig#vocab_size.
  GriffinConfig.width: GriffinConfig#width.
  GriffinConfig.block_types: GriffinConfig#block_types.
  GriffinConfig.attention_window_size: GriffinConfig#attention_window_size.
  GriffinConfig.from_torch_params: GriffinConfig#from_torch_params().
  GriffinConfig: GriffinConfig#
  GriffinConfig.num_heads: GriffinConfig#num_heads.
  GriffinConfig.lru_width: GriffinConfig#lru_width.
  GriffinConfig._from_parameter_kwargs: GriffinConfig#_from_parameter_kwargs().
  GriffinConfig.mlp_expanded_width: GriffinConfig#mlp_expanded_width.
  GriffinConfig.embeddings_scale_by_sqrt_dim: GriffinConfig#embeddings_scale_by_sqrt_dim.
  GriffinConfig.logits_soft_cap: GriffinConfig#logits_soft_cap.
  ScanType.AUTO: ScanType#AUTO.
  Preset: Preset#
  GriffinConfig.from_preset: GriffinConfig#from_preset().
  ScanType.LINEAR_NATIVE: ScanType#LINEAR_NATIVE.
  GriffinConfig.scan_type: GriffinConfig#scan_type.
  Preset.RECURRENT_GEMMA_2B_V1: Preset#RECURRENT_GEMMA_2B_V1.
  GriffinConfig.num_layers: GriffinConfig#num_layers().
  ScanType.LINEAR_PALLAS: ScanType#LINEAR_PALLAS.
  GriffinConfig.max_cache_length: GriffinConfig#max_cache_length().
  ScanType.ASSOCIATIVE_NATIVE: ScanType#ASSOCIATIVE_NATIVE.
  apply_it_formatter: apply_it_formatter().
  Preset.GRIFFIN_PAPER_7B: Preset#GRIFFIN_PAPER_7B.
  Preset.HAWK_PAPER_7B: Preset#HAWK_PAPER_7B.
  Preset.RECURRENT_GEMMA_9B_V1: Preset#RECURRENT_GEMMA_9B_V1.
---
# Module: [`recurrentgemma/common.py`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py)

## Classes
### `GriffinConfig`  ·  implements/extends NamedTuple
- def: [`recurrentgemma/common.py:117`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L117) — documented in [recurrentgemma-jax-griffin](../../concepts/recurrentgemma-jax-griffin.md)
- doc: Griffin config - https://arxiv.org/abs/2402.19427.
- signature: `class GriffinConfig(NamedTuple):`
- members:
  - `_from_parameter_kwargs(cls, kwargs: dict[str, int | tuple[TemporalBlockType, ...]], preset: Preset | None = None, embeddings_scale_by_sqrt_dim: bool | None = None, attention_window_size: int | None = None, logits_soft_cap: float | None = None, scan_type: ScanType = ScanType.AUTO, max_sequence_length: int | None = None)` — [`L180`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L180) — Creates a `GriffinConfig` from kwargs inferred from parameters. — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `from_flax_params_or_variables(cls, flax_params_or_variables: Mapping[str, Any], preset: Preset | None = None, embeddings_scale_by_sqrt_dim: bool | None = None, attention_window_size: int | None = None, logits_soft_cap: float | None = None, scan_type: ScanType = ScanType.AUTO, max_sequence_length: int | None = None)` — [`L224`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L224) — Creates a `GriffinConfig` from Flax parameters. — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `from_preset(cls, preset: Preset, vocab_size: int = 256000, max_sequence_length: int | None = None)` — [`L165`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L165) — Creates a `GriffinConfig` for a given preset. — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `from_torch_params(cls, params: dict[str, Any], preset: Preset | None = None, embeddings_scale_by_sqrt_dim: bool | None = None, attention_window_size: int | None = None, logits_soft_cap: float | None = None, scan_type: ScanType | None = None, max_sequence_length: int | None = None)` — [`L323`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L323) — Creates a `GriffinConfig` from Pytorch parameters. — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `max_cache_length(self)` — [`L155`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L155) — The maximum length of the cache used for the model.
  - `num_layers(self)` — [`L160`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L160) — The number of layers of the model.
  - `attention_window_size` — [`L149`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L149)
  - `block_types` — [`L147`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L147) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `embeddings_scale_by_sqrt_dim` — [`L148`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L148)
  - `logits_soft_cap` — [`L150`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L150)
  - `lru_width` — [`L151`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L151)
  - `mlp_expanded_width` — [`L145`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L145)
  - `num_heads` — [`L146`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L146)
  - `scan_type` — [`L152`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L152) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `vocab_size` — [`L143`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L143)
  - `width` — [`L144`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L144)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](common.md#TemporalBlockType), [`ScanType`](common.md#ScanType), [`RECURRENT`](common.md#TemporalBlockType.RECURRENT), [`config_dict`](common.md#Preset.config_dict), [`ATTENTION`](common.md#TemporalBlockType.ATTENTION), [`AUTO`](common.md#ScanType.AUTO), [`Preset`](common.md#Preset)
- used by: [`blocks`](jax/griffin.md#Griffin.blocks), [`test_conversion_back_and_forth`](conversion_test.md#ConversionTest.test_conversion_back_and_forth), [`test_forward_equivalence`](jax/sampler_test.md#SamplerTest.test_forward_equivalence), [`test_griffin_output_shape`](jax/griffin_test.md#GriffinTest.test_griffin_output_shape), [`test_output_shapes`](jax/sampler_test.md#SamplerTest.test_output_shapes), [`test_forward_equivalence`](torch/sampler_test.md#SamplerTest.test_forward_equivalence), [`test_numerically_to_jax`](torch/griffin_test.md#GriffinTest.test_numerically_to_jax), [`test_output_shapes`](torch/sampler_test.md#SamplerTest.test_output_shapes), [`test_samples`](jax/sampler_test.md#SamplerTest.test_samples), [`test_samples`](torch/sampler_test.md#SamplerTest.test_samples), [`Griffin`](torch/griffin.md#Griffin), [`Griffin`](jax/griffin.md#Griffin), [`embedder`](jax/griffin.md#Griffin.embedder), [`test_numerically_to_jax`](torch/modules_test.md#EmbedderTest.test_numerically_to_jax), [`init_cache`](jax/griffin.md#Griffin.init_cache), [`blocks`](torch/griffin.md#Griffin.blocks), [`config`](jax/griffin.md#Griffin.config), [`init_cache`](torch/griffin.md#Griffin.init_cache), [`final_norm`](jax/griffin.md#Griffin.final_norm), [`embedder`](torch/griffin.md#Griffin.embedder), [`final_norm`](torch/griffin.md#Griffin.final_norm), [`vocab_size`](torch/sampler.md#Sampler.vocab_size), [`GriffinConfig`](jax/__init__.md#GriffinConfig), [`GriffinConfig`](torch/__init__.md#GriffinConfig), [`__init__`](torch/griffin.md#Griffin.__init__)  (4 test-only)

### `Preset`  ·  implements/extends Enum
- def: [`recurrentgemma/common.py:46`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L46) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
- doc: All default preset variants.
- signature: `class Preset(enum.Enum):`
- members:
  - `config_dict(self)` — [`L55`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L55) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `GRIFFIN_PAPER_7B` — [`L49`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L49) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `HAWK_PAPER_7B` — [`L50`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L50) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `RECURRENT_GEMMA_2B_V1` — [`L51`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L51) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `RECURRENT_GEMMA_9B_V1` — [`L52`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L52) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
- uses (calls/refs, reference-scoped): [`TemporalBlockType`](common.md#TemporalBlockType), [`ScanType`](common.md#ScanType), [`RECURRENT`](common.md#TemporalBlockType.RECURRENT), [`ATTENTION`](common.md#TemporalBlockType.ATTENTION), [`AUTO`](common.md#ScanType.AUTO)
- used by: [`from_flax_params_or_variables`](common.md#GriffinConfig.from_flax_params_or_variables), [`from_torch_params`](common.md#GriffinConfig.from_torch_params), [`_from_parameter_kwargs`](common.md#GriffinConfig._from_parameter_kwargs), [`from_preset`](common.md#GriffinConfig.from_preset), [`Preset`](jax/__init__.md#Preset), [`Preset`](torch/__init__.md#Preset)  (4 test-only)

### `ScanType`  ·  implements/extends Enum
- def: [`recurrentgemma/common.py:31`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L31) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
- doc: Which Jax implementation to use for the scan in the RG-LRU in Jax.
- signature: `class ScanType(enum.Enum):`
- members:
  - `ASSOCIATIVE_NATIVE` — [`L41`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L41)
  - `AUTO` — [`L39`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L39) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `LINEAR_NATIVE` — [`L40`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L40)
  - `LINEAR_PALLAS` — [`L42`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L42)
- used by: [`test_numerically_to_jax`](torch/griffin_test.md#GriffinTest.test_numerically_to_jax), [`test_numerically_to_jax`](torch/modules_test.md#ResidualBlockTest.test_numerically_to_jax), [`linear_scan`](jax/scan.md#linear_scan), [`single_shard_rnn_scan`](jax/scan.md#single_shard_rnn_scan), [`config_dict`](common.md#Preset.config_dict), [`test_numerically_to_jax`](torch/modules_test.md#RecurrentBlockTest.test_numerically_to_jax), [`test_numerically_to_jax`](torch/layers_test.md#RGLRUTest.test_numerically_to_jax), [`from_flax_params_or_variables`](common.md#GriffinConfig.from_flax_params_or_variables), [`from_torch_params`](common.md#GriffinConfig.from_torch_params), [`_from_parameter_kwargs`](common.md#GriffinConfig._from_parameter_kwargs), [`resolve_scan_type`](jax/scan.md#resolve_scan_type), [`test_scan`](jax/layers_test.md#LayersTest.test_scan), [`scan_type`](jax/layers.md#RGLRU.scan_type), [`use_custom_complex`](jax/layers.md#RGLRU.use_custom_complex), [`scan_type`](jax/modules.md#RecurrentBlock.scan_type), [`scan_type`](jax/modules.md#ResidualBlock.scan_type), [`scan_type`](common.md#GriffinConfig.scan_type), [`ScanType`](jax/__init__.md#ScanType), [`ScanType`](torch/__init__.md#ScanType)

### `TemporalBlockType`  ·  implements/extends Enum
- def: [`recurrentgemma/common.py:23`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L23) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
- doc: Type of temporal mixing to use in a residual block.
- signature: `class TemporalBlockType(enum.Enum):`
- members:
  - `ATTENTION` — [`L26`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L26) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
  - `RECURRENT` — [`L27`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L27) — documented in [recurrentgemma-common](../../concepts/recurrentgemma-common.md)
- used by: [`recurrent_block`](jax/modules.md#ResidualBlock.recurrent_block), [`test_forward_equivalence`](jax/sampler_test.md#SamplerTest.test_forward_equivalence), [`test_griffin_output_shape`](jax/griffin_test.md#GriffinTest.test_griffin_output_shape), [`test_output_shapes`](jax/sampler_test.md#SamplerTest.test_output_shapes), [`test_forward_equivalence`](torch/sampler_test.md#SamplerTest.test_forward_equivalence), [`test_numerically_to_jax`](torch/griffin_test.md#GriffinTest.test_numerically_to_jax), [`test_output_shapes`](torch/sampler_test.md#SamplerTest.test_output_shapes), [`test_samples`](jax/sampler_test.md#SamplerTest.test_samples), [`test_samples`](torch/sampler_test.md#SamplerTest.test_samples), [`test_numerically_to_jax`](torch/modules_test.md#ResidualBlockTest.test_numerically_to_jax), [`config_dict`](common.md#Preset.config_dict), [`temporal_pre_norm`](jax/modules.md#ResidualBlock.temporal_pre_norm), [`init_cache`](jax/modules.md#ResidualBlock.init_cache), [`test_recurrent_block_output_shapes`](jax/modules_test.md#RecurrentBlockTest.test_recurrent_block_output_shapes), [`from_flax_params_or_variables`](common.md#GriffinConfig.from_flax_params_or_variables), [`init_cache`](torch/modules.md#ResidualBlock.init_cache), [`recurrent_block`](torch/modules.md#ResidualBlock.recurrent_block), [`block_types`](common.md#GriffinConfig.block_types), [`from_torch_params`](common.md#GriffinConfig.from_torch_params), [`temporal_block`](torch/modules.md#ResidualBlock.temporal_block), [`temporal_block`](jax/modules.md#ResidualBlock.temporal_block), [`_from_parameter_kwargs`](common.md#GriffinConfig._from_parameter_kwargs), [`temporal_pre_norm`](torch/modules.md#ResidualBlock.temporal_pre_norm), [`BLOCK_TYPES`](conversion_test.md#BLOCK_TYPES), [`temporal_block_type`](jax/modules.md#ResidualBlock.temporal_block_type), [`TemporalBlockType`](jax/__init__.md#TemporalBlockType), [`TemporalBlockType`](torch/__init__.md#TemporalBlockType), [`__init__`](torch/modules.md#ResidualBlock.__init__)  (2 test-only)

## Functions
- `apply_it_formatter(input_string: str)` — [`L412`](../../../../../raw/code/recurrentgemma/recurrentgemma/common.py#L412) — documented in [recurrentgemma-jax-sampler](../../concepts/recurrentgemma-jax-sampler.md)

