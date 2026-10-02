---
title: 'Module: easydel/layers/quantization/_quants.py'
type: catalog
provenance: extracted
module: easydel/layers/quantization/_quants.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.layers.quantization._quants`/
symbols:
  EasyQuantizer: EasyQuantizer#
  quantize: quantize().
  EasyQuantizer.apply_quantization: EasyQuantizer#apply_quantization().
  EasyQuantizer.pattern: EasyQuantizer#pattern().
  EasyQuantizer.config: EasyQuantizer#config().
  EasyQuantizer._config: EasyQuantizer#_config.
  EasyQuantizer.dequantize_modules: EasyQuantizer#dequantize_modules().
  EasyQuantizer.__call__: EasyQuantizer#__call__().
  EasyQuantizer.quantize_array: EasyQuantizer#quantize_array().
  _extract_explicit_qmm_kwargs: _extract_explicit_qmm_kwargs().
  EasyQuantizer.__str__: EasyQuantizer#__str__().
  EasyQuantizer.__init__: EasyQuantizer#__init__().
  _extract_model_qmm_defaults: _extract_model_qmm_defaults().
  _filter_supported_to_quantized_kwargs: _filter_supported_to_quantized_kwargs().
  _QMM_KWARG_ALIASES._QMM_KWARG_ALIASES: _QMM_KWARG_ALIASES._QMM_KWARG_ALIASES.
  EasyQuantizer.__repr__: EasyQuantizer#__repr__.
---
# Module: [`easydel/layers/quantization/_quants.py`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py)

## Classes
### `EasyQuantizer`
- def: [`easydel/layers/quantization/_quants.py:280`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L280)
- doc: High-level quantization interface for EasyDeL models.
- signature: `class EasyQuantizer:`
- members:
  - `__call__(self, array: jax.Array, path: str | tuple[str] | None = None)` — [`L366`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L366) — Quantize an array with optional path-based filtering.
  - `__init__(self, quantization_config: QuantizationConfig | None = None)` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L326) — Initialize the EasyQuantizer with a configuration.
  - `__str__(self)` — [`L592`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L592) — Return a string representation of the quantizer.
  - `apply_quantization(self, model: nn.Module,, *, quantization_pattern: str | None = None, **kwargs)` — [`L460`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L460) — Quantize compatible modules in a model to lower precision.
  - `config(self)` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L337) — Get the quantization configuration.
  - `dequantize_modules(self, model: nn.Module)` — [`L549`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L549) — Restore quantized modules to their full-precision equivalents.
  - `pattern(self)` — [`L347`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L347) — Get the regex pattern for layer selection.
  - `quantize_array(self, array: jax.Array, simulate: bool = False)` — [`L422`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L422) — Quantize a single array using the configured quantization method.
- protocol/private: `__repr__`[`L601`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L601), `_config`[`L334`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L334)
- uses (calls/refs, reference-scoped): [`QuantizationConfig`](_configs.md#QuantizationConfig), [`iter_module_search`](../../utils/traversals.md#iter_module_search), [`quantize`](_quants.md#quantize), [`set_module_from_path`](../../utils/traversals.md#set_module_from_path), [`DEFAULT_QUANTIZATION_PATTERN`](_configs.md#DEFAULT_QUANTIZATION_PATTERN), [`_extract_explicit_qmm_kwargs`](_quants.md#_extract_explicit_qmm_kwargs), [`pattern`](_configs.md#QuantizationConfig.pattern), [`_extract_model_qmm_defaults`](_quants.md#_extract_model_qmm_defaults), [`_filter_supported_to_quantized_kwargs`](_quants.md#_filter_supported_to_quantized_kwargs)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`_load_model_weights`](../../infra/mixins/bridge.md#EasyBridgeMixin._load_model_weights), [`concatenate_to_cache`](../../caching/transformer/cache.md#TransformerCacheView.concatenate_to_cache), [`init`](../../caching/transformer/cache.md#TransformerCacheView.init), [`init`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCacheView.init), [`init`](../../caching/unified_attention/cache.md#UnifiedAttentionCacheView.init), [`insert`](../../caching/transformer/cache.md#TransformerCache.insert), [`_lazy_init_rebuilt_module`](../../infra/base_module.md#EasyDeLBaseModule._lazy_init_rebuilt_module), [`init_cache`](../../caching/ragged_page/cache.md#RaggedPagesCache.init_cache), [`init`](../../caching/ragged_page/cache.md#RaggedPagesCacheView.init), [`quantize`](../../infra/base_module.md#EasyDeLBaseModule.quantize), [`init_cache`](../../caching/unified_attention/cache.md#UnifiedAttentionCache.init_cache), [`init_cache`](../../caching/mla_ragged_page/cache.md#MLARaggedPagesCache.init_cache), [`init_cache`](../../caching/transformer/cache.md#TransformerCache.init_cache), [`init_cache`](../../caching/turboquant_ragged_page/cache.md#TurboQuantRaggedPagesCache.init_cache), [`init`](../../caching/mla_ragged_page/cache.md#MLARaggedPagesCacheView.init), [`quantizer`](../attention/_flexible.md#AttentionModule.quantizer), [`_quant_class`](../../infra/mixins/generation.md#EasyGenerationMixin._quant_class), [`quantize_linear_layers`](../../infra/utils.md#quantize_linear_layers), [`concatenate_to_cache`](../../caching/lightning/cache.md#LightningCacheView.concatenate_to_cache)

## Functions
- `_extract_explicit_qmm_kwargs(kwargs: dict[str, typing.Any])` — [`L97`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L97)
- `_extract_model_qmm_defaults(model: nn.Module)` — [`L106`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L106)
- `_filter_supported_to_quantized_kwargs(to_quantized: typing.Callable[..., typing.Any], overrides: dict[str, typing.Any])` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L134)
- `quantize(array: jax.Array, config: QuantizationConfig | None = None, dtype: QuantizationType | str | None = None, group_size: int | None = None, simulate: bool = False)` — [`L156`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L156) — Quantize an array using the specified quantization format.

## Module values
- `_QMM_KWARG_ALIASES` — [`L81`](../../../../../../../raw/code/EasyDeL/easydel/layers/quantization/_quants.py#L81)

