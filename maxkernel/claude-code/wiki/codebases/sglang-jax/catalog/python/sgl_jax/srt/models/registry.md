---
title: 'Module: python/sgl_jax/srt/models/registry.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/models/registry.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.models.registry`/
symbols:
  ModelRegistry: ModelRegistry.
  _ModelRegistry.resolve_model_cls: _ModelRegistry#resolve_model_cls().
  _ModelRegistry._normalize_archs: _ModelRegistry#_normalize_archs().
  _ModelRegistry.get_supported_archs: _ModelRegistry#get_supported_archs().
  _ModelRegistry.models: _ModelRegistry#models.
  _ModelRegistry._raise_for_unsupported: _ModelRegistry#_raise_for_unsupported().
  _ModelRegistry._try_load_model_cls: _ModelRegistry#_try_load_model_cls().
  import_model_classes: import_model_classes().
  logger: logger.
  _ModelRegistry: _ModelRegistry#
---
# Module: [`python/sgl_jax/srt/models/registry.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py)

## Classes
### `_ModelRegistry`
- def: [`python/sgl_jax/srt/models/registry.py:13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L13)
- signature: `class _ModelRegistry:`
- members:
  - `get_supported_archs(self)` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L17)
  - `resolve_model_cls(self, architectures: str | list[str])` — [`L57`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L57)
  - `models` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L15)
- protocol/private: `_normalize_archs`[`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L40), `_raise_for_unsupported`[`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L20), `_try_load_model_cls`[`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L34)
- uses (calls/refs, reference-scoped): [`logger`](registry.md#logger)
- used by: [`get_model_architecture`](../model_loader/arch.md#get_model_architecture), [`_apply_model_specific_config`](../configs/model_config.md#ModelConfig._apply_model_specific_config), [`ModelRegistry`](registry.md#ModelRegistry)

## Functions
- `import_model_classes()` — [`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L72)

## Module values
- `ModelRegistry` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L100)
- `logger` — [`L9`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/models/registry.py#L9)

