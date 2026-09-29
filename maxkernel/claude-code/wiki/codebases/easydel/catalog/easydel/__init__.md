---
title: 'Module: easydel/__init__.py'
type: catalog
provenance: extracted
module: easydel/__init__.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 easydel/_
symbols:
  _import_structure: import_structure.
  _distributed_msg: distributed_msg.
  __version__: _version__.
  _distributed_init_enabled: distributed_init_enabled.
  _logger: logger.
  _patch_eformer_exception_serialization._patched_ser_exc_info: patch_eformer_exception_serialization()._patched_ser_exc_info().
  _patch_transformers_import_utils: patch_transformers_import_utils().
  _patch_transformers_rope_scaling_property: patch_transformers_rope_scaling_property().
  _patch_transformers_init_weights_tie_signature: patch_transformers_init_weights_tie_signature().
  _patch_eformer_exception_serialization: patch_eformer_exception_serialization().
  _patch_transformers_autoconfig_gated_repo_skip: patch_transformers_autoconfig_gated_repo_skip().
  _patch_transformers_rope_scaling_property._patched_get: patch_transformers_rope_scaling_property()._patched_get().
  _patch_transformers_init_weights_tie_signature._patched_init_weights: patch_transformers_init_weights_tie_signature()._patched_init_weights().
  _patch_transformers_autoconfig_gated_repo_skip._patched_from_pretrained: patch_transformers_autoconfig_gated_repo_skip()._patched_from_pretrained().
  _ensure_optional_deepspeed_stub: ensure_optional_deepspeed_stub().
  _patch_transformers_import_utils._is_torch_fx_available: patch_transformers_import_utils()._is_torch_fx_available().
  _patch_eformer_exception_serialization._coerce_picklable_exception: patch_eformer_exception_serialization()._coerce_picklable_exception().
---
# Module: [`easydel/__init__.py`](../../../../../raw/code/EasyDeL/easydel/__init__.py)

## Functions
- `_coerce_picklable_exception(exception: BaseException | None)` — [`L205`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L205)
- `_ensure_optional_deepspeed_stub()` — [`L60`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L60) — Provide a minimal deepspeed module for remote-code import checks.
- `_is_torch_fx_available()` — [`L96`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L96)
- `_patch_eformer_exception_serialization()` — [`L182`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L182) — Replace non-picklable remote exceptions with a safe fallback.
- `_patch_transformers_autoconfig_gated_repo_skip()` — [`L239`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L239) — Convert gated-repo config load failures to ``pytest.skip``.
- `_patch_transformers_import_utils()` — [`L87`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L87) — Backfill removed transformers import-utils symbols for remote model code.
- `_patch_transformers_init_weights_tie_signature()` — [`L150`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L150) — Handle legacy remote-model ``tie_weights()`` signature changes.
- `_patch_transformers_rope_scaling_property()` — [`L109`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L109) — Normalize HF ``rope_scaling`` property for legacy DeepSeek remote modules.
- `_patched_from_pretrained(cls, *args, **kwargs)` — [`L259`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L259)
- `_patched_get(self)` — [`L130`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L130)
- `_patched_init_weights(self)` — [`L167`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L167)
- `_patched_ser_exc_info(cls, exception: BaseException | None = None)` — [`L227`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L227)

## Module values
- `__version__` — [`L31`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L31)
- `_distributed_init_enabled` — [`L1455`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L1455)
- `_distributed_msg` — [`L1473`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L1473)
- `_import_structure` — [`L373`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L373)
- `_logger` — [`L57`](../../../../../raw/code/EasyDeL/easydel/__init__.py#L57)

