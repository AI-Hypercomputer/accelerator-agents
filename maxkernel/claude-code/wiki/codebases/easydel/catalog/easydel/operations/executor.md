---
title: 'Module: easydel/operations/executor.py'
type: catalog
provenance: extracted
module: easydel/operations/executor.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.operations.executor`/
symbols:
  OperationExecutor.get_combined_requirements: OperationExecutor#get_combined_requirements().
  OperationExecutor.get_operation: OperationExecutor#get_operation().
  OperationExecutor.get_requirements: OperationExecutor#get_requirements().
  OperationExecutor.decode_operation: OperationExecutor#decode_operation().
  OperationExecutor.prefill_impl: OperationExecutor#prefill_impl.
  OperationExecutor.prefill_operation: OperationExecutor#prefill_operation().
  OperationExecutor.get_operation_name: OperationExecutor#get_operation_name().
  OperationExecutor.decode_impl: OperationExecutor#decode_impl.
  OperationExecutor.requires_cache: OperationExecutor#requires_cache().
  OperationExecutor.has_separate_decode: OperationExecutor#has_separate_decode().
  OperationExecutor.is_valid: OperationExecutor#is_valid().
  OperationExecutor.mixin_impl: OperationExecutor#mixin_impl.
  OperationExecutor: OperationExecutor#
  OperationExecutor.from_operations: OperationExecutor#from_operations().
  OperationExecutor.from_flexible_attention: OperationExecutor#from_flexible_attention().
  __all__: __all__.
---
# Module: [`easydel/operations/executor.py`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py)

## Classes
### `OperationExecutor`
- def: [`easydel/operations/executor.py:31`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L31)
- doc: Mode-bound operation executor for dynamic discovery.
- signature: `class OperationExecutor:`
- members:
  - `decode_operation(self)` — [`L71`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L71) — Get the operation for decode mode.
  - `from_flexible_attention(cls, flex_attn)` — [`L191`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L191) — Create from a FlexibleAttentionModule instance.
  - `from_operations(cls, prefill: BaseOperation | None = None, decode: BaseOperation | None = None, mixin: BaseOperation | None = None)` — [`L207`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L207) — Create from individual operation instances.
  - `get_combined_requirements(self)` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L115) — Get combined requirements from both prefill and decode operations.
  - `get_operation(self, mode: ExecutionMode)` — [`L79`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L79) — Get operation for a specific execution mode.
  - `get_operation_name(self, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L173`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L173) — Get the name of the operation for a specific mode.
  - `get_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED)` — [`L95`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L95) — Get requirements for the specified mode.
  - `has_separate_decode(self)` — [`L162`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L162) — Whether decode uses a different operation than prefill.
  - `is_valid(self)` — [`L169`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L169) — Whether at least one operation is available.
  - `prefill_operation(self)` — [`L64`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L64) — Get the operation for prefill mode.
  - `requires_cache(self)` — [`L156`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L156) — Whether any operation requires cache.
  - `decode_impl` — [`L60`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L60)
  - `mixin_impl` — [`L61`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L61)
  - `prefill_impl` — [`L59`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L59)
- uses (calls/refs, reference-scoped): [`ExecutionMode`](requirements/types.md#ExecutionMode), [`OperationRequirements`](requirements/requirements.md#OperationRequirements), [`get_requirements`](_base_operation.md#BaseOperation.get_requirements), [`cache`](requirements/requirements.md#OperationRequirements.cache), [`get_impl_name`](_base_operation.md#BaseOperation.get_impl_name), [`MIXED`](requirements/types.md#ExecutionMode.MIXED), [`default`](requirements/requirements.md#OperationRequirements.default), [`BaseOperation`](_base_operation.md#BaseOperation), [`requires_cache`](requirements/requirements.md#CacheRequirements.requires_cache), [`PREFILL`](requirements/types.md#ExecutionMode.PREFILL), [`DECODE`](requirements/types.md#ExecutionMode.DECODE)
- used by: [`get_operations_cache_info_dynamic`](../infra/mixins/operation_cache.md#OperationCacheMixin.get_operations_cache_info_dynamic), [`operation_executor`](../layers/attention/_flexible.md#FlexibleAttentionModule.operation_executor), [`operation_requirements`](../layers/attention/_flexible.md#FlexibleAttentionModule.operation_requirements), [`requires_cache`](../layers/attention/_flexible.md#FlexibleAttentionModule.requires_cache), [`has_separate_decode`](../layers/attention/_flexible.md#FlexibleAttentionModule.has_separate_decode)

## Module values
- `__all__` — [`L27`](../../../../../../raw/code/EasyDeL/easydel/operations/executor.py#L27)

