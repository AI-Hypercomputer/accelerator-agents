---
title: 'Module: easydel/axis.py'
type: catalog
provenance: extracted
module: easydel/axis.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.axis`/
symbols:
  ATTN_DP: ATTN_DP.
  resolve_attention_data_parallel_axis: resolve_attention_data_parallel_axis().
  register_attention_data_parallel_axis: register_attention_data_parallel_axis().
  reset_attention_data_parallel_axis: reset_attention_data_parallel_axis().
  _DEFAULT_ATTN_DP_RULE: _DEFAULT_ATTN_DP_RULE.
  _normalize_axis_rule: _normalize_axis_rule().
  __all__: __all__.
---
# Module: [`easydel/axis.py`](../../../../../raw/code/EasyDeL/easydel/axis.py)

## Functions
- `_normalize_axis_rule(axis_rule: tp.Any)` — [`L28`](../../../../../raw/code/EasyDeL/easydel/axis.py#L28)
- `register_attention_data_parallel_axis(axis_rule: tp.Any = _DEFAULT_ATTN_DP_RULE, *, generation_axis_rule: tp.Any = NOT_GIVEN)` — [`L37`](../../../../../raw/code/EasyDeL/easydel/axis.py#L37) — Register the semantic axis used for attention/KV-cache data parallelism.
- `reset_attention_data_parallel_axis()` — [`L58`](../../../../../raw/code/EasyDeL/easydel/axis.py#L58) — Reset ``ATTN_DP`` to follow ``PartitionAxis.data_parallel_axis``.
- `resolve_attention_data_parallel_axis(partition_axis_or_manager: PartitionAxis | PartitionManager, *, mode: str = MODE_PREFILL)` — [`L67`](../../../../../raw/code/EasyDeL/easydel/axis.py#L67) — Resolve the configured attention/KV-cache data-parallel axis rule.

## Module values
- `ATTN_DP` — [`L24`](../../../../../raw/code/EasyDeL/easydel/axis.py#L24)
- `_DEFAULT_ATTN_DP_RULE` — [`L25`](../../../../../raw/code/EasyDeL/easydel/axis.py#L25)
- `__all__` — [`L93`](../../../../../raw/code/EasyDeL/easydel/axis.py#L93)

