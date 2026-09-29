---
title: 'Module: jax/_src/lax/ann.py'
type: catalog
provenance: extracted
module: jax/_src/lax/ann.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.lax.ann`/
symbols:
  approx_top_k_p: approx_top_k_p.
  _approx_top_k_jvp: _approx_top_k_jvp().
  _approx_top_k_lowering: _approx_top_k_lowering().
  approx_max_k: approx_max_k().
  approx_min_k: approx_min_k().
  _approx_top_k_abstract_eval: _approx_top_k_abstract_eval().
  _approx_top_k_batch_rule: _approx_top_k_batch_rule().
  _get_init_val_literal: _get_init_val_literal().
  _comparator_builder_mlir: _comparator_builder_mlir().
---
# Module: [`jax/_src/lax/ann.py`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py)

## Functions
- `_approx_top_k_abstract_eval(operand, *, k, reduction_dimension, recall_target, is_max_k, reduction_input_size_override, aggregate_to_topk)` — [`L214`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L214)
- `_approx_top_k_batch_rule(batch_operands, batch_axes, *, k, reduction_dimension, recall_target, is_max_k, reduction_input_size_override, aggregate_to_topk)` — [`L345`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L345)
- `_approx_top_k_jvp(primals, tangents, *, k, reduction_dimension, recall_target, is_max_k, reduction_input_size_override, aggregate_to_topk)` — [`L372`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L372)
- `_approx_top_k_lowering(ctx, operand, *, k, reduction_dimension, recall_target, is_max_k, reduction_input_size_override, aggregate_to_topk, fallback=False)` — [`L279`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L279)
- `_comparator_builder_mlir(ctx, op_type, is_max_k)` — [`L257`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L257)
- `_get_init_val_literal(op_type, is_max_k)` — [`L254`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L254)
- `approx_max_k(operand: Array, k: int, reduction_dimension: int = -1, recall_target: float = 0.95, reduction_input_size_override: int = -1, aggregate_to_topk: bool = True)` — [`L92`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L92) — Returns max ``k`` values and their indices of the ``operand`` in an approximate manner.
- `approx_min_k(operand: Array, k: int, reduction_dimension: int = -1, recall_target: float = 0.95, reduction_input_size_override: int = -1, aggregate_to_topk: bool = True)` — [`L151`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L151) — Returns min ``k`` values and their indices of the ``operand`` in an approximate manner.

## Module values
- `approx_top_k_p` — [`L398`](../../../../../../../raw/code/jax/jax/_src/lax/ann.py#L398)

