---
title: 'Module: recurrentgemma/jax/scan.py'
type: catalog
provenance: extracted
module: recurrentgemma/jax/scan.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.jax.scan`/
symbols:
  linear_scan: linear_scan().
  single_shard_rnn_scan: single_shard_rnn_scan().
  resolve_scan_type: resolve_scan_type().
  ShardingSpec: ShardingSpec.
  lru_linear_scan: lru_linear_scan().
  lru_associative_scan: lru_associative_scan().
  lru_associative_scan.lru_associative_bin_op: lru_associative_scan().lru_associative_bin_op().
  lru_pallas_scan: lru_pallas_scan.
  lru_linear_scan.body_fn: lru_linear_scan().body_fn().
---
# Module: [`recurrentgemma/jax/scan.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py)

## Functions
- `body_fn(carry, current_inputs)` — [`L86`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L86)
- `linear_scan(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None = None, reverse: bool = False, scan_type: common.ScanType = common.ScanType.AUTO, acc_float_dtype: jnp.dtype = jnp.float32, sharding_spec: pallas.ShardingSpec | None = None, unroll: int = 1, pallas_max_seq_block_size: int = 256, pallas_min_seq_block_size: int = 16, pallas_pad_seq_to_min_block_size: bool = True, pallas_pad_last_dim_to_128: bool = False)` — [`L352`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L352) — Runs the recurrence of a linear RNN on a single (sequence) shard. — documented in [recurrentgemma-jax-layers](../../../concepts/recurrentgemma-jax-layers.md)
- `lru_associative_bin_op(element_i: tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex], element_j: tuple[complex_lib.RealOrComplex, complex_lib.RealOrComplex])` — [`L161`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L161)
- `lru_associative_scan(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None = None, reverse: bool = False, acc_float_dtype: jnp.dtype = jnp.float32, return_a_prod: Literal[False] = False)` — [`L124`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L124)
- `lru_linear_scan(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None = None, reverse: bool = False, return_a_prod: Literal[False] = False, acc_float_dtype: jnp.dtype = jnp.float32, unroll: int = 1)` — [`L46`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L46)
- `resolve_scan_type(scan_type: common.ScanType)` — [`L33`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L33) — Resolves the scan type if its AUTO.
- `single_shard_rnn_scan(x: complex_lib.RealOrComplex, a: complex_lib.RealOrComplex, h0: complex_lib.RealOrComplex | None = None, reverse: bool = False, scan_type: common.ScanType = common.ScanType.AUTO, acc_float_dtype: jnp.dtype = jnp.float32, seq_axis: str | None = None, seq_axis_index_groups: list[list[int]] | None = None, unroll: int = 1, pallas_max_seq_block_size: int = 256, pallas_min_seq_block_size: int = 16, pallas_pad_seq_to_min_block_size: bool = True, pallas_pad_last_dim_to_128: bool = False)` — [`L207`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L207) — Runs the recurrence of a linear RNN on a single (sequence) shard. — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)

## Module values
- `ShardingSpec` — [`L29`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L29)
- `lru_pallas_scan` — [`L30`](../../../../../../raw/code/recurrentgemma/recurrentgemma/jax/scan.py#L30) — documented in [recurrentgemma-jax-pallas](../../../concepts/recurrentgemma-jax-pallas.md)

