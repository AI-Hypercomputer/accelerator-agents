---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/bref_override.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/bref_override.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.bref_override`/
symbols:
  _BypassRef: _BypassRef#
  _BypassRef.__post_init__: _BypassRef#__post_init__().
  KVBufferedRef: KVBufferedRef#
  KVBufferedRef.cfgs: KVBufferedRef#cfgs.
  KVBufferedRef.create: KVBufferedRef#create().
  KVBufferedRef.copy_in: KVBufferedRef#copy_in().
  KVBufferedRef.copy_out: KVBufferedRef#copy_out().
  KVBufferedRef.wait_in: KVBufferedRef#wait_in().
  KVBufferedRef.wait_out: KVBufferedRef#wait_out().
  BatchingORef: BatchingORef#
  BatchingORef.cfgs: BatchingORef#cfgs.
  BatchingORef.create: BatchingORef#create().
  BatchingORef.copy_out: BatchingORef#copy_out().
  BatchingORef.wait_out: BatchingORef#wait_out().
  BatchingQRef: BatchingQRef#
  BatchingQRef.cfgs: BatchingQRef#cfgs.
  BatchingQRef.create: BatchingQRef#create().
  BatchingQRef.copy_in: BatchingQRef#copy_in().
  BatchingQRef.wait_in: BatchingQRef#wait_in().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/bref_override.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py)

## Classes
### `BatchingORef`
- def: [`tpu_inference/kernels/experimental/batched_rpa/bref_override.py:209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L209)
- doc: Handles normalizing and storing the final attention output.
- signature: `class BatchingORef(pltpu.BufferedRef):`
- members:
  - `copy_out(self, dst_ref: tuple[jax.Ref, schedule.RpaSchedule], grid_indices: tuple[int | jax.Array, ...])` — [`L244`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L244)
  - `create(cls, spec: pl.BlockSpec, dtype_or_type: jax.Array, buffer_type, buffer_count: int, use_lookahead: bool, cfgs: configs.RpaConfigs)` — [`L216`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L216)
  - `wait_out(self, dst_ref: tuple[jax.Ref, schedule.RpaSchedule], grid_indices: tuple[int | jax.Array, ...])` — [`L272`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L272)
  - `cfgs` — [`L212`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L212)

### `BatchingQRef`
- def: [`tpu_inference/kernels/experimental/batched_rpa/bref_override.py:300`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L300)
- doc: Handles fetching Q blocks using precomputed metadata.
- signature: `class BatchingQRef(pltpu.BufferedRef):`
- members:
  - `copy_in(self, src_ref: tuple[jax.Ref, schedule.RpaSchedule], grid_indices: tuple[int | jax.Array, ...])` — [`L335`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L335)
  - `create(cls, spec: pl.BlockSpec, dtype_or_type: jax.Array, buffer_type, buffer_count: int, use_lookahead: bool, cfgs: configs.RpaConfigs)` — [`L307`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L307)
  - `wait_in(self, src_ref: tuple[jax.Ref, schedule.RpaSchedule], grid_indices: tuple[int | jax.Array, ...])` — [`L360`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L360)
  - `cfgs` — [`L303`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L303)

### `KVBufferedRef`
- def: [`tpu_inference/kernels/experimental/batched_rpa/bref_override.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L38)
- doc: Handles fetching and updating KV cache using precomputed metadata.
- signature: `class KVBufferedRef(_BypassRef):`
- members:
  - `copy_in(self, src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref], grid_indices: tuple[int | jax.Array, ...])` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L73)
  - `copy_out(self, dst_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref], grid_indices: tuple[int | jax.Array, ...])` — [`L122`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L122)
  - `create(cls, spec: pl.BlockSpec, dtype_or_type: jax.Array, buffer_type, buffer_count: int, use_lookahead: bool, cfgs: configs.RpaConfigs)` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L45)
  - `wait_in(self, src_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref], grid_indices: tuple[int | jax.Array, ...])` — [`L150`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L150)
  - `wait_out(self, dst_ref: tuple[jax.Ref, jax.Ref, schedule.RpaSchedule, jax.Ref], grid_indices: tuple[int | jax.Array, ...])` — [`L180`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L180)
  - `cfgs` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L41)

### `_BypassRef`
- def: [`tpu_inference/kernels/experimental/batched_rpa/bref_override.py:27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L27)
- doc: Helper class to safely bypass buffer_count checks during creation.
- signature: `class _BypassRef(pltpu.BufferedRef):`
- protocol/private: `__post_init__`[`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/bref_override.py#L30)

