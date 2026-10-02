---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/schedule.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/schedule.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.schedule`/
symbols:
  generate_rpa_metadata: generate_rpa_metadata().
  SmemWrapper: SmemWrapper#
  SmemWrapper.data: SmemWrapper#data.
  SmemWrapper.shape: SmemWrapper#shape.
  SmemWrapper.create_shape_dtype: SmemWrapper#create_shape_dtype().
  SmemWrapper._get_pos: SmemWrapper#_get_pos().
  SmemWrapper.__getitem__: SmemWrapper#__getitem__().
  SmemWrapper.__setitem__: SmemWrapper#__setitem__().
  RpaSchedule: RpaSchedule#
  RpaSchedule.s_idx: RpaSchedule#s_idx.
  RpaSchedule.q_idx: RpaSchedule#q_idx.
  RpaSchedule.k_idx: RpaSchedule#k_idx.
  RpaSchedule.is_last_k: RpaSchedule#is_last_k.
  RpaSchedule.do_writeback: RpaSchedule#do_writeback.
  RpaSchedule.dma_q: RpaSchedule#dma_q.
  RpaSchedule.dma_kv_cache: RpaSchedule#dma_kv_cache.
  RpaSchedule.dma_kv_new: RpaSchedule#dma_kv_new.
  RpaSchedule.actual_steps: RpaSchedule#actual_steps.
  RpaSchedule.cfgs: RpaSchedule#cfgs.
  RpaSchedule.create_shape_dtype: RpaSchedule#create_shape_dtype().
  RpaSchedule.get_dma_kv_cache: RpaSchedule#get_dma_kv_cache().
  RpaSchedule.get_dma_kv_new: RpaSchedule#get_dma_kv_new().
  RpaSchedule.get_dma_q: RpaSchedule#get_dma_q().
  RpaSchedule.scratch_shapes: RpaSchedule#scratch_shapes().
  RpaSchedule.in_specs: RpaSchedule#in_specs().
  RpaSchedule.out_specs: RpaSchedule#out_specs().
  compute_metadata: compute_metadata().
  rpa_metadata_schedule_kernel: rpa_metadata_schedule_kernel().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/schedule.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py)

## Classes
### `RpaSchedule`
- def: [`tpu_inference/kernels/experimental/batched_rpa/schedule.py:59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L59)
- doc: Container for metadata arrays with integrated shape/spec logic.
- signature: `class RpaSchedule:`
- members:
  - `create_shape_dtype(cls, cfgs: configs.RpaConfigs)` — [`L75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L75)
  - `get_dma_kv_cache(self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike, page_idx: jax.typing.ArrayLike)` — [`L96`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L96)
  - `get_dma_kv_new(self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike, page_idx: jax.typing.ArrayLike)` — [`L108`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L108)
  - `get_dma_q(self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike)` — [`L121`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L121)
  - `in_specs(self)` — [`L137`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L137) — Returns a Pytree of input BlockSpecs.
  - `out_specs(self)` — [`L152`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L152) — Returns a Pytree of output BlockSpecs.
  - `scratch_shapes(self)` — [`L129`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L129) — Returns a Pytree of SMEM scratch memory.
  - `actual_steps` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L70)
  - `cfgs` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L72)
  - `dma_kv_cache` — [`L68`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L68)
  - `dma_kv_new` — [`L69`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L69)
  - `dma_q` — [`L67`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L67)
  - `do_writeback` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L66)
  - `is_last_k` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L65)
  - `k_idx` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L64)
  - `q_idx` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L63)
  - `s_idx` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L62)

### `SmemWrapper`
- def: [`tpu_inference/kernels/experimental/batched_rpa/schedule.py:30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L30)
- doc: Maps physical 1-D data into logical N-D representation.
- signature: `class SmemWrapper:`
- members:
  - `create_shape_dtype(cls, shape)` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L37)
  - `data` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L33)
  - `shape` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L34)
- protocol/private: `__getitem__`[`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L50), `__setitem__`[`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L53), `_get_pos`[`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L41)

## Functions
- `compute_metadata(cu_q_lens_ref: jax.Ref, kv_lens_ref: jax.Ref, distribution_ref: jax.Ref, schedule: RpaSchedule, lane_lengths_ref: jax.Ref, *, cfgs: configs.RpaConfigs, update_kv_cache: bool=True)` — [`L161`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L161) — Fill metadata using triple nested loop of seq->q->k loop.
- `generate_rpa_metadata(cu_q_lens: jax.Array, kv_lens: jax.Array, distribution: jax.Array, cfgs: configs.RpaConfigs, *, interpret=False, update_kv_cache: bool=True)` — [`L461`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L461)
- `rpa_metadata_schedule_kernel(cu_q_lens_ref: jax.Ref, kv_lens_ref: jax.Ref, distribution_ref: jax.Ref, schedule_hbm_ref: RpaSchedule, schedule_ref: RpaSchedule, lane_lengths_ref: jax.Ref, dma_sem: jax.Ref, *, cfgs: configs.RpaConfigs, update_kv_cache: bool=True)` — [`L345`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/schedule.py#L345) — Generates the HBM-to-VMEM DMA schedule.

