---
title: 'Module: jax/_src/pallas/mosaic/interpret/race_detection_state.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/mosaic/interpret/race_detection_state.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.mosaic.interpret.race_detection_state`/
symbols:
  RaceDetectionState.check_write: RaceDetectionState#check_write().
  RaceDetectionState.check_read: RaceDetectionState#check_read().
  RaceDetectionState: RaceDetectionState#
  _ranges_overlap: _ranges_overlap().
  RaceDetectionState.lock: RaceDetectionState#lock.
  _slices_overlap: _slices_overlap().
  RaceDetectionState.writes: RaceDetectionState#writes.
  RaceDetectionState.races_found: RaceDetectionState#races_found.
  _is_empty_slice: _is_empty_slice().
  RaceDetectionState.num_cores: RaceDetectionState#num_cores.
  RaceDetectionState.reads: RaceDetectionState#reads.
---
# Module: [`jax/_src/pallas/mosaic/interpret/race_detection_state.py`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py)

## Classes
### `RaceDetectionState`
- def: [`jax/_src/pallas/mosaic/interpret/race_detection_state.py:71`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L71)
- signature: `class RaceDetectionState:`
- members:
  - `check_read(self, thread: ThreadKey, clock, buffer_key, rnge, source_info=None)` — [`L89`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L89)
  - `check_write(self, thread: ThreadKey, clock, buffer_key, rnge, source_info=None)` — [`L127`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L127)
  - `lock` — [`L85`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L85)
  - `num_cores` — [`L73`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L73)
  - `races_found` — [`L87`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L87)
  - `reads` — [`L76`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L76)
  - `writes` — [`L81`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L81)
- uses (calls/refs, reference-scoped): [`summarize`](../../../source_info_util.md#summarize), [`ordered`](vector_clock.md#ordered), [`_ranges_overlap`](race_detection_state.md#_ranges_overlap)
- used by: [`get`](interpret_pallas_call.md#get), [`_get`](../../mosaic_gpu/interpret/gpu_callbacks.md#_get), [`_swap`](../../mosaic_gpu/interpret/gpu_callbacks.md#_swap), [`store`](interpret_pallas_call.md#store), [`post_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopySmemToGmemTask.post_read), [`pre_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopySmemToGmemTask.pre_read), [`_initialize_shared_memory`](../../mosaic_gpu/interpret/gpu_callbacks.md#_initialize_shared_memory), [`post_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyGmemToSmemTask.post_read), [`pre_read`](../../mosaic_gpu/interpret/gpu_callbacks.md#AsyncCopyGmemToSmemTask.pre_read), [`swap`](interpret_pallas_call.md#swap), [`_initialize_shared_memory`](interpret_pallas_call.md#_initialize_shared_memory), [`races`](interpret_pallas_call.md#races.races), [`get_races`](../../mosaic_gpu/interpret/gpu_callbacks.md#get_races), [`_races`](../../mosaic_gpu/interpret/gpu_callbacks.md#_races._races), [`get_races`](../../mosaic_gpu/interpret/interpret_pallas_call.md#get_races)

## Functions
- `_is_empty_slice(slice_or_idx: slice | int)` — [`L24`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L24)
- `_ranges_overlap(range1: tuple[slice | int, ...], range2: tuple[slice | int, ...])` — [`L61`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L61)
- `_slices_overlap(slice_or_idx1: slice | int, slice_or_idx2: slice | int)` — [`L34`](../../../../../../../../../raw/code/jax/jax/_src/pallas/mosaic/interpret/race_detection_state.py#L34)

