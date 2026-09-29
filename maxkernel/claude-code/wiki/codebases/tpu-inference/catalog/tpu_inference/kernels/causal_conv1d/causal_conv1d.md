---
title: 'Module: tpu_inference/kernels/causal_conv1d/causal_conv1d.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/causal_conv1d/causal_conv1d.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.causal_conv1d.causal_conv1d`/
symbols:
  ConvConfigs: ConvConfigs#
  ConvConfigs.batch_size: ConvConfigs#batch_size.
  ConvConfigs.dim_size: ConvConfigs#dim_size.
  ConvConfigs.kernel_size: ConvConfigs#kernel_size.
  ConvConfigs.tile_size: ConvConfigs#tile_size.
  ConvConfigs.prev_kernel_size: ConvConfigs#prev_kernel_size().
  ConvRhsRef: ConvRhsRef#
  ConvRhsRef.weight: ConvRhsRef#weight.
  ConvRhsRef.bias: ConvRhsRef#bias.
  MetadataRef: MetadataRef#
  MetadataRef.num_tiles: MetadataRef#num_tiles.
  MetadataRef.b_idx_to_s_idx: MetadataRef#b_idx_to_s_idx.
  MetadataRef.b_idx_to_sz_from_old: MetadataRef#b_idx_to_sz_from_old.
  MetadataRef.b_idx_should_write: MetadataRef#b_idx_should_write.
  MetadataRef.s_idx_to_state_idx: MetadataRef#s_idx_to_state_idx.
  MetadataRef.s_idx_has_initial_state: MetadataRef#s_idx_has_initial_state.
  MetadataRef.__len__: MetadataRef#__len__().
  BufferWrapper: BufferWrapper#
  BufferWrapper.hbm_ref: BufferWrapper#hbm_ref.
  BufferWrapper.vmem_ref: BufferWrapper#vmem_ref.
  BufferWrapper.metadata_ref: BufferWrapper#metadata_ref.
  BufferWrapper.cfgs: BufferWrapper#cfgs.
  BufferWrapper.get_slot_vmem: BufferWrapper#get_slot_vmem().
  BufferWrapper.is_lower_oob: BufferWrapper#is_lower_oob().
  BufferWrapper.is_upper_oob: BufferWrapper#is_upper_oob().
  BufferWrapper.copy_in: BufferWrapper#copy_in().
  BufferWrapper.wait_in: BufferWrapper#wait_in().
  BufferWrapper.copy_out: BufferWrapper#copy_out().
  BufferWrapper.wait_out: BufferWrapper#wait_out().
  XBuffer: XBuffer#
  XBuffer.copy_in: XBuffer#copy_in().
  XBuffer.wait_in: XBuffer#wait_in().
  XBuffer.copy_out: XBuffer#copy_out().
  XBuffer.wait_out: XBuffer#wait_out().
  ConvStateBuffer: ConvStateBuffer#
  ConvStateBuffer.copy_in: ConvStateBuffer#copy_in().
  ConvStateBuffer.wait_in: ConvStateBuffer#wait_in().
  ConvStateBuffer.copy_out: ConvStateBuffer#copy_out().
  ConvStateBuffer.wait_out: ConvStateBuffer#wait_out().
  inner_kernel: inner_kernel().
  main_kernel: main_kernel().
  preprocess_metadata: preprocess_metadata().
  ragged_causal_conv1d: ragged_causal_conv1d().
---
# Module: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py)

## Classes
### `BufferWrapper`
- def: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py:64`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L64)
- signature: `class BufferWrapper(ABC):`
- members:
  - `copy_in(self, b_start, slot, sem)` — [`L82`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L82)
  - `copy_out(self, b_start, slot, sem)` — [`L90`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L90)
  - `get_slot_vmem(self, slot)` — [`L70`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L70)
  - `is_lower_oob(self, row)` — [`L73`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L73)
  - `is_upper_oob(self, row)` — [`L76`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L76)
  - `wait_in(self, b_start, slot, sem)` — [`L86`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L86)
  - `wait_out(self, b_start, slot, sem)` — [`L94`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L94)
  - `cfgs` — [`L68`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L68)
  - `hbm_ref` — [`L65`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L65)
  - `metadata_ref` — [`L67`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L67)
  - `vmem_ref` — [`L66`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L66)

### `ConvConfigs`
- def: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py:30`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L30)
- signature: `class ConvConfigs:`
- members:
  - `prev_kernel_size(self)` — [`L37`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L37)
  - `batch_size` — [`L31`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L31)
  - `dim_size` — [`L32`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L32)
  - `kernel_size` — [`L33`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L33)
  - `tile_size` — [`L34`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L34)

### `ConvRhsRef`
- def: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py:43`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L43)
- signature: `class ConvRhsRef:`
- members:
  - `bias` — [`L45`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L45)
  - `weight` — [`L44`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L44)

### `ConvStateBuffer`
- def: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py:139`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L139)
- signature: `class ConvStateBuffer(BufferWrapper):`
- members:
  - `copy_in(self, b_start, slot, sem)` — [`L141`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L141)
  - `copy_out(self, b_start, slot, sem)` — [`L172`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L172)
  - `wait_in(self, b_start, slot, sem)` — [`L160`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L160)
  - `wait_out(self, b_start, slot, sem)` — [`L185`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L185)

### `MetadataRef`
- def: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py:50`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L50)
- signature: `class MetadataRef:`
- members:
  - `b_idx_should_write` — [`L54`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L54)
  - `b_idx_to_s_idx` — [`L52`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L52)
  - `b_idx_to_sz_from_old` — [`L53`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L53)
  - `num_tiles` — [`L51`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L51)
  - `s_idx_has_initial_state` — [`L56`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L56)
  - `s_idx_to_state_idx` — [`L55`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L55)
- protocol/private: `__len__`[`L58`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L58)

### `XBuffer`
- def: [`tpu_inference/kernels/causal_conv1d/causal_conv1d.py:100`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L100)
- signature: `class XBuffer(BufferWrapper):`
- members:
  - `copy_in(self, b_start, slot, sem)` — [`L102`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L102)
  - `copy_out(self, b_start, slot, sem)` — [`L120`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L120)
  - `wait_in(self, b_start, slot, sem)` — [`L113`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L113)
  - `wait_out(self, b_start, slot, sem)` — [`L127`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L127)

## Functions
- `inner_kernel(p_id: jax.Array, *, x_buffer: XBuffer, conv_state_buffer: ConvStateBuffer, sem_ref: jax.Array, metadata_ref: MetadataRef, conv_rhs_ref: ConvRhsRef, prev_x_scratch_ref: jax.Array, cfgs: ConvConfigs)` — [`L202`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L202)
- `main_kernel(metadata_ref: MetadataRef, x_ref: jax.Array, conv_state_ref: jax.Array, conv_rhs_ref: ConvRhsRef, x_out_ref: jax.Array, conv_state_out_ref: jax.Array, x_scratch_ref: jax.Array, conv_state_scratch_ref: jax.Array, prev_x_scratch_ref: jax.Array, sem_ref: jax.Array, *, cfgs: ConvConfigs)` — [`L319`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L319)
- `preprocess_metadata(cfgs: ConvConfigs, query_start_loc: jax.Array, state_indices: jax.Array, has_initial_state: jax.Array, num_seqs: jax.Array)` — [`L379`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L379) — Preprocesses metadata required for DMA, and compute.
- `ragged_causal_conv1d(x: jax.Array, conv_state: jax.Array, conv_weight: jax.Array, conv_bias: jax.Array | None, query_start_loc: jax.Array, state_indices: jax.Array, distribution: jax.Array, has_initial_state: jax.Array, *, kernel_size: int, tile_size: int=64)` — [`L434`](../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/causal_conv1d/causal_conv1d.py#L434) — Perform Conv1D where input is a ragged sequence.

