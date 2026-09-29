---
title: 'Module: tpu_inference/kernels/gdn/v3/memory_ref.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/gdn/v3/memory_ref.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.gdn.v3.memory_ref`/
symbols:
  ConvWeightsRef: ConvWeightsRef#
  ConvWeightsRef.weight: ConvWeightsRef#weight.
  ConvWeightsRef.bias: ConvWeightsRef#bias.
  GDNWeightsRef: GDNWeightsRef#
  GDNWeightsRef.a_log: GDNWeightsRef#a_log.
  GDNWeightsRef.dt_bias: GDNWeightsRef#dt_bias.
  WeightRefs: WeightRefs#
  WeightRefs.conv: WeightRefs#conv.
  WeightRefs.gdn: WeightRefs#gdn.
  SmemWrapper: SmemWrapper#
  SmemWrapper.data: SmemWrapper#data.
  SmemWrapper.shape: SmemWrapper#shape.
  SmemWrapper._get_pos: SmemWrapper#_get_pos().
  SmemWrapper.__getitem__: SmemWrapper#__getitem__().
  MetadataRef: MetadataRef#
  MetadataRef.num_tiles: MetadataRef#num_tiles.
  MetadataRef.p_id_to_s_idx: MetadataRef#p_id_to_s_idx.
  MetadataRef.p_id_to_r_base: MetadataRef#p_id_to_r_base.
  MetadataRef.p_id_to_r_size: MetadataRef#p_id_to_r_size.
  MetadataRef.p_id_is_first_tile: MetadataRef#p_id_is_first_tile.
  MetadataRef.p_id_is_last_tile: MetadataRef#p_id_is_last_tile.
  MetadataRef.s_idx_has_initial_state: MetadataRef#s_idx_has_initial_state.
  MetadataRef.s_idx_to_state_indices: MetadataRef#s_idx_to_state_indices.
  MetadataRef.create: MetadataRef#create().
  MetadataRef.__len__: MetadataRef#__len__().
  BaseBufferedRef: BaseBufferedRef#
  BaseBufferedRef.cfg: BaseBufferedRef#cfg.
  BaseBufferedRef.metadata_ref: BaseBufferedRef#metadata_ref.
  BaseBufferedRef.create: BaseBufferedRef#create().
  InBufferedRef: InBufferedRef#
  InBufferedRef.copy_in: InBufferedRef#copy_in().
  InBufferedRef.wait_in: InBufferedRef#wait_in().
  OutBufferedRef: OutBufferedRef#
  OutBufferedRef.copy_out: OutBufferedRef#copy_out().
  OutBufferedRef.wait_out: OutBufferedRef#wait_out().
  StateBufferedRef: StateBufferedRef#
  StateBufferedRef.copy_in: StateBufferedRef#copy_in().
  StateBufferedRef.wait_in: StateBufferedRef#wait_in().
  StateBufferedRef.copy_out: StateBufferedRef#copy_out().
  StateBufferedRef.wait_out: StateBufferedRef#wait_out().
  create_allocs: create_allocs().
---
# Module: [`tpu_inference/kernels/gdn/v3/memory_ref.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py)

## Classes
### `BaseBufferedRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:112`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L112)
- signature: `class BaseBufferedRef(pltpu.BufferedRef):`
- members:
  - `create(cls, spec: pl.BlockSpec, dtype_or_type: jax.Array, buffer_type: pltpu.BufferType, buffer_count: int, use_lookahead: bool, cfg: config.GDNConfig, metadata_ref: MetadataRef)` — [`L121`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L121)
  - `cfg` — [`L114`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L114)
  - `metadata_ref` — [`L118`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L118)

### `ConvWeightsRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L29)
- signature: `class ConvWeightsRef:`
- members:
  - `bias` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L31)
  - `weight` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L30)

### `GDNWeightsRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:36`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L36)
- signature: `class GDNWeightsRef:`
- members:
  - `a_log` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L37)
  - `dt_bias` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L38)

### `InBufferedRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:151`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L151)
- signature: `class InBufferedRef(BaseBufferedRef):`
- members:
  - `copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L153`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L153)
  - `wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L170`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L170)

### `MetadataRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:71`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L71)
- signature: `class MetadataRef:`
- members:
  - `create(cls, cfgs: config.GDNConfig, num_tiles: jax.Array, p_id_to_s_idx: jax.Array, p_id_to_r_base: jax.Array, p_id_to_r_size: jax.Array, p_id_is_first_tile: jax.Array, p_id_is_last_tile: jax.Array, s_idx_has_initial_state: jax.Array, s_idx_to_state_indices: jax.Array)` — [`L82`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L82)
  - `num_tiles` — [`L72`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L72)
  - `p_id_is_first_tile` — [`L76`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L76)
  - `p_id_is_last_tile` — [`L77`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L77)
  - `p_id_to_r_base` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L74)
  - `p_id_to_r_size` — [`L75`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L75)
  - `p_id_to_s_idx` — [`L73`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L73)
  - `s_idx_has_initial_state` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L78)
  - `s_idx_to_state_indices` — [`L79`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L79)
- protocol/private: `__len__`[`L107`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L107)

### `OutBufferedRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:191`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L191)
- signature: `class OutBufferedRef(BaseBufferedRef):`
- members:
  - `copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L193`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L193)
  - `wait_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L210`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L210)

### `SmemWrapper`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L50)
- doc: Maps physical 1-D data into logical N-D representation.
- signature: `class SmemWrapper:`
- members:
  - `data` — [`L53`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L53)
  - `shape` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L54)
- protocol/private: `__getitem__`[`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L65), `_get_pos`[`L56`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L56)

### `StateBufferedRef`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:231`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L231)
- signature: `class StateBufferedRef(BaseBufferedRef):`
- members:
  - `copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L233`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L233)
  - `copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L280`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L280)
  - `wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L257`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L257)
  - `wait_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array])` — [`L300`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L300)

### `WeightRefs`
- def: [`tpu_inference/kernels/gdn/v3/memory_ref.py:43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L43)
- signature: `class WeightRefs:`
- members:
  - `conv` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L44)
  - `gdn` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L45)

## Functions
- `create_allocs(metadata_ref: MetadataRef, qkv_ref: jax.Array, b_ref: jax.Array, a_ref: jax.Array, out_ref: jax.Array, conv_state_ref: jax.Array, recurrent_state_ref: jax.Array, cfg: config.GDNConfig)` — [`L320`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/gdn/v3/memory_ref.py#L320)

