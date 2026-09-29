---
title: 'Module: axlearn/common/flash_attention/tpu_splash_attention.py'
type: catalog
provenance: extracted
module: axlearn/common/flash_attention/tpu_splash_attention.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.flash_attention.tpu_splash_attention`/
symbols:
  _splash_attention_bwd_dkv: _splash_attention_bwd_dkv().
  _splash_attention_bwd_dq: _splash_attention_bwd_dq().
  NUM_LANES: NUM_LANES.
  SplashAttentionKernel.manual_sharding_spec: SplashAttentionKernel#manual_sharding_spec().
  SplashAttentionKernel.__call__: SplashAttentionKernel#__call__().
  SplashAttentionKernel.tree_flatten: SplashAttentionKernel#tree_flatten().
  NUM_SUBLANES: NUM_SUBLANES.
  flash_attention_kernel.body: flash_attention_kernel().body().
  _flash_attention_dkv_kernel.body: _flash_attention_dkv_kernel().body().
  SplashAttentionKernel.fwd_mask_info: SplashAttentionKernel#fwd_mask_info.
  make_splash_mha: make_splash_mha.
  _splash_attention_bwd: _splash_attention_bwd().
  get_dropout_mask: get_dropout_mask().
  partial: partial.
  NN_DIM_NUMBERS: NN_DIM_NUMBERS.
  NT_DIM_NUMBERS: NT_DIM_NUMBERS.
  _splash_attention_custom: _splash_attention_custom().
  _splash_attention: _splash_attention().
  _make_splash_attention: _make_splash_attention().
  _splash_attention_fwd: _splash_attention_fwd().
  _flash_attention_dq_kernel.run: _flash_attention_dq_kernel().run().
  make_splash_mqa: make_splash_mqa.
  make_splash_mha_single_device: make_splash_mha_single_device.
  make_splash_mqa_single_device: make_splash_mqa_single_device.
  SplashAttentionKernel.kwargs: SplashAttentionKernel#kwargs.
  _splash_attention_forward: _splash_attention_forward().
  SplashAttentionKernel: SplashAttentionKernel#
  flash_attention_kernel: flash_attention_kernel().
  _get_dropout_mask_kernel: _get_dropout_mask_kernel().
  _generate_blockwise_dropout_mask: _generate_blockwise_dropout_mask().
  SplashAttentionKernel.dq_mask_info: SplashAttentionKernel#dq_mask_info.
  SplashAttentionKernel.dkv_mask_info: SplashAttentionKernel#dkv_mask_info.
  flash_attention_kernel.run: flash_attention_kernel().run().
  flash_attention_kernel.end: flash_attention_kernel().end().
  _flash_attention_dkv_kernel.run: _flash_attention_dkv_kernel().run().
  SplashAttentionKernel.tree_unflatten: SplashAttentionKernel#tree_unflatten().
  DEFAULT_MASK_VALUE: DEFAULT_MASK_VALUE.
  SplashResidualsType: SplashResidualsType.
  _splash_attention_forward.q_segment_ids_index_map: _splash_attention_forward().q_segment_ids_index_map().
  _splash_attention_bwd_dq.q_segment_ids_index_map: _splash_attention_bwd_dq().q_segment_ids_index_map().
  _splash_attention_bwd_dq.logsumexp_index_map: _splash_attention_bwd_dq().logsumexp_index_map().
  _flash_attention_dkv_kernel.body._load_kv: _flash_attention_dkv_kernel().body()._load_kv().
  _splash_attention_bwd_dkv.dkv_index_map: _splash_attention_bwd_dkv().dkv_index_map().
  _splash_attention_bwd_dkv.q_segment_ids_index_map: _splash_attention_bwd_dkv().q_segment_ids_index_map().
  _splash_attention_bwd_dkv.logsumexp_index_map: _splash_attention_bwd_dkv().logsumexp_index_map().
  _splash_attention_forward.q_index_map: _splash_attention_forward().q_index_map().
  _splash_attention_forward.out_index_map: _splash_attention_forward().out_index_map().
  _splash_attention_forward.k_index_map: _splash_attention_forward().k_index_map().
  _splash_attention_forward.v_index_map: _splash_attention_forward().v_index_map().
  _splash_attention_forward.mask_index_map: _splash_attention_forward().mask_index_map().
  _splash_attention_forward.kv_segment_ids_index_map: _splash_attention_forward().kv_segment_ids_index_map().
  _splash_attention_forward.logsumexp_index_map: _splash_attention_forward().logsumexp_index_map().
  _flash_attention_dq_kernel: _flash_attention_dq_kernel().
  _splash_attention_bwd_dq.o_index_map: _splash_attention_bwd_dq().o_index_map().
  _splash_attention_bwd_dq.q_index_map: _splash_attention_bwd_dq().q_index_map().
  _splash_attention_bwd_dq.k_index_map: _splash_attention_bwd_dq().k_index_map().
  _splash_attention_bwd_dq.v_index_map: _splash_attention_bwd_dq().v_index_map().
  _splash_attention_bwd_dq.mask_index_map: _splash_attention_bwd_dq().mask_index_map().
  _splash_attention_bwd_dq.kv_segment_ids_index_map: _splash_attention_bwd_dq().kv_segment_ids_index_map().
  _flash_attention_dkv_kernel: _flash_attention_dkv_kernel().
  _splash_attention_bwd_dkv.o_index_map: _splash_attention_bwd_dkv().o_index_map().
  _splash_attention_bwd_dkv.q_index_map: _splash_attention_bwd_dkv().q_index_map().
  _splash_attention_bwd_dkv.k_index_map: _splash_attention_bwd_dkv().k_index_map().
  _splash_attention_bwd_dkv.v_index_map: _splash_attention_bwd_dkv().v_index_map().
  _splash_attention_bwd_dkv.dq_index_map: _splash_attention_bwd_dkv().dq_index_map().
  _splash_attention_bwd_dkv.mask_index_map: _splash_attention_bwd_dkv().mask_index_map().
  _splash_attention_bwd_dkv.kv_segment_ids_index_map: _splash_attention_bwd_dkv().kv_segment_ids_index_map().
  get_dropout_mask.out_index_map: get_dropout_mask().out_index_map().
  flash_attention_kernel.init: flash_attention_kernel().init().
  _flash_attention_dq_kernel.init: _flash_attention_dq_kernel().init().
  _flash_attention_dq_kernel.end: _flash_attention_dq_kernel().end().
  _flash_attention_dkv_kernel.init: _flash_attention_dkv_kernel().init().
  _flash_attention_dkv_kernel.end: _flash_attention_dkv_kernel().end().
  SplashAttentionKernel.__init__: SplashAttentionKernel#__init__().
---
# Module: [`axlearn/common/flash_attention/tpu_splash_attention.py`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py)

## Classes
### `SplashAttentionKernel`
- def: [`axlearn/common/flash_attention/tpu_splash_attention.py:1959`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1959)
- doc: A class wrapper around the splash attention kernel.
- signature: `class SplashAttentionKernel:`
- members:
  - `manual_sharding_spec(self, sharding: jax.sharding.NamedSharding)` — [`L1984`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1984) — Returns a value that can be used as a shard_map partition spec for the kernel.
  - `tree_flatten(self)` — [`L2017`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2017)
  - `tree_unflatten(cls, kwargs, values)` — [`L2024`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2024)
  - `dkv_mask_info` — [`L1972`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1972)
  - `dq_mask_info` — [`L1971`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1971)
  - `fwd_mask_info` — [`L1970`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1970)
  - `kwargs` — [`L1969`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1969)
- protocol/private: `__call__`[`L1974`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1974), `__init__`[`L1962`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1962)
- uses (calls/refs, reference-scoped): [`_splash_attention`](tpu_splash_attention.md#_splash_attention)
- used by: [`build`](tpu_attention.md#TPUSplashAttentionWithAllGather.build), [`__call__`](tpu_attention.md#TPUIncrementalSplashAttention.__call__), [`_make_splash_attention`](tpu_splash_attention.md#_make_splash_attention)

## Functions
- `_flash_attention_dkv_kernel(data_next_ref, block_mask_ref, mask_next_ref, prng_key, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dk_scratch_ref, dv_scratch_ref, dq_ref, dk_ref, dv_ref, *, num_q_heads: int, num_kv_heads: int, mask_value: float, grid_width: int, bq: int, bkv_compute: int, is_mqa: bool, attn_logits_soft_cap: float | None, dropout_rate: float, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, bkv: int, mask_function: MaskFunctionType | None)` — [`L1138`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1138)
- `_flash_attention_dq_kernel(data_next_ref, block_mask_ref, mask_next_ref, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, logsumexp_ref, do_ref, di_ref, mask_ref, q_sequence_ref, dq_scratch_ref, dq_ref, *, mask_value: float, grid_width: int, bq: int, bkv: int, attn_logits_soft_cap: float | None = None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None)` — [`L788`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L788)
- `_generate_blockwise_dropout_mask(prng_key: jax.Array, head_idx: int, q_block_idx: int, kv_block_idx: int, q_block_size: int, kv_block_size: int, dropout_rate: float)` — [`L91`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L91)
- `_get_dropout_mask_kernel(prng_key: jax.Array, output_ref: jax.Array, *, bq: int, bkv_compute: int, dropout_rate: float)` — [`L2127`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2127)
- `_load_kv(ref, layout)` — [`L1226`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1226)
- `_make_splash_attention(mask: np.ndarray | mask_lib.MultiHeadMask, *, block_sizes: BlockSizes | None = None, is_mqa: bool, save_residuals: bool = False, mask_value: float = DEFAULT_MASK_VALUE, attn_logits_soft_cap: float | None = None, dropout_rate: float = 0, downcast_smem_data: bool = True, head_shards: int, q_seq_shards: int, residual_checkpoint_name: str | None = None, interpret: bool = False)` — [`L2039`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2039)
- `_splash_attention(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None = None, logit_sink: jax.Array | None = None, *, is_mqa: bool, block_sizes: BlockSizes | None, save_residuals: bool, mask_value: float, attn_logits_soft_cap: float | None, dropout_rate: float, prng_key: jax.Array | None = None, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, q_positions: jax.Array | None = None, interpret: bool)` — [`L1903`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1903)
- `_splash_attention_bwd(save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, attn_logits_soft_cap: float | None, dropout_rate: float, interpret: bool, res: SplashResidualsType, do: jax.Array)` — [`L1746`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1746)
- `_splash_attention_bwd_dkv(q, k, v, segment_ids, logsumexp, do, di, *, bq: int, bkv: int, bkv_compute: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, attn_logits_soft_cap: float | None, dropout_rate: float, prng_key: jax.Array | None, use_fused_bwd_kernel: bool, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, q_positions: jax.Array | None = None, interpret: bool)` — [`L1369`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1369)
- `_splash_attention_bwd_dq(q, k, v, segment_ids, logsumexp, do, di, *, bq: int, bkv: int, is_mqa: bool, mask_info: mask_info_lib.MaskInfo, mask_value: float, attn_logits_soft_cap: float | None, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, mask_function: MaskFunctionType | None, q_positions: jax.Array | None = None, interpret: bool)` — [`L911`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L911)
- `_splash_attention_custom(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, logit_sink: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, q_positions: jax.Array | None = None, attn_logits_soft_cap: float | None = None, dropout_rate: float = 0, prng_key: jax.Array | None = None, interpret: bool = False)` — [`L670`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L670)
- `_splash_attention_forward(fwd_mask_info: mask_info_lib.MaskInfo, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, logit_sink: jax.Array | None, dropout_mask: jax.Array | None, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, save_residuals: Literal[False] = False, q_positions: jax.Array | None = None, attn_logits_soft_cap: float | None = None, dropout_rate: float = 0, prng_key: jax.Array | None = None)` — [`L287`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L287)
- `_splash_attention_fwd(fwd_mask_info: mask_info_lib.MaskInfo, dq_mask_info: mask_info_lib.MaskInfo | None, dkv_mask_info: mask_info_lib.MaskInfo | None, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: SegmentIds | None, logit_sink: jax.Array | None, save_residuals: bool, mask_value: float, is_mqa: bool, block_sizes: BlockSizes, residual_checkpoint_name: str | None, mask_function: MaskFunctionType | None, q_positions: jax.Array | None = None, attn_logits_soft_cap: float | None = None, dropout_rate: float = 0, prng_key: jax.Array | None = None, interpret: bool = False)` — [`L725`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L725)
- `body(kv_compute_index, _)` — [`L172`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L172)
- `body(i, _)` — [`L1222`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1222)
- `dkv_index_map(kv_index, head_index, *_)` — [`L1517`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1517)
- `dq_index_map(kv_index, head_index, q_index, *_)` — [`L1504`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1504)
- `end()` — [`L268`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L268)
- `end()` — [`L906`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L906)
- `end()` — [`L1359`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1359)
- `flash_attention_kernel(data_next_ref, block_mask_ref, mask_next_ref, prng_key: jax.Array | None, q_ref, k_ref, v_ref, q_segment_ids_ref, kv_segment_ids_ref, mask_ref, q_sequence_ref, logit_sink_ref: jax.Array | None, m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref, logsumexp_ref=None, *, mask_value: float, grid_width: int, bq: int, bkv: int, bkv_compute: int, head_dim: int, q_layout: QKVLayout, k_layout: QKVLayout, v_layout: QKVLayout, attn_logits_soft_cap: float | None, mask_function: MaskFunctionType | None, dropout_rate: float)` — [`L109`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L109)
- `get_dropout_mask(query: jax.Array, key: jax.Array, block_sizes: BlockSizes, dropout_rate: float, prng_key: jax.Array)` — [`L2149`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2149) — Generates a dropout mask for debugging purposes.
- `init()` — [`L153`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L153)
- `init()` — [`L825`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L825)
- `init()` — [`L1208`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1208)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L449`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L449)
- `k_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L980`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L980)
- `k_index_map(kv_index, head_index, *_)` — [`L1478`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1478)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L496`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L496)
- `kv_segment_ids_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1015`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1015)
- `kv_segment_ids_index_map(kv_index, *_)` — [`L1577`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1577)
- `logsumexp_index_map(h, i, *_)` — [`L578`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L578)
- `logsumexp_index_map(h, i, *_)` — [`L1029`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1029)
- `logsumexp_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L1590`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1590)
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L479`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L479)
- `mask_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L1003`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1003)
- `mask_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref, prng_key=None)` — [`L1531`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1531)
- `o_index_map(h, i, *_)` — [`L970`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L970)
- `o_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L1432`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1432)
- `out_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L435`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L435)
- `out_index_map(h, i, j, prng_key=None)` — [`L2178`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2178)
- `q_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L423`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L423)
- `q_index_map(h, i, *_)` — [`L975`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L975)
- `q_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L1455`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1455)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L492`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L492)
- `q_segment_ids_index_map(h, i, j, *_)` — [`L1009`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1009)
- `q_segment_ids_index_map(kv_index, head_index, q_index, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L1554`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1554)
- `run()` — [`L262`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L262)
- `run()` — [`L833`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L833)
- `run()` — [`L1342`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1342)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref=None, prng_key=None)` — [`L465`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L465)
- `v_index_map(h, i, j, data_next_ref, block_mask_ref, mask_next_ref, *_)` — [`L990`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L990)
- `v_index_map(kv_index, head_index, *_)` — [`L1490`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L1490)

## Module values
- `DEFAULT_MASK_VALUE` — [`L68`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L68)
- `NN_DIM_NUMBERS` — [`L72`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L72)
- `NT_DIM_NUMBERS` — [`L73`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L73)
- `NUM_LANES` — [`L69`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L69)
- `NUM_SUBLANES` — [`L70`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L70)
- `SplashResidualsType` — [`L76`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L76)
- `make_splash_mha` — [`L2117`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2117)
- `make_splash_mha_single_device` — [`L2120`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2120)
- `make_splash_mqa` — [`L2118`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2118)
- `make_splash_mqa_single_device` — [`L2124`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L2124)
- `partial` — [`L67`](../../../../../../../raw/code/axlearn/axlearn/common/flash_attention/tpu_splash_attention.py#L67)

