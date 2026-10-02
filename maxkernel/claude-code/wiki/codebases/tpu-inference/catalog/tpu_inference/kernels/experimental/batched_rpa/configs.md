---
title: 'Module: tpu_inference/kernels/experimental/batched_rpa/configs.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/experimental/batched_rpa/configs.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.experimental.batched_rpa.configs`/
symbols:
  BlockSizes: BlockSizes#
  ModelConfigs.num_q_heads: ModelConfigs#num_q_heads.
  ModelConfigs.num_kv_heads: ModelConfigs#num_kv_heads.
  RpaCase: RpaCase#
  ModelConfigs.head_dim: ModelConfigs#head_dim.
  ModelConfigs: ModelConfigs#
  ServingConfigs: ServingConfigs#
  ServingConfigs.dtype_q: ServingConfigs#dtype_q.
  ServingConfigs.dtype_kv: ServingConfigs#dtype_kv.
  ServingConfigs.dtype_out: ServingConfigs#dtype_out.
  ServingConfigs.packing_kv: ServingConfigs#packing_kv().
  RpaCase.DECODE: RpaCase#DECODE.
  BlockSizes.bq_sz: BlockSizes#bq_sz.
  BlockSizes.bq_c_sz: BlockSizes#bq_c_sz.
  BlockSizes.bkv_sz: BlockSizes#bkv_sz.
  BlockSizes.batch_size: BlockSizes#batch_size.
  BlockSizes.n_buffer: BlockSizes#n_buffer.
  ModelConfigs.mask_value: ModelConfigs#mask_value.
  ModelConfigs.sm_scale: ModelConfigs#sm_scale.
  ModelConfigs.soft_cap: ModelConfigs#soft_cap.
  ModelConfigs.sliding_window: ModelConfigs#sliding_window.
  ModelConfigs.num_q_heads_per_kv_head: ModelConfigs#num_q_heads_per_kv_head().
  ServingConfigs.num_seqs: ServingConfigs#num_seqs.
  ServingConfigs.page_size: ServingConfigs#page_size.
  ServingConfigs.total_q_tokens: ServingConfigs#total_q_tokens.
  ServingConfigs.num_page_indices: ServingConfigs#num_page_indices.
  ServingConfigs.scale_q: ServingConfigs#scale_q.
  ServingConfigs.scale_k: ServingConfigs#scale_k.
  ServingConfigs.scale_v: ServingConfigs#scale_v.
  ServingConfigs.packing_q: ServingConfigs#packing_q().
  RpaCase.MIXED: RpaCase#MIXED.
  RpaConfigs: RpaConfigs#
  RpaConfigs.block: RpaConfigs#block.
  RpaConfigs.model: RpaConfigs#model.
  RpaConfigs.serve: RpaConfigs#serve.
  RpaConfigs.mode: RpaConfigs#mode.
  RpaConfigs.vmem_limit_bytes: RpaConfigs#vmem_limit_bytes.
  RpaConfigs.validate_inputs: RpaConfigs#validate_inputs().
  ServingConfigs.pages_per_seq: ServingConfigs#pages_per_seq().
  ServingConfigs.page_size_log2: ServingConfigs#page_size_log2().
  ServingConfigs.page_size_mask: ServingConfigs#page_size_mask().
  ServingConfigs.int_ty: ServingConfigs#int_ty().
  RpaCase.PREFILL: RpaCase#PREFILL.
  RpaCase.symbol: RpaCase#symbol().
  RpaCase.get_range: RpaCase#get_range().
  RpaConfigs.bq_sz: RpaConfigs#bq_sz().
  RpaConfigs.bq_c_sz: RpaConfigs#bq_c_sz().
  RpaConfigs.bkv_sz: RpaConfigs#bkv_sz().
  RpaConfigs.batch_size: RpaConfigs#batch_size().
  RpaConfigs.n_buffer: RpaConfigs#n_buffer().
  RpaConfigs.max_steps_ub: RpaConfigs#max_steps_ub().
  RpaConfigs.bkv_p: RpaConfigs#bkv_p().
  RpaConfigs.bkv_p_cache: RpaConfigs#bkv_p_cache().
  RpaConfigs.bkv_p_new: RpaConfigs#bkv_p_new().
  RpaConfigs.bkv_stride: RpaConfigs#bkv_stride().
  RpaConfigs.aligned_head_dim: RpaConfigs#aligned_head_dim().
  RpaConfigs.aligned_num_kv_heads_x2: RpaConfigs#aligned_num_kv_heads_x2().
  RpaConfigs.aligned_num_q_heads_per_kv_head: RpaConfigs#aligned_num_q_heads_per_kv_head().
  RpaConfigs.kv_hbm_stride: RpaConfigs#kv_hbm_stride().
  RpaConfigs.fuse_accum: RpaConfigs#fuse_accum().
  RpaConfigs.q_vmem_shape: RpaConfigs#q_vmem_shape().
  RpaConfigs.kv_vmem_shape: RpaConfigs#kv_vmem_shape().
  RpaConfigs.lm_scratch_shape: RpaConfigs#lm_scratch_shape().
  RpaConfigs.acc_scratch_shape: RpaConfigs#acc_scratch_shape().
---
# Module: [`tpu_inference/kernels/experimental/batched_rpa/configs.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py)

## Classes
### `BlockSizes`
- def: [`tpu_inference/kernels/experimental/batched_rpa/configs.py:27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L27)
- doc: Tuning parameters for the RPA kernel.
- signature: `class BlockSizes:`
- members:
  - `batch_size` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L33)
  - `bkv_sz` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L32)
  - `bq_c_sz` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L31)
  - `bq_sz` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L30)
  - `n_buffer` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L34)
- used by: [`ragged_paged_attention`](wrapper.md#ragged_paged_attention), [`calculate_block_sizes`](wrapper.md#calculate_block_sizes), [`find_best_block_sizes`](wrapper.md#calculate_block_sizes.find_best_block_sizes)

### `ModelConfigs`
- def: [`tpu_inference/kernels/experimental/batched_rpa/configs.py:38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L38)
- doc: Model config that will always stay constant.
- signature: `class ModelConfigs:`
- members:
  - `num_q_heads_per_kv_head(self)` — [`L50`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L50)
  - `head_dim` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L43)
  - `mask_value` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L44)
  - `num_kv_heads` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L42)
  - `num_q_heads` — [`L41`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L41)
  - `sliding_window` — [`L47`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L47)
  - `sm_scale` — [`L45`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L45)
  - `soft_cap` — [`L46`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L46)
- used by: [`ragged_paged_attention`](wrapper.md#ragged_paged_attention), [`calculate_block_sizes`](wrapper.md#calculate_block_sizes), [`calculate_vmem_usage`](wrapper.md#calculate_block_sizes.calculate_vmem_usage), [`calculate_compute_buffer_time`](wrapper.md#calculate_block_sizes.calculate_compute_buffer_time)

### `RpaCase`
- def: [`tpu_inference/kernels/experimental/batched_rpa/configs.py:101`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L101)
- doc: Represents the different cases for Ragged Paged Attention.
- signature: `class RpaCase(enum.StrEnum):`
- members:
  - `get_range(self, distribution: jax.Array)` — [`L121`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L121)
  - `symbol(self)` — [`L114`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L114)
  - `DECODE` — [`L109`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L109)
  - `MIXED` — [`L111`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L111)
  - `PREFILL` — [`L110`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L110)
- used by: [`ragged_paged_attention`](wrapper.md#ragged_paged_attention), [`run_rpa_kernel`](wrapper.md#ragged_paged_attention.run_rpa_kernel)

### `RpaConfigs`
- def: [`tpu_inference/kernels/experimental/batched_rpa/configs.py:135`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L135)
- signature: `class RpaConfigs:`
- members:
  - `acc_scratch_shape(self)` — [`L282`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L282)
  - `aligned_head_dim(self)` — [`L224`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L224)
  - `aligned_num_kv_heads_x2(self)` — [`L229`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L229)
  - `aligned_num_q_heads_per_kv_head(self)` — [`L234`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L234)
  - `batch_size(self)` — [`L157`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L157)
  - `bkv_p(self)` — [`L199`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L199)
  - `bkv_p_cache(self)` — [`L203`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L203)
  - `bkv_p_new(self)` — [`L209`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L209)
  - `bkv_stride(self)` — [`L215`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L215)
  - `bkv_sz(self)` — [`L153`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L153)
  - `bq_c_sz(self)` — [`L149`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L149)
  - `bq_sz(self)` — [`L145`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L145)
  - `fuse_accum(self)` — [`L245`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L245)
  - `kv_hbm_stride(self)` — [`L239`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L239)
  - `kv_vmem_shape(self)` — [`L262`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L262)
  - `lm_scratch_shape(self)` — [`L272`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L272)
  - `max_steps_ub(self)` — [`L167`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L167) — Get maximum upper bound of kernel steps based on SMEM limit.
  - `n_buffer(self)` — [`L161`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L161)
  - `q_vmem_shape(self)` — [`L249`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L249)
  - `validate_inputs(self, q: jax.Array, k: jax.Array, v: jax.Array, kv_cache: jax.Array, kv_lens: jax.Array, page_indices: jax.Array, cu_q_lens: jax.Array, distribution: jax.Array)` — [`L290`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L290) — Validate inputs to the RPA kernel statically.
  - `block` — [`L136`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L136)
  - `mode` — [`L139`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L139)
  - `model` — [`L137`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L137)
  - `serve` — [`L138`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L138)
  - `vmem_limit_bytes` — [`L140`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L140)
- used by: [`run_rpa_kernel`](wrapper.md#ragged_paged_attention.run_rpa_kernel)

### `ServingConfigs`
- def: [`tpu_inference/kernels/experimental/batched_rpa/configs.py:55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L55)
- doc: Serving config that can change depending on use cases.
- signature: `class ServingConfigs:`
- members:
  - `int_ty(self)` — [`L82`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L82)
  - `packing_kv(self)` — [`L97`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L97)
  - `packing_q(self)` — [`L93`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L93)
  - `page_size_log2(self)` — [`L74`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L74)
  - `page_size_mask(self)` — [`L78`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L78)
  - `pages_per_seq(self)` — [`L70`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L70)
  - `dtype_kv` — [`L63`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L63)
  - `dtype_out` — [`L64`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L64)
  - `dtype_q` — [`L62`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L62)
  - `num_page_indices` — [`L61`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L61)
  - `num_seqs` — [`L58`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L58)
  - `page_size` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L59)
  - `scale_k` — [`L66`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L66)
  - `scale_q` — [`L65`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L65)
  - `scale_v` — [`L67`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L67)
  - `total_q_tokens` — [`L60`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/experimental/batched_rpa/configs.py#L60)
- used by: [`ragged_paged_attention`](wrapper.md#ragged_paged_attention), [`calculate_block_sizes`](wrapper.md#calculate_block_sizes)

