---
title: 'Module: ejkernel/ops/utils/datacarrier.py'
type: catalog
provenance: extracted
module: ejkernel/ops/utils/datacarrier.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.ops.utils.datacarrier`/
symbols:
  FwdParams: FwdParams#
  BwdParams: BwdParams#
  FwdParams.kv_blocksize: FwdParams#kv_blocksize.
  BwdParams.kv_blocksize: BwdParams#kv_blocksize.
  FwdParams.q_blocksize: FwdParams#q_blocksize.
  BwdParams.q_blocksize: BwdParams#q_blocksize.
  FwdParams.num_warps: FwdParams#num_warps.
  FwdParams.num_stages: FwdParams#num_stages.
  BwdParams.num_warps: BwdParams#num_warps.
  BwdParams.num_stages: BwdParams#num_stages.
  hash_fn: hash_fn().
  FwdParams.blocksize_heads: FwdParams#blocksize_heads.
  FwdParams.num_key_splits: FwdParams#num_key_splits.
  FwdParams.__hash__: FwdParams#__hash__.
  get_safe_hash_int: get_safe_hash_int().
  FwdParams.blocksize_m: FwdParams#blocksize_m.
  FwdParams.blocksize_k: FwdParams#blocksize_k.
  FwdParams.blocksize_n: FwdParams#blocksize_n.
  FwdParams.blocksize_keys: FwdParams#blocksize_keys.
  BwdParams.blocksize_m: BwdParams#blocksize_m.
  BwdParams.blocksize_k: BwdParams#blocksize_k.
  BwdParams.blocksize_n: BwdParams#blocksize_n.
  BwdParams.__hash__: BwdParams#__hash__.
---
# Module: [`ejkernel/ops/utils/datacarrier.py`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py)

## Classes
### `BwdParams`
- def: [`ejkernel/ops/utils/datacarrier.py:148`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L148) — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)
- doc: Backward pass parameters for kernel configuration.
- signature: `class BwdParams:`
- members:
  - `blocksize_k` — [`L169`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L169)
  - `blocksize_m` — [`L168`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L168)
  - `blocksize_n` — [`L170`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L170)
  - `kv_blocksize` — [`L172`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L172) — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)
  - `num_stages` — [`L174`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L174)
  - `num_warps` — [`L173`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L173)
  - `q_blocksize` — [`L171`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L171)
- protocol/private: `__hash__`[`L176`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L176)
- uses (calls/refs, reference-scoped): [`hash_fn`](datacarrier.md#hash_fn)
- used by: [`flash_attention`](../../kernels/_pallas/tpu/flash_attention/_interface.md#flash_attention), [`blocksparse_attention`](../../kernels/_pallas/tpu/blocksparse_attention/_kernel.md#blocksparse_attention), [`run`](../../modules/operations/scaled_dot_product_attention.md#ScaledDotProductAttention.run), [`candidate_cfgs_gpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_tpu), [`candidate_cfgs_tpu`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs_tpu), [`candidate_cfgs_xla`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_xla), [`candidate_cfgs_xla`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs_xla), [`candidate_cfgs_gpu`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs_gpu), [`candidate_cfgs_gpu`](../../modules/operations/ring_attention.md#RingAttention.candidate_cfgs_gpu), [`heuristic_cfg`](../../modules/operations/ring_attention.md#RingAttention.heuristic_cfg), [`heuristic_cfg_gpu`](../../modules/operations/flash_attention.md#FlashAttention.heuristic_cfg_gpu), [`run`](../../modules/operations/attention.md#Attention.run), [`_make_block_sizes`](../../kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.md#_make_block_sizes), [`candidate_cfgs`](../../modules/operations/ring_attention.md#RingAttention.candidate_cfgs), [`candidate_cfgs_tpu`](../../modules/operations/ring_attention.md#RingAttention.candidate_cfgs_tpu), [`heuristic_cfg`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.heuristic_cfg), [`candidate_cfgs`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs), [`candidate_cfgs`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs), [`heuristic_cfg`](../../modules/operations/flash_attention.md#FlashAttention.heuristic_cfg), [`heuristic_cfg_gpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_gpu), [`heuristic_cfg_tpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_tpu), [`heuristic_cfg_tpu`](../../modules/operations/flash_attention.md#FlashAttention.heuristic_cfg_tpu), [`ring_attention`](../../kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.md#ring_attention), [`ring_attention`](../../kernels/_xla/ring_attention/_interface.md#ring_attention), [`ring_attention`](../../kernels/_pallas/tpu/ring_attention/_interface.md#ring_attention), [`attention`](../../kernels/_xla/attention/_interface.md#attention), [`blocksparse_attention`](../../kernels/_pallas/tpu/blocksparse_attention/_interface.md#blocksparse_attention), [`bwd_params`](../../modules/operations/configs.md#BlockSparseAttentionConfig.bwd_params), [`bwd_params`](../../modules/operations/configs.md#FlashAttentionConfig.bwd_params), [`bwd_params`](../../modules/operations/configs.md#RingAttentionConfig.bwd_params), [`__post_init__`](../../modules/operations/configs.md#BlockSparseAttentionConfig.__post_init__), [`__post_init__`](../../modules/operations/configs.md#FlashAttentionConfig.__post_init__), [`__post_init__`](../../modules/operations/configs.md#RingAttentionConfig.__post_init__)

### `FwdParams`
- def: [`ejkernel/ops/utils/datacarrier.py:108`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L108) — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)
- doc: Forward pass parameters for kernel configuration.
- signature: `class FwdParams:`
- members:
  - `blocksize_heads` — [`L137`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L137)
  - `blocksize_k` — [`L132`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L132)
  - `blocksize_keys` — [`L138`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L138)
  - `blocksize_m` — [`L131`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L131)
  - `blocksize_n` — [`L133`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L133)
  - `kv_blocksize` — [`L135`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L135) — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)
  - `num_key_splits` — [`L139`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L139)
  - `num_stages` — [`L142`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L142)
  - `num_warps` — [`L141`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L141)
  - `q_blocksize` — [`L134`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L134)
- protocol/private: `__hash__`[`L144`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L144)
- uses (calls/refs, reference-scoped): [`hash_fn`](datacarrier.md#hash_fn)
- used by: [`flash_attention`](../../kernels/_pallas/tpu/flash_attention/_interface.md#flash_attention), [`blocksparse_attention`](../../kernels/_pallas/tpu/blocksparse_attention/_kernel.md#blocksparse_attention), [`run`](../../modules/operations/scaled_dot_product_attention.md#ScaledDotProductAttention.run), [`candidate_cfgs_gpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_gpu), [`candidate_cfgs_tpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_tpu), [`candidate_cfgs_tpu`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs_tpu), [`candidate_cfgs_xla`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs_xla), [`candidate_cfgs_xla`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs_xla), [`candidate_cfgs_gpu`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs_gpu), [`candidate_cfgs_gpu`](../../modules/operations/ring_attention.md#RingAttention.candidate_cfgs_gpu), [`heuristic_cfg`](../../modules/operations/ring_attention.md#RingAttention.heuristic_cfg), [`heuristic_cfg_gpu`](../../modules/operations/flash_attention.md#FlashAttention.heuristic_cfg_gpu), [`run`](../../modules/operations/attention.md#Attention.run), [`_make_block_sizes`](../../kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.md#_make_block_sizes), [`candidate_cfgs`](../../modules/operations/ring_attention.md#RingAttention.candidate_cfgs), [`candidate_cfgs_tpu`](../../modules/operations/ring_attention.md#RingAttention.candidate_cfgs_tpu), [`heuristic_cfg`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.heuristic_cfg), [`candidate_cfgs`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.candidate_cfgs), [`candidate_cfgs`](../../modules/operations/flash_attention.md#FlashAttention.candidate_cfgs), [`heuristic_cfg`](../../modules/operations/flash_attention.md#FlashAttention.heuristic_cfg), [`heuristic_cfg_gpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_gpu), [`heuristic_cfg_tpu`](../../modules/operations/blocksparse_attention.md#BlockSparseAttention.heuristic_cfg_tpu), [`heuristic_cfg_tpu`](../../modules/operations/flash_attention.md#FlashAttention.heuristic_cfg_tpu), [`ring_attention`](../../kernels/_pallas/tpu/ring_attention/_pallas_impl_fwd.md#ring_attention), [`try_add`](../../modules/operations/ragged_decode_attention.md#RaggedDecodeAttention.try_add), [`candidate_cfgs`](../../modules/operations/ragged_decode_attention.md#RaggedDecodeAttention.candidate_cfgs), [`ring_attention`](../../kernels/_xla/ring_attention/_interface.md#ring_attention), [`ring_attention`](../../kernels/_pallas/tpu/ring_attention/_interface.md#ring_attention), [`attention`](../../kernels/_xla/attention/_interface.md#attention), [`blocksparse_attention`](../../kernels/_pallas/tpu/blocksparse_attention/_interface.md#blocksparse_attention), [`ragged_decode_attention`](../../kernels/_pallas/tpu/ragged_decode_attention/_interface.md#ragged_decode_attention), [`_fwd_params_for_seq_len`](../../modules/operations/ragged_decode_attention.md#RaggedDecodeAttention._fwd_params_for_seq_len), [`_default_fwd_params`](../../modules/operations/ragged_decode_attention.md#RaggedDecodeAttention._default_fwd_params), [`inner_decode_tpu`](../../kernels/_pallas/tpu/ragged_decode_attention/_pallas_impl_fwd.md#inner_decode_tpu), [`fwd_params`](../../modules/operations/configs.md#BlockSparseAttentionConfig.fwd_params), [`fwd_params`](../../modules/operations/configs.md#FlashAttentionConfig.fwd_params), [`fwd_params`](../../modules/operations/configs.md#RaggedDecodeAttentionConfig.fwd_params), [`fwd_params`](../../modules/operations/configs.md#RingAttentionConfig.fwd_params), [`__post_init__`](../../modules/operations/configs.md#BlockSparseAttentionConfig.__post_init__), [`__post_init__`](../../modules/operations/configs.md#FlashAttentionConfig.__post_init__)  (+2 more)

## Functions
- `get_safe_hash_int(text, algorithm="md5")` — [`L42`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L42) — Generate an integer hash of text using the specified algorithm. — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)
- `hash_fn(self)` — [`L76`](../../../../../../../raw/code/ejkernel/ejkernel/ops/utils/datacarrier.py#L76) — Compute an integer hash from the numeric/collection attributes of an object. — documented in [ejkernel-ops-utils-datacarrier](../../../../concepts/ejkernel-ops-utils-datacarrier.md)

