---
title: 'Module: tpu_inference/kernels/mla/v2/tuned_params.py'
type: catalog
provenance: extracted
module: tpu_inference/kernels/mla/v2/tuned_params.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.kernels.mla.v2.tuned_params`/
symbols:
  tuned_params_mapping.tuned_params_mapping: tuned_params_mapping.tuned_params_mapping.
  TuningKey: TuningKey#
  TunableParams: TunableParams#
  TunableParams.decode_batch_size: TunableParams#decode_batch_size.
  TunableParams.num_kv_pages_per_block: TunableParams#num_kv_pages_per_block.
  TunableParams.num_queries_per_block: TunableParams#num_queries_per_block.
  get_tuned_params: get_tuned_params().
  TuningKey.case: TuningKey#case.
  TuningKey.max_num_tokens: TuningKey#max_num_tokens.
  TuningKey.actual_num_q_heads: TuningKey#actual_num_q_heads.
  TuningKey.actual_lkv_dim: TuningKey#actual_lkv_dim.
  TuningKey.actual_r_dim: TuningKey#actual_r_dim.
  TunableParams.vmem_limit_bytes: TunableParams#vmem_limit_bytes.
  TuningKey.total_num_pages: TuningKey#total_num_pages.
  TuningKey.page_size_per_kv_packing: TuningKey#page_size_per_kv_packing.
  TuningKey.kv_packing: TuningKey#kv_packing.
  TuningKey.max_num_seqs: TuningKey#max_num_seqs.
  TuningKey.pages_per_seq: TuningKey#pages_per_seq.
  logger: logger.
  TuningKey.kv_dtype: TuningKey#kv_dtype.
  TuningKey.q_dtype: TuningKey#q_dtype.
  TuningKey.s_dtype: TuningKey#s_dtype.
  TuningKey.soft_cap: TuningKey#soft_cap.
  TuningKey.chunk_prefill_size: TuningKey#chunk_prefill_size.
  TuningKey.sliding_window: TuningKey#sliding_window.
  TuningKey.p_same_dtype_as_v: TuningKey#p_same_dtype_as_v.
---
# Module: [`tpu_inference/kernels/mla/v2/tuned_params.py`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py)

## Classes
### `TunableParams`
- def: [`tpu_inference/kernels/mla/v2/tuned_params.py:48`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L48)
- signature: `class TunableParams:`
- members:
  - `decode_batch_size` — [`L49`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L49)
  - `num_kv_pages_per_block` — [`L52`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L52)
  - `num_queries_per_block` — [`L54`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L54)
  - `vmem_limit_bytes` — [`L55`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L55)
- used by: [`_mla_ragged_paged_attention`](../../../layers/common/attention_interface.md#mla_attention._mla_ragged_paged_attention), [`tuned_params_mapping`](tuned_params.md#tuned_params_mapping.tuned_params_mapping), [`get_tuned_params`](tuned_params.md#get_tuned_params)

### `TuningKey`
- def: [`tpu_inference/kernels/mla/v2/tuned_params.py:23`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L23)
- signature: `class TuningKey:`
- members:
  - `actual_lkv_dim` — [`L27`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L27)
  - `actual_num_q_heads` — [`L26`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L26)
  - `actual_r_dim` — [`L28`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L28)
  - `case` — [`L24`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L24)
  - `chunk_prefill_size` — [`L42`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L42)
  - `kv_dtype` — [`L29`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L29)
  - `kv_packing` — [`L33`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L33)
  - `max_num_seqs` — [`L34`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L34)
  - `max_num_tokens` — [`L25`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L25)
  - `p_same_dtype_as_v` — [`L44`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L44)
  - `page_size_per_kv_packing` — [`L32`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L32)
  - `pages_per_seq` — [`L35`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L35)
  - `q_dtype` — [`L30`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L30)
  - `s_dtype` — [`L37`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L37)
  - `sliding_window` — [`L43`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L43)
  - `soft_cap` — [`L38`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L38)
  - `total_num_pages` — [`L31`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L31)
- used by: [`_mla_ragged_paged_attention`](../../../layers/common/attention_interface.md#mla_attention._mla_ragged_paged_attention), [`tuned_params_mapping`](tuned_params.md#tuned_params_mapping.tuned_params_mapping), [`get_tuned_params`](tuned_params.md#get_tuned_params)

## Functions
- `get_tuned_params(tuning_key: TuningKey)` — [`L379`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L379)

## Module values
- `logger` — [`L19`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L19)
- `tuned_params_mapping` — [`L59`](../../../../../../../../raw/code/tpu-inference/tpu_inference/kernels/mla/v2/tuned_params.py#L59)

