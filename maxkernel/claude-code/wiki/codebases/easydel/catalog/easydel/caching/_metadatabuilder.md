---
title: 'Module: easydel/caching/_metadatabuilder.py'
type: catalog
provenance: extracted
module: easydel/caching/_metadatabuilder.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.caching._metadatabuilder`/
symbols:
  IntVectorLike: IntVectorLike.
  AttentionMetadataBuilder.build: AttentionMetadataBuilder#build().
  AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu: AttentionMetadataBuilder#compute_paged_attention_batch_fields_cpu().
  AttentionMetadataBuilder._compute_ragged_from_batch_cpu: AttentionMetadataBuilder#_compute_ragged_from_batch_cpu().
  AttentionMetadataBuilder._ensure_cpu_array: AttentionMetadataBuilder#_ensure_cpu_array().
  AttentionMetadataBuilder.build_ragged_page_metadata: AttentionMetadataBuilder#build_ragged_page_metadata().
  AttentionMetadataBuilder.build_page_metadata: AttentionMetadataBuilder#build_page_metadata().
  AttentionMetadataBuilder.build_paged_metadata: AttentionMetadataBuilder#build_paged_metadata().
  AttentionMetadataBuilder.compute_ragged_batch_fields_cpu: AttentionMetadataBuilder#compute_ragged_batch_fields_cpu().
  PageTableLike: PageTableLike.
  AttentionMetadataBuilder.build_transformer_metadata: AttentionMetadataBuilder#build_transformer_metadata().
  IntMatrixLike: IntMatrixLike.
  ArrayLike: ArrayLike.
  BoolVectorLike: BoolVectorLike.
  SupportsStartsIndexs.starts: SupportsStartsIndexs#starts.
  SupportsStartsIndexs.indexs: SupportsStartsIndexs#indexs.
  AttentionMetadataBuilder.build_recurrent_metadata: AttentionMetadataBuilder#build_recurrent_metadata().
  AttentionMetadataBuilder._compute_slot_mapping_v2_cpu_padded: AttentionMetadataBuilder#_compute_slot_mapping_v2_cpu_padded().
  _RaggedComputed: _RaggedComputed#
  AttentionMetadataBuilder: AttentionMetadataBuilder#
  SupportsStartsIndexs: SupportsStartsIndexs#
  _HasCpuTensor: _HasCpuTensor#
  _HasDeviceTensor: _HasDeviceTensor#
  _PagedBatchComputed: _PagedBatchComputed#
  AttentionMetadataBuilder._compute_slot_mapping_v2_cpu: AttentionMetadataBuilder#_compute_slot_mapping_v2_cpu().
  AttentionMetadataBuilder.compute_padded_num_reqs: AttentionMetadataBuilder#compute_padded_num_reqs().
  JaxArray: JaxArray.
  NumpyArray: NumpyArray.
  _HasCpuTensor.get_cpu_tensor: _HasCpuTensor#get_cpu_tensor().
  _HasDeviceTensor.get_device_tensor: _HasDeviceTensor#get_device_tensor().
  AttentionMetadataBuilder._RAGGED_MECH_PREFIXES: AttentionMetadataBuilder#_RAGGED_MECH_PREFIXES.
  AttentionMetadataBuilder._RAGGED_MECH_EXACT: AttentionMetadataBuilder#_RAGGED_MECH_EXACT.
  _RaggedComputed.pages_tables: _RaggedComputed#pages_tables.
  _RaggedComputed.context_lens: _RaggedComputed#context_lens.
  _RaggedComputed.query_start_loc: _RaggedComputed#query_start_loc.
  _RaggedComputed.num_seqs: _RaggedComputed#num_seqs.
  _RaggedComputed.request_distribution: _RaggedComputed#request_distribution.
  _RaggedComputed.slot_mapping: _RaggedComputed#slot_mapping.
  _RaggedComputed.num_kv_update_slices: _RaggedComputed#num_kv_update_slices.
  _PagedBatchComputed.input_ids: _PagedBatchComputed#input_ids.
  _PagedBatchComputed.position_ids: _PagedBatchComputed#position_ids.
  _PagedBatchComputed.query_start_loc: _PagedBatchComputed#query_start_loc.
  _PagedBatchComputed.seq_lens: _PagedBatchComputed#seq_lens.
  _PagedBatchComputed.logits_indices: _PagedBatchComputed#logits_indices.
  _PagedBatchComputed.pages_tables: _PagedBatchComputed#pages_tables.
  _PagedBatchComputed.scheduled: _PagedBatchComputed#scheduled.
  _PagedBatchComputed.num_requests: _PagedBatchComputed#num_requests.
  _PagedBatchComputed.padded_num_reqs: _PagedBatchComputed#padded_num_reqs.
  _PagedBatchComputed.temperature: _PagedBatchComputed#temperature.
  _PagedBatchComputed.top_p: _PagedBatchComputed#top_p.
  _PagedBatchComputed.top_k: _PagedBatchComputed#top_k.
  _PagedBatchComputed.min_p: _PagedBatchComputed#min_p.
  _PagedBatchComputed.request_distribution: _PagedBatchComputed#request_distribution.
  _PagedBatchComputed.slot_mapping: _PagedBatchComputed#slot_mapping.
  _PagedBatchComputed.num_kv_update_slices: _PagedBatchComputed#num_kv_update_slices.
  _PagedBatchComputed.actual_num_tokens: _PagedBatchComputed#actual_num_tokens.
---
# Module: [`easydel/caching/_metadatabuilder.py`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py)

## Classes
### `AttentionMetadataBuilder`
- def: [`easydel/caching/_metadatabuilder.py:232`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L232)
- doc: Factory for runtime attention metadata across cache types.
- signature: `class AttentionMetadataBuilder:`
- members:
  - `_compute_ragged_from_batch_cpu(cls, *, scheduled_full: IntVectorLike, active_mask_full: BoolVectorLike, num_computed_tokens: IntVectorLike, page_table: PageTableLike, version: tp.Literal["v2", "v3"], page_size: int, num_slices_per_kv_cache_update_page: int | None, max_num_reqs: int | None, max_num_tokens: int | None, ragged_config: RaggedPagesCacheConfig | None)` — [`L885`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L885) — Compute ragged metadata fields from raw batch inputs on CPU.
  - `_compute_slot_mapping_v2_cpu(cls, *, num_requests: int, scheduled: np.ndarray, num_computed_tokens: np.ndarray, page_table: np.ndarray, page_size: int, max_pages_per_req: int, slices_per_page: int, max_padded_slices: int | None)` — [`L783`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L783) — Compute v2 slot_mapping and total updated pages on CPU.
  - `_compute_slot_mapping_v2_cpu_padded(cls, *, num_requests: int, scheduled: np.ndarray, num_computed_tokens: np.ndarray, page_table: np.ndarray, page_size: int, max_pages_per_req: int, slices_per_page: int, max_padded_slices: int | None, pad_value: int)` — [`L1418`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L1418) — Compute v2 slot_mapping with a fixed padded shape.
  - `_ensure_cpu_array(x: ArrayLike | _HasCpuTensor | _HasDeviceTensor | Sequence[int] | Sequence[bool] | Sequence[Sequence[int]] | None)` — [`L744`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L744) — Convert various array-like inputs to a NumPy array on CPU.
  - `build(cls, *, attention_mechanism: str | None = None, expected_cache_type: tp.Literal["auto", "transformer", "ragged", "recurrent"] = "auto", postpadded: bool = False, starts: IntVectorLike | None = None, indexs: IntVectorLike | None = None, cache_view: SupportsStartsIndexs | None = None, pages_tables: IntMatrixLike | None = None, block_tables: IntMatrixLike | None = None, context_lens: IntVectorLike | None = None, seq_lens: IntVectorLike | None = None, query_start_loc: IntVectorLike | None = None, num_seqs: int | IntVectorLike | None = None, slot_mapping: IntVectorLike | None = None, position_ids: IntVectorLike | None = None, request_distribution: IntVectorLike | None = None, num_kv_update_slices: IntVectorLike | None = None, version: tp.Literal["v2", "v3"] = "v3", page_size: int = 128, prefill_chunk_size: int = 512, num_slices_per_kv_cache_update_page: int | None = None, scheduled_full: IntVectorLike | None = None, active_mask_full: BoolVectorLike | None = None, num_computed_tokens: IntVectorLike | None = None, page_table: PageTableLike | None = None, max_num_reqs: int | None = None, max_num_tokens: int | None = None, ragged_config: RaggedPagesCacheConfig | None = None)` — [`L635`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L635) — Build the appropriate runtime metadata object.
  - `build_page_metadata(cls, *, pages_tables: IntMatrixLike | None = None, block_tables: IntMatrixLike | None = None, context_lens: IntVectorLike | None = None, seq_lens: IntVectorLike | None = None, query_start_loc: IntVectorLike | None = None, num_seqs: int | IntVectorLike | None = None, slot_mapping: IntVectorLike | None = None, position_ids: IntVectorLike | None = None, request_distribution: IntVectorLike | None = None, num_kv_update_slices: IntVectorLike | None = None, version: tp.Literal["v2", "v3"] = "v3", page_size: int = 128, prefill_chunk_size: int = 512, num_slices_per_kv_cache_update_page: int | None = None, scheduled_full: IntVectorLike | None = None, active_mask_full: BoolVectorLike | None = None, num_computed_tokens: IntVectorLike | None = None, page_table: PageTableLike | None = None, max_num_reqs: int | None = None, max_num_tokens: int | None = None, ragged_config: RaggedPagesCacheConfig | None = None)` — [`L515`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L515) — Alias of `build_ragged_page_metadata`.
  - `build_paged_metadata(cls, *, pages_tables: IntMatrixLike | None = None, block_tables: IntMatrixLike | None = None, context_lens: IntVectorLike | None = None, seq_lens: IntVectorLike | None = None, query_start_loc: IntVectorLike | None = None, num_seqs: int | IntVectorLike | None = None, slot_mapping: IntVectorLike | None = None, position_ids: IntVectorLike | None = None, request_distribution: IntVectorLike | None = None, num_kv_update_slices: IntVectorLike | None = None, version: tp.Literal["v2", "v3"] = "v3", page_size: int = 128, prefill_chunk_size: int = 512, num_slices_per_kv_cache_update_page: int | None = None, scheduled_full: IntVectorLike | None = None, active_mask_full: BoolVectorLike | None = None, num_computed_tokens: IntVectorLike | None = None, page_table: PageTableLike | None = None, max_num_reqs: int | None = None, max_num_tokens: int | None = None, ragged_config: RaggedPagesCacheConfig | None = None)` — [`L575`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L575) — Alias of `build_ragged_page_metadata`.
  - `build_ragged_page_metadata(cls, *, pages_tables: IntMatrixLike | None = None, block_tables: IntMatrixLike | None = None, context_lens: IntVectorLike | None = None, seq_lens: IntVectorLike | None = None, query_start_loc: IntVectorLike | None = None, num_seqs: int | IntVectorLike | None = None, slot_mapping: IntVectorLike | None = None, position_ids: IntVectorLike | None = None, request_distribution: IntVectorLike | None = None, num_kv_update_slices: IntVectorLike | None = None, version: tp.Literal["v2", "v3"] = "v3", page_size: int = 128, prefill_chunk_size: int = 512, num_slices_per_kv_cache_update_page: int | None = None, scheduled_full: IntVectorLike | None = None, active_mask_full: BoolVectorLike | None = None, num_computed_tokens: IntVectorLike | None = None, page_table: PageTableLike | None = None, max_num_reqs: int | None = None, max_num_tokens: int | None = None, ragged_config: RaggedPagesCacheConfig | None = None)` — [`L355`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L355) — Build ragged/paged attention runtime metadata (v2 or v3).
  - `build_recurrent_metadata(cls)` — [`L338`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L338) — Build recurrent/SSM runtime metadata.
  - `build_transformer_metadata(cls, *, postpadded: bool = False, starts: IntVectorLike | None = None, indexs: IntVectorLike | None = None, cache_view: SupportsStartsIndexs | None = None)` — [`L293`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L293) — Build transformer-style runtime metadata.
  - `compute_padded_num_reqs(*, num_requests: int, max_num_reqs: int, min_input_pad: int, padded_num_reqs_in: int | None)` — [`L1089`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L1089) — Compute a compilation-friendly padded request count.
  - `compute_paged_attention_batch_fields_cpu(cls, *, num_tokens_static: int, scheduled_full: IntVectorLike, active_mask_full: BoolVectorLike, token_ids: IntMatrixLike, num_computed_tokens: IntVectorLike, page_table: PageTableLike, padded_num_reqs_in: int | None, min_input_pad: int, version: tp.Literal["v2", "v3"] = "v3", ragged_config: RaggedPagesCacheConfig | None = None, max_num_reqs: int | None = None, max_num_tokens: int | None = None, page_size: int | None = None, max_pages_per_req: int | None = None, num_slices_per_kv_cache_update_page: int | None = None, temperature: IntVectorLike | None = None, top_p: IntVectorLike | None = None, top_k: IntVectorLike | None = None, min_p: IntVectorLike | None = None, page_table_padding_val: int = 0, slot_mapping_padding_val: int = 0)` — [`L1143`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L1143) — Compute CPU batch fields for paged/ragged attention runners.
  - `compute_ragged_batch_fields_cpu(cls, *, scheduled_full: IntVectorLike, active_mask_full: BoolVectorLike, num_computed_tokens: IntVectorLike, page_table: PageTableLike, version: tp.Literal["v2", "v3"], page_size: int = 128, num_slices_per_kv_cache_update_page: int | None = None, max_num_reqs: int | None = None, max_num_tokens: int | None = None, ragged_config: RaggedPagesCacheConfig | None = None)` — [`L1023`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L1023) — Compute ragged/paged batch fields on CPU.
- protocol/private: `_RAGGED_MECH_EXACT`[`L282`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L282), `_RAGGED_MECH_PREFIXES`[`L278`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L278)
- uses (calls/refs, reference-scoped): [`RaggedPagesMetadata`](ragged_page/cache.md#RaggedPagesMetadata), [`TransformerMetadata`](transformer/cache.md#TransformerMetadata), [`RaggedPagesCacheConfig`](ragged_page/cache.md#RaggedPagesCacheConfig), [`IntVectorLike`](_metadatabuilder.md#IntVectorLike), [`get_padded_num_slices`](ragged_page/cache.md#RaggedPagesCacheConfig.get_padded_num_slices), [`get_max_num_seqs`](ragged_page/cache.md#RaggedPagesCacheConfig.get_max_num_seqs), [`PageTableLike`](_metadatabuilder.md#PageTableLike), [`RecurrentMetadata`](recurrent/cache.md#RecurrentMetadata), [`IntMatrixLike`](_metadatabuilder.md#IntMatrixLike), [`page_size`](ragged_page/cache.md#RaggedPagesCacheConfig.page_size), [`ArrayLike`](_metadatabuilder.md#ArrayLike), [`BoolVectorLike`](_metadatabuilder.md#BoolVectorLike), [`num_slices_per_kv_cache_update_page`](ragged_page/cache.md#RaggedPagesCacheConfig.num_slices_per_kv_cache_update_page), [`indexs`](_metadatabuilder.md#SupportsStartsIndexs.indexs), [`starts`](_metadatabuilder.md#SupportsStartsIndexs.starts), [`_RaggedComputed`](_metadatabuilder.md#_RaggedComputed), [`max_num_pages_per_req`](ragged_page/cache.md#RaggedPagesCacheConfig.max_num_pages_per_req), [`SupportsStartsIndexs`](_metadatabuilder.md#SupportsStartsIndexs), [`_HasCpuTensor`](_metadatabuilder.md#_HasCpuTensor), [`_HasDeviceTensor`](_metadatabuilder.md#_HasDeviceTensor), [`_PagedBatchComputed`](_metadatabuilder.md#_PagedBatchComputed), [`get_cpu_tensor`](_metadatabuilder.md#_HasCpuTensor.get_cpu_tensor), [`get_device_tensor`](_metadatabuilder.md#_HasDeviceTensor.get_device_tensor)
- used by: [`_build_host_payload`](../inference/esurge/runners/executors/batch_preparer.md#BatchMetadataPreparer._build_host_payload)

### `SupportsStartsIndexs`  ·  implements/extends Protocol
- def: [`easydel/caching/_metadatabuilder.py:71`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L71)
- doc: Protocol for cache views that carry transformer-style starts/indexs.
- signature: `class SupportsStartsIndexs(tp.Protocol):`
- members:
  - `indexs` — [`L87`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L87)
  - `starts` — [`L86`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L86)
- uses (calls/refs, reference-scoped): [`ArrayLike`](_metadatabuilder.md#ArrayLike)
- used by: [`build`](_metadatabuilder.md#AttentionMetadataBuilder.build), [`build_transformer_metadata`](_metadatabuilder.md#AttentionMetadataBuilder.build_transformer_metadata)

### `_HasCpuTensor`  ·  implements/extends Protocol
- def: [`easydel/caching/_metadatabuilder.py:90`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L90)
- doc: Protocol for objects exposing a CPU-side tensor.
- signature: `class _HasCpuTensor(tp.Protocol):`
- members:
  - `get_cpu_tensor(self)` — [`L101`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L101) — Return the CPU-resident NumPy array.
- used by: [`_ensure_cpu_array`](_metadatabuilder.md#AttentionMetadataBuilder._ensure_cpu_array), [`PageTableLike`](_metadatabuilder.md#PageTableLike)

### `_HasDeviceTensor`  ·  implements/extends Protocol
- def: [`easydel/caching/_metadatabuilder.py:110`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L110)
- doc: Protocol for objects exposing a device-side tensor.
- signature: `class _HasDeviceTensor(tp.Protocol):`
- members:
  - `get_device_tensor(self)` — [`L121`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L121) — Return the device-resident JAX array.
- used by: [`_ensure_cpu_array`](_metadatabuilder.md#AttentionMetadataBuilder._ensure_cpu_array), [`PageTableLike`](_metadatabuilder.md#PageTableLike)

### `_PagedBatchComputed`  ·  implements/extends _TypedDict
- def: [`easydel/caching/_metadatabuilder.py:166`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L166)
- doc: CPU-computed fields for a paged/ragged attention batch.
- signature: `class _PagedBatchComputed(tp.TypedDict):`
- members:
  - `actual_num_tokens` — [`L229`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L229)
  - `input_ids` — [`L213`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L213)
  - `logits_indices` — [`L217`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L217)
  - `min_p` — [`L225`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L225)
  - `num_kv_update_slices` — [`L228`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L228)
  - `num_requests` — [`L220`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L220)
  - `padded_num_reqs` — [`L221`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L221)
  - `pages_tables` — [`L218`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L218)
  - `position_ids` — [`L214`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L214)
  - `query_start_loc` — [`L215`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L215)
  - `request_distribution` — [`L226`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L226)
  - `scheduled` — [`L219`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L219)
  - `seq_lens` — [`L216`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L216)
  - `slot_mapping` — [`L227`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L227)
  - `temperature` — [`L222`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L222)
  - `top_k` — [`L224`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L224)
  - `top_p` — [`L223`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L223)
- used by: [`compute_paged_attention_batch_fields_cpu`](_metadatabuilder.md#AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu)

### `_RaggedComputed`  ·  implements/extends _TypedDict
- def: [`easydel/caching/_metadatabuilder.py:133`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L133)
- doc: TypedDict for computed ragged/paged attention metadata fields.
- signature: `class _RaggedComputed(tp.TypedDict):`
- members:
  - `context_lens` — [`L158`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L158)
  - `num_kv_update_slices` — [`L163`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L163)
  - `num_seqs` — [`L160`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L160)
  - `pages_tables` — [`L157`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L157)
  - `query_start_loc` — [`L159`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L159)
  - `request_distribution` — [`L161`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L161)
  - `slot_mapping` — [`L162`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L162)
- used by: [`_compute_ragged_from_batch_cpu`](_metadatabuilder.md#AttentionMetadataBuilder._compute_ragged_from_batch_cpu), [`compute_ragged_batch_fields_cpu`](_metadatabuilder.md#AttentionMetadataBuilder.compute_ragged_batch_fields_cpu)

## Module values
- `ArrayLike` — [`L65`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L65)
- `BoolVectorLike` — [`L68`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L68)
- `IntMatrixLike` — [`L67`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L67)
- `IntVectorLike` — [`L66`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L66)
- `JaxArray` — [`L61`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L61)
- `NumpyArray` — [`L62`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L62)
- `PageTableLike` — [`L130`](../../../../../../raw/code/EasyDeL/easydel/caching/_metadatabuilder.py#L130)

