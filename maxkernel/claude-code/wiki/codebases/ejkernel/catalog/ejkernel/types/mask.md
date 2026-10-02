---
title: 'Module: ejkernel/types/mask.py'
type: catalog
provenance: extracted
module: ejkernel/types/mask.py
status: fresh
symbol_base: scip-python python ejkernel 0.0.0 `ejkernel.types.mask`/
symbols:
  MaskInfo.tree_flatten: MaskInfo#tree_flatten().
  MaskInfo._q_segment_ids: MaskInfo#_q_segment_ids.
  MaskInfo._kv_segment_ids: MaskInfo#_kv_segment_ids.
  MaskInfo.get_shardings: MaskInfo#get_shardings().
  MaskInfo._attention_mask: MaskInfo#_attention_mask.
  _debug_trace: _debug_trace().
  MaskInfo.apply_sliding_window: MaskInfo#apply_sliding_window().
  MaskInfo: MaskInfo#
  MaskInfo.get_or_compute_attention_mask: MaskInfo#get_or_compute_attention_mask().
  MaskInfo.visualize: MaskInfo#visualize().
  MaskInfo.get_or_compute_qkv_cu_seqlens: MaskInfo#get_or_compute_qkv_cu_seqlens().
  MaskInfo.get_or_compute_positions: MaskInfo#get_or_compute_positions().
  MaskInfo.get_or_compute_segment_ids: MaskInfo#get_or_compute_segment_ids().
  MaskInfo.get_empty_sharding: MaskInfo#get_empty_sharding().
  MaskInfo.apply_kv_lengths: MaskInfo#apply_kv_lengths().
  MaskInfo.materialize_attention_mask: MaskInfo#materialize_attention_mask().
  MaskInfo.materialize_segment_ids: MaskInfo#materialize_segment_ids().
  MaskInfo.apply_causal: MaskInfo#apply_causal().
  MaskInfo.apply_chunked: MaskInfo#apply_chunked().
  MaskInfo.__repr__: MaskInfo#__repr__().
  _DEBUG_MODE: _DEBUG_MODE.
  MaskInfo._cu_seqlens_q: MaskInfo#_cu_seqlens_q.
  MaskInfo._cu_seqlens_kv: MaskInfo#_cu_seqlens_kv.
  MaskInfo.q_len: MaskInfo#q_len().
  MaskInfo.kv_len: MaskInfo#kv_len().
  MaskInfo.from_attention_mask: MaskInfo#from_attention_mask().
  MaskInfo.replace: MaskInfo#replace().
  MaskInfo.apply_token_type_ids: MaskInfo#apply_token_type_ids().
  MaskInfo.is_self_attention: MaskInfo#is_self_attention().
  MaskInfo.batch_size: MaskInfo#batch_size().
  mask_to_segment_ids: mask_to_segment_ids().
  MaskInfo.q_segment_ids: MaskInfo#q_segment_ids().
  MaskInfo.baked_in_masks: MaskInfo#baked_in_masks().
  qkv_masks_to_cu_seqlens: qkv_masks_to_cu_seqlens().
  MaskInfo.kv_segment_ids: MaskInfo#kv_segment_ids().
  MaskInfo.cu_seqlens_q: MaskInfo#cu_seqlens_q().
  MaskInfo.cu_seqlens_kv: MaskInfo#cu_seqlens_kv().
  MaskInfo.to_dtype: MaskInfo#to_dtype().
  MaskInfo.shape: MaskInfo#shape().
  MaskInfo.dynamic_init: MaskInfo#dynamic_init().
  MaskInfo.kv_positions: MaskInfo#kv_positions.
  attention_mask_to_qkv_cu_seqlens: attention_mask_to_qkv_cu_seqlens().
  MaskInfo.attention_mask: MaskInfo#attention_mask().
  MaskInfo.is_multi_sequence: MaskInfo#is_multi_sequence().
  MaskInfo.q_position_ids: MaskInfo#q_position_ids().
  MaskInfo.from_cu_seqlens: MaskInfo#from_cu_seqlens().
  MaskInfo.get_qkv_masks: MaskInfo#get_qkv_masks().
  qkv_cu_seqlens_to_qkv_masks: qkv_cu_seqlens_to_qkv_masks().
  MaskInfo.q_positions: MaskInfo#q_positions.
  _mask_to_segments_single: _mask_to_segments_single().
  segment_ids_to_mask: segment_ids_to_mask().
  segment_ids_to_qkv_masks: segment_ids_to_qkv_masks().
  MaskInfo.from_segments: MaskInfo#from_segments().
  MaskInfo.create_bias: MaskInfo#create_bias().
  _positions_from_segments_2d: _positions_from_segments_2d().
  _to_bool_mask: _to_bool_mask().
  qkv_cu_seqlens_to_attention_mask: qkv_cu_seqlens_to_attention_mask().
  MaskInfo.q_lens: MaskInfo#q_lens().
  MaskInfo.kv_lens: MaskInfo#kv_lens().
  MaskInfo.from_random: MaskInfo#from_random().
  MaskInfo.bias: MaskInfo#bias().
  _segment_ids_to_cu_seqlens: _segment_ids_to_cu_seqlens().
  cu_seqlens_to_mask: cu_seqlens_to_mask().
  MaskSharding: MaskSharding#
  MaskSharding.q_segment_ids: MaskSharding#q_segment_ids.
  MaskSharding.kv_segment_ids: MaskSharding#kv_segment_ids.
  _debug_trace.wrapper: _debug_trace().wrapper().
  _positions_from_segments_2d._scan_1d: _positions_from_segments_2d()._scan_1d().
  _attention_mask_to_padding_segment_ids: _attention_mask_to_padding_segment_ids().
  MaskSharding.attention_mask: MaskSharding#attention_mask.
  MaskSharding.q_positions: MaskSharding#q_positions.
  MaskSharding.kv_positions: MaskSharding#kv_positions.
  MaskInfo.sequence_axis_name: MaskInfo#sequence_axis_name.
  MaskInfo._check: MaskInfo#_check().
  set_debug_mode: set_debug_mode().
  get_debug_mode: get_debug_mode().
  MaskInfo.q_attention_mask: MaskInfo#q_attention_mask().
  mdim_t: mdim_t.
  _compress_ids_from_anchors: _compress_ids_from_anchors().
  _mask_to_start_end: _mask_to_start_end().
  MaskInfo.causal_mask_baked_in: MaskInfo#causal_mask_baked_in.
  MaskInfo.sliding_window_baked_in: MaskInfo#sliding_window_baked_in.
  MaskInfo.chunked_mask_baked_in: MaskInfo#chunked_mask_baked_in.
  MaskInfo.token_type_ids_baked_in: MaskInfo#token_type_ids_baked_in.
  MaskInfo.batch_axis_name: MaskInfo#batch_axis_name.
  MaskInfo.qheads_axis_name: MaskInfo#qheads_axis_name.
  MaskInfo._segment_block_labels: MaskInfo#_segment_block_labels().
  MaskInfo._downsample_labels: MaskInfo#_downsample_labels().
  _positions_from_segments_2d._scan_1d.step: _positions_from_segments_2d()._scan_1d().step().
  _segment_ids_to_cu_seqlens.count_segment: _segment_ids_to_cu_seqlens().count_segment().
  MaskSharding.cu_seqlens_q: MaskSharding#cu_seqlens_q.
  MaskSharding.cu_seqlens_kv: MaskSharding#cu_seqlens_kv.
  MaskInfo.kvheads_axis_name: MaskInfo#kvheads_axis_name.
  MaskInfo._slice_q: MaskInfo#_slice_q().
  MaskInfo._slice_k: MaskInfo#_slice_k().
  MaskInfo._mk_causal: MaskInfo#_mk_causal().
  MaskInfo._window_mask_single: MaskInfo#_window_mask_single().
  MaskInfo._slice_decode: MaskInfo#_slice_decode().
  MaskInfo._slice_prefill: MaskInfo#_slice_prefill().
  MaskInfo.tree_unflatten: MaskInfo#tree_unflatten().
  MaskInfo._block_classify: MaskInfo#_block_classify().
  MaskInfo._two_char: MaskInfo#_two_char().
  MaskInfo._left_label: MaskInfo#_left_label().
  MaskInfo.create_chunked_attention_mask: MaskInfo#create_chunked_attention_mask().
---
# Module: [`ejkernel/types/mask.py`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py)

## Classes
### `MaskInfo`
- def: [`ejkernel/types/mask.py:984`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L984)
- doc: Container for attention mask information with utilities for conversion and manipulation.
- signature: `class MaskInfo:`
- members:
  - `__repr__(self)` — [`L2807`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2807) — Enhanced string representation with shape information.
  - `apply_causal(self, offset: int | Int[Array, batch] = 0)` — [`L2290`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2290) — Apply causal (autoregressive) masking to the attention pattern.
  - `apply_chunked(self, chunk_size: int, offset: int = 0)` — [`L2570`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2570) — Apply chunked causal attention and ALWAYS update q/kv segment IDs to chunk IDs.
  - `apply_kv_lengths(self, kv_lengths: Int[Array, batch], *, q_len: int | None = None, end_index: Int[Array, batch] | None = None, sliding_window: int | None = None)` — [`L2211`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2211) — Mask out key/value positions beyond per-example lengths and keep a trailing query window. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `apply_sliding_window(self, sliding_window: int | tuple[int, int], *, offset: int | Int[Array, batch] = 0, mode: Literal["default", "decode", "prefill"] | None = None, index: int | Int[Array, batch] | None = None)` — [`L2348`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2348) — Apply sliding window attention to the attention pattern. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `apply_token_type_ids(self, token_type_ids: Int[Array, "batch q_len"] | tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]], *, combine: Literal["union", "intersect", "replace"] = "union", zero_policy: Literal["q", "kv", "both", "none"] = "q", update_segment_ids: bool | None = None)` — [`L2641`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2641) — Integrate token_type_ids into the attention pattern. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `attention_mask(self)` — [`L1027`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1027) — Get the 4D attention mask, computing it from segment IDs if necessary.
  - `baked_in_masks(self)` — [`L1229`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1229) — Get a dictionary of all baked-in mask operation flags.
  - `batch_size(self)` — [`L2139`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2139) — Get batch size from available data.
  - `bias(self)` — [`L1724`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1724) — Create attention bias from the mask (convenience property).
  - `create_bias(self, dtype: jnp.dtype = jnp.float32)` — [`L1739`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1739) — Create attention bias from the mask.
  - `create_chunked_attention_mask(chunk_size: int, q_len: int, kv_len: int | None = None, offset: int = 0, dtype=jnp.bool_)` — [`L2761`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2761) — Create a chunked causal attention mask (static method).
  - `cu_seqlens_kv(self)` — [`L1103`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1103) — Get cumulative key-value sequence lengths, computing them if necessary.
  - `cu_seqlens_q(self)` — [`L1084`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1084) — Get cumulative query sequence lengths, computing them if necessary.
  - `dynamic_init(cls, *, mask_info: MaskInfo | None = None, input_ids: Int[Array, "batch seqlen"] | None = None, inputs_embeds: Float[Array, "batch seqlen dim"] | None = None, attention_mask: Int[Array, "batch seqlen"] | Bool[Array, "batch seqlen"] | None = None)` — [`L3000`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3000) — Dynamically initialize a MaskInfo from various input sources.
  - `from_attention_mask(cls, attention_mask: Bool[Array, mdim_t] | Int[Array, mdim_t], q_positions: Int[Array, "batch qlen"] | None = None, kv_positions: Int[Array, "batch kvlen"] | None = None, *, mask_is_valid: bool = True, batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"), qheads_axis_name: tuple[str] | str | None = "tp", kvheads_axis_name: tuple[str] | str | None = "tp", sequence_axis_name: tuple[str] | str | None = "sp")` — [`L1459`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1459) — Create MaskInfo from an existing attention mask.
  - `from_cu_seqlens(cls, cu_seqlens_q: Int[Array, batch_plus_one], *, max_q_len: int, cu_seqlens_kv: Int[Array, batch_plus_one] | None = None, max_kv_len: int | None = None, q_positions: Int[Array, "batch qlen"] | None = None, kv_positions: Int[Array, "batch kvlen"] | None = None, batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"), qheads_axis_name: tuple[str] | str | None = "tp", kvheads_axis_name: tuple[str] | str | None = "tp", sequence_axis_name: tuple[str] | str | None = "sp")` — [`L1555`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1555) — Create a padding-style MaskInfo from cumulative sequence lengths.
  - `from_random(cls, batch_size: int, q_len: int, kv_len: int | None = None, sparsity: float = 0.5, seed: int = 0, q_positions: Int[Array, "batch qlen"] | None = None, kv_positions: Int[Array, "batch kvlen"] | None = None, batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"), qheads_axis_name: tuple[str] | str | None = "tp", kvheads_axis_name: tuple[str] | str | None = "tp", sequence_axis_name: tuple[str] | str | None = "sp")` — [`L1638`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1638) — Create MaskInfo with random attention pattern.
  - `from_segments(cls, q_segment_ids: Int[Array, "batch qlen"], kv_segment_ids: Int[Array, "batch kvlen"] | None = None, q_positions: Int[Array, "batch qlen"] | None = None, kv_positions: Int[Array, "batch kvlen"] | None = None, batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"), qheads_axis_name: tuple[str] | str | None = "tp", kvheads_axis_name: tuple[str] | str | None = "tp", sequence_axis_name: tuple[str] | str | None = "sp", is_attn_mask: bool = False)` — [`L1378`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1378) — Create MaskInfo from segment IDs.
  - `get_empty_sharding()` — [`L1764`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1764) — Create an empty MaskSharding with all specs set to None.
  - `get_or_compute_attention_mask(self, dtype: DTypeLike = jnp.bool_)` — [`L1948`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1948) — Get attention mask, using cached data when available and deriving from segment IDs otherwise. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `get_or_compute_positions(self)` — [`L1901`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1901) — Get position arrays, computing them if not already available. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `get_or_compute_qkv_cu_seqlens(self, *, out_dtype: DTypeLike = jnp.int32, max_segments: int = 128)` — [`L2031`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2031) — Derive (cu_seqlens_q, cu_seqlens_kv) from the available mask representation. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `get_or_compute_segment_ids(self, per_head: bool = False)` — [`L1977`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1977) — Get segment IDs, computing from attention mask if not available. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `get_qkv_masks(self, dtype: DTypeLike = jnp.bool_)` — [`L2005`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2005) — Get separate query mask, key-value mask, and attention mask.
  - `get_shardings(self, sequence_parallel: bool = False, *, mesh: Mesh)` — [`L1784`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1784) — Generate sharding specifications for all mask components. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `is_multi_sequence(self)` — [`L1174`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1174) — Check if the segment IDs represent multiple sequences (packed format). — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `is_self_attention(self)` — [`L2098`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2098) — Check if this represents self-attention (same query and key-value sequences). — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `kv_len(self)` — [`L2179`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2179) — Get key-value sequence length. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `kv_lens(self)` — [`L1148`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1148) — Get per-segment lengths for keys/values. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `kv_segment_ids(self)` — [`L1065`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1065) — Get key-value segment IDs, computing them from the attention mask if necessary. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `materialize_attention_mask(self, dtype: DTypeLike = jnp.bool_)` — [`L1291`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1291) — Ensure the attention mask is materialized and return a new MaskInfo. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `materialize_segment_ids(self, per_head: bool = False)` — [`L1334`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1334) — Ensure segment IDs are materialized and return a new MaskInfo. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `q_attention_mask(self)` — [`L1253`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1253) — Get a 1D query attention mask from segment IDs.
  - `q_len(self)` — [`L2161`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2161) — Get query sequence length. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `q_lens(self)` — [`L1122`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1122) — Get per-segment lengths for queries. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `q_position_ids(self)` — [`L1274`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1274) — Compute position IDs from the query segment IDs.
  - `q_segment_ids(self)` — [`L1046`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1046) — Get query segment IDs, computing them from the attention mask if necessary. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `replace(self, *, attention_mask=None, q_segment_ids=None, kv_segment_ids=None, cu_seqlens_q=None, cu_seqlens_kv=None, **kw)` — [`L2935`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2935) — Create a new MaskInfo with specified fields replaced.
  - `shape(self)` — [`L2197`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2197) — Get (batch_size, q_len, kv_len) shape tuple.
  - `to_dtype(self, dtype: DTypeLike)` — [`L2121`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2121) — Convert attention mask to specified dtype, returning a new MaskInfo.
  - `tree_flatten(self)` — [`L2833`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2833) — Flatten MaskInfo for JAX pytree registration. — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `tree_unflatten(cls, aux_data, children)` — [`L2880`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2880) — Reconstruct MaskInfo from flattened pytree representation.
  - `visualize(self, block_size: int | tuple[int, int] = 32, batch: int = 0, head: int = 0, fit_in_screen: bool = True, max_rows: int = 32, max_cols: int = 64, charset: Literal["unicode", "ascii"] = "unicode", show_segments: bool = True, return_str: bool = False)` — [`L3070`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3070) — Pretty-print the attention mask as block-aggregated ASCII/Unicode visualization.
  - `batch_axis_name` — [`L1021`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1021) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `causal_mask_baked_in` — [`L1016`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1016) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `chunked_mask_baked_in` — [`L1018`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1018) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `kv_positions` — [`L1014`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1014)
  - `kvheads_axis_name` — [`L1023`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1023) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `q_positions` — [`L1013`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1013)
  - `qheads_axis_name` — [`L1022`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1022) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `sequence_axis_name` — [`L1024`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1024) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `sliding_window_baked_in` — [`L1017`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1017) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
  - `token_type_ids_baked_in` — [`L1019`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1019) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
- protocol/private: `_attention_mask`[`L1007`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1007), `_block_classify`[`L3157`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3157), `_check`[`L1822`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1822), `_cu_seqlens_kv`[`L1011`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1011), `_cu_seqlens_q`[`L1010`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1010), `_downsample_labels`[`L3185`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3185), `_kv_segment_ids`[`L1009`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1009), `_left_label`[`L3205`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3205), `_mk_causal`[`L2337`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2337), `_q_segment_ids`[`L1008`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L1008), `_segment_block_labels`[`L3175`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3175), `_slice_decode`[`L2471`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2471), `_slice_k`[`L2278`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2278), `_slice_prefill`[`L2541`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2541), `_slice_q`[`L2263`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2263), `_two_char`[`L3198`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L3198), `_window_mask_single`[`L2448`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L2448)
- uses (calls/refs, reference-scoped): [`_debug_trace`](mask.md#_debug_trace), [`_DEBUG_MODE`](mask.md#_DEBUG_MODE), [`mask_to_segment_ids`](mask.md#mask_to_segment_ids), [`qkv_masks_to_cu_seqlens`](mask.md#qkv_masks_to_cu_seqlens), [`get_corrected_named_sharding`](../xla_utils/shardings.md#get_corrected_named_sharding), [`qkv_cu_seqlens_to_qkv_masks`](mask.md#qkv_cu_seqlens_to_qkv_masks), [`segment_ids_to_mask`](mask.md#segment_ids_to_mask), [`segment_ids_to_qkv_masks`](mask.md#segment_ids_to_qkv_masks), [`_positions_from_segments_2d`](mask.md#_positions_from_segments_2d), [`_to_bool_mask`](mask.md#_to_bool_mask), [`_segment_ids_to_cu_seqlens`](mask.md#_segment_ids_to_cu_seqlens), [`MaskSharding`](mask.md#MaskSharding), [`kv_segment_ids`](mask.md#MaskSharding.kv_segment_ids), [`q_segment_ids`](mask.md#MaskSharding.q_segment_ids), [`_attention_mask_to_padding_segment_ids`](mask.md#_attention_mask_to_padding_segment_ids), [`attention_mask`](mask.md#MaskSharding.attention_mask), [`kv_positions`](mask.md#MaskSharding.kv_positions), [`q_positions`](mask.md#MaskSharding.q_positions), [`mdim_t`](mask.md#mdim_t), [`cu_seqlens_kv`](mask.md#MaskSharding.cu_seqlens_kv), [`cu_seqlens_q`](mask.md#MaskSharding.cu_seqlens_q)
- used by: [`flash_attention`](../modules/operations/flash_attention.md#flash_attention), [`blocksparse_attention`](../modules/operations/blocksparse_attention.md#blocksparse_attention), [`ring_attention`](../modules/operations/ring_attention.md#ring_attention), [`scaled_dot_product_attention`](../modules/operations/scaled_dot_product_attention.md#scaled_dot_product_attention), [`attention`](../modules/operations/attention.md#attention)

### `MaskSharding`  ·  implements/extends NamedTuple
- def: [`ejkernel/types/mask.py:957`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L957) — documented in [ejkernel-types-mask](../../../concepts/ejkernel-types-mask.md)
- doc: Container for sharding specifications of attention mask components.
- signature: `class MaskSharding(NamedTuple):`
- members:
  - `attention_mask` — [`L974`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L974)
  - `cu_seqlens_kv` — [`L978`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L978)
  - `cu_seqlens_q` — [`L977`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L977)
  - `kv_positions` — [`L980`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L980)
  - `kv_segment_ids` — [`L976`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L976)
  - `q_positions` — [`L979`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L979)
  - `q_segment_ids` — [`L975`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L975)
- used by: [`flash_attention`](../modules/operations/flash_attention.md#flash_attention), [`blocksparse_attention`](../modules/operations/blocksparse_attention.md#blocksparse_attention), [`get_shardings`](mask.md#MaskInfo.get_shardings), [`ring_attention`](../modules/operations/ring_attention.md#ring_attention), [`get_empty_sharding`](mask.md#MaskInfo.get_empty_sharding), [`scaled_dot_product_attention`](../modules/operations/scaled_dot_product_attention.md#scaled_dot_product_attention)

## Functions
- `_attention_mask_to_padding_segment_ids(attention_mask: Bool[Array, ...])` — [`L575`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L575) — Extract Q/KV padding-style segment IDs from an attention mask.
- `_compress_ids_from_anchors(anchors: jnp.ndarray, pad_mask: jnp.ndarray)` — [`L170`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L170) — Convert anchors (minimum representative index per element) into contiguous segment IDs.
- `_debug_trace(func)` — [`L143`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L143) — Decorator that logs function calls to stdout when debug mode is enabled.
- `_mask_to_segments_single(m: jnp.ndarray)` — [`L210`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L210) — Convert a single 2D attention mask to query and key-value segment IDs.
- `_mask_to_start_end(mask: Bool[Array, "batch seq_len"], out_dtype: DTypeLike)` — [`L621`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L621) — Convert a boolean mask to start/end positions for each batch element.
- `_positions_from_segments_2d(segment_ids: jnp.ndarray, *, pad_value: int)` — [`L324`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L324) — Compute 0-based positions per segment with reset at segment boundaries.
- `_scan_1d(ids_1d)` — [`L357`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L357)
- `_segment_ids_to_cu_seqlens(segment_ids: Int[Array, "batch seq_len"], out_dtype: DTypeLike = jnp.int32, max_segments: int = 128)` — [`L694`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L694) — Convert segment IDs to FlashAttention-style cumulative sequence lengths.
- `_to_bool_mask(x: Array)` — [`L552`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L552) — Convert an array to a boolean mask.
- `attention_mask_to_qkv_cu_seqlens(attention_mask: Array, *, reduce_heads: Literal["any", "all", "first"] = "any", out_dtype: DTypeLike = jnp.int32)` — [`L885`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L885) — Derive Q/KV cumulative sequence lengths from an attention mask.
- `count_segment(seg_id)` — [`L737`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L737)
- `cu_seqlens_to_mask(cu_seqlens: Int[Array, batch * 2], max_len: int, dtype: DTypeLike = jnp.bool_)` — [`L748`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L748) — Convert start/end position pairs into a 2D mask.
- `get_debug_mode()` — [`L121`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L121) — Check if debug mode is currently enabled.
- `mask_to_segment_ids(mask: jnp.ndarray, per_head: bool = False)` — [`L259`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L259) — Convert attention mask to segment IDs (JIT-friendly).
- `qkv_cu_seqlens_to_attention_mask(cu_seqlens_q: Int[Array, batch_plus_one], *, max_q_len: int, cu_seqlens_kv: Int[Array, batch_plus_one] | None = None, max_kv_len: int | None = None, dtype: DTypeLike = jnp.bool_)` — [`L835`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L835) — Convert Q/KV cumulative sequence lengths into a broadcastable 4D attention mask.
- `qkv_cu_seqlens_to_qkv_masks(cu_seqlens_q: Int[Array, batch_plus_one], *, max_q_len: int, cu_seqlens_kv: Int[Array, batch_plus_one] | None = None, max_kv_len: int | None = None, dtype: DTypeLike = jnp.bool_)` — [`L788`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L788) — Convert Q/KV cumulative sequence lengths back into 2D padding masks.
- `qkv_masks_to_cu_seqlens(q_mask: Bool[Array, "batch q_len"] | Int[Array, "batch q_len"], kv_mask: Bool[Array, "batch kv_len"] | Int[Array, "batch kv_len"] | None = None, *, out_dtype: DTypeLike = jnp.int32)` — [`L655`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L655) — Convert per-token Q/KV masks into start/end position pairs.
- `segment_ids_to_mask(segment_ids: Int[Array, "batch seq_len"] | tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]], dtype: DTypeLike = jnp.bool_, return_separate_masks: bool = False)` — [`L376`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L376) — Converts segment IDs to an attention mask.
- `segment_ids_to_qkv_masks(q_segment_ids: Int[Array, "batch q_len"], kv_segment_ids: Int[Array, "batch kv_len"] | None = None, dtype: DTypeLike = jnp.bool_)` — [`L488`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L488) — Converts query and key-value segment IDs to separate Q mask, KV mask, and attention mask.
- `set_debug_mode(enabled: bool)` — [`L89`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L89) — Enable or disable debug tracing for MaskInfo operations.
- `step(carry, seg_i)` — [`L358`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L358)
- `wrapper(*args, **kwargs)` — [`L158`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L158)

## Module values
- `_DEBUG_MODE` — [`L86`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L86)
- `mdim_t` — [`L83`](../../../../../../raw/code/ejkernel/ejkernel/types/mask.py#L83)

