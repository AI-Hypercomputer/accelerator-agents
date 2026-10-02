---
title: 'Module: simply/utils/ragged_paged_attention.py'
type: catalog
provenance: extracted
module: simply/utils/ragged_paged_attention.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.ragged_paged_attention`/
symbols:
  DecodeStateConfig.init: DecodeStateConfig#init().
  DecodeState.update_decode_state_and_compute_attn: DecodeState#update_decode_state_and_compute_attn().
  SamplingState.mixed_step: SamplingState#mixed_step().
  DecodeState.release_for_window: DecodeState#release_for_window().
  DecodeState.insert: DecodeState#insert().
  DecodeState.allocate: DecodeState#allocate().
  DecodeState.__post_init__: DecodeState#__post_init__().
  SamplingState.update_with_ragged_output: SamplingState#update_with_ragged_output().
  DecodeState.release: DecodeState#release().
  SamplingState.push: SamplingState#push().
  SamplingState.issue_lens: SamplingState#issue_lens().
  SamplingState.has_ended: SamplingState#has_ended().
  DecodeState.kv_lens: DecodeState#kv_lens.
  DecodeState.num_shards: DecodeState#num_shards().
  SamplingState.ragged_issue_tokens: SamplingState#ragged_issue_tokens().
  SamplingState.desired_issue_lens: SamplingState#desired_issue_lens().
  SamplingState.get: SamplingState#get().
  DecodeState.max_num_pages_per_seq_per_shard: DecodeState#max_num_pages_per_seq_per_shard().
  SamplingState: SamplingState#
  DecodeState.page_size: DecodeState#page_size().
  SamplingState.continue_decode: SamplingState#continue_decode().
  SamplingState.tokens: SamplingState#tokens.
  DecodeStateConfig.max_num_pages_per_seq_per_shard: DecodeStateConfig#max_num_pages_per_seq_per_shard().
  DecodeState.page_indices: DecodeState#page_indices.
  DecodeState.head_partition: DecodeState#head_partition.
  DecodeState.total_num_pages_per_shard: DecodeState#total_num_pages_per_shard().
  SamplingState.position: SamplingState#position.
  SamplingState.is_continuable: SamplingState#is_continuable().
  DecodeState.pages: DecodeState#pages.
  DecodeState.window_size: DecodeState#window_size.
  DecodeState.local_num_pages: DecodeState#local_num_pages().
  DecodeState.max_available_kv_lens: DecodeState#max_available_kv_lens().
  SamplingState.create: SamplingState#create().
  SamplingState.rank_indices: SamplingState#rank_indices().
  DecodeState.kv_np: DecodeState#kv_np().
  SamplingState.is_pad_seq: SamplingState#is_pad_seq().
  autotune_block_sizes: autotune_block_sizes().
  SamplingState.release: SamplingState#release().
  SamplingState.max_seq_len: SamplingState#max_seq_len().
  DecodeState.available_page_indices: DecodeState#available_page_indices.
  DecodeState.num_available_pages: DecodeState#num_available_pages.
  DecodeState.seq_partition: DecodeState#seq_partition.
  DecodeState.available_page_indices_np: DecodeState#available_page_indices_np().
  DecodeState.page_indices_np: DecodeState#page_indices_np().
  SamplingState.input_lens: SamplingState#input_lens.
  DecodeState._sharded_rpa_fn: DecodeState#_sharded_rpa_fn().
  SamplingState.reached_eos: SamplingState#reached_eos().
  DecodeState.page_manage_key: DecodeState#page_manage_key().
  SamplingState.batch_size: SamplingState#batch_size().
  max_num_pages_per_seq_per_shard: max_num_pages_per_seq_per_shard().
  DecodeState.batch_size: DecodeState#batch_size().
  DecodeStateConfig.total_num_pages: DecodeStateConfig#total_num_pages.
  DecodeStateConfig.page_size: DecodeStateConfig#page_size.
  DecodeStateConfig.batch_size: DecodeStateConfig#batch_size.
  DecodeStateConfig.dtype: DecodeStateConfig#dtype.
  DecodeStateConfig.max_seq_len: DecodeStateConfig#max_seq_len.
  SamplingState.num_used_tokens: SamplingState#num_used_tokens().
  SamplingState.lens: SamplingState#lens().
  DecodeState: DecodeState#
  RaggedArray: RaggedArray.
  SamplingState.current_tokens: SamplingState#current_tokens().
  DecodeStateConfig.n_kv_heads: DecodeStateConfig#n_kv_heads.
  DecodeStateConfig.per_head_dim: DecodeStateConfig#per_head_dim.
  DecodeState.num_kv_heads: DecodeState#num_kv_heads().
  DecodeState.padded_per_head_dim: DecodeState#padded_per_head_dim().
  SamplingState.__post_init__: SamplingState#__post_init__().
  DecodeStateConfig: DecodeStateConfig#
  DecodeStateConfig.seq_partition: DecodeStateConfig#seq_partition.
  DecodeState.attrs_from_tree: DecodeState#attrs_from_tree().
  DecodeState.page_indices_nplist: DecodeState#page_indices_nplist().
  SamplingState.rank: SamplingState#rank.
  DecodeState.kv_nplist: DecodeState#kv_nplist().
  SamplingState.max_total_num_tokens: SamplingState#max_total_num_tokens.
  SamplingState.max_rank: SamplingState#max_rank().
  DecodeState.total_num_pages: DecodeState#total_num_pages().
  DecodeState.dtype: DecodeState#dtype().
  SamplingState.decode_state: SamplingState#decode_state.
  DecodeStateConfig.head_partition: DecodeStateConfig#head_partition.
  _StepState.state: _StepState#state.
  DecodeState.pad_per_head_dim: DecodeState#pad_per_head_dim().
  _StepState: _StepState#
  SamplingState.rank_inv_indices: SamplingState#rank_inv_indices().
  DecodeStateConfig.window_size: DecodeStateConfig#window_size.
  DecodeState.max_seq_len: DecodeState#max_seq_len.
  DecodeStateConfig.padded_per_head_dim: DecodeStateConfig#padded_per_head_dim().
  _StepState.step: _StepState#step.
  SamplingState.max_decode_steps: SamplingState#max_decode_steps.
  SamplingState.num_available_slots: SamplingState#num_available_slots().
  DecodeState.num_kv_pages_per_block: DecodeState#num_kv_pages_per_block.
  DecodeState.num_queries_per_block: DecodeState#num_queries_per_block.
  SamplingState.token_logprobs: SamplingState#token_logprobs.
  SamplingState.token_scores: SamplingState#token_scores.
  DecodeStateConfig.num_kv_pages_per_block: DecodeStateConfig#num_kv_pages_per_block.
  DecodeStateConfig.num_queries_per_block: DecodeStateConfig#num_queries_per_block.
  SamplingState.prng_key: SamplingState#prng_key.
  SamplingState.eos_ids: SamplingState#eos_ids.
---
# Module: [`simply/utils/ragged_paged_attention.py`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py)

## Classes
### `DecodeState`
- def: [`simply/utils/ragged_paged_attention.py:176`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L176)
- members:
  - `allocate(self, q_lens: jax.Array)` — [`L441`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L441) — Allocates pages for new tokens. — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `attrs_from_tree(cls, tree: common.PyTree, attr_names: Iterable[str])` — [`L262`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L262) — Returns attributes from the tree.
  - `available_page_indices_np(self)` — [`L335`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L335)
  - `batch_size(self)` — [`L278`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L278)
  - `dtype(self)` — [`L312`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L312)
  - `insert(self, k: jax.Array, v: jax.Array, q_lens: jax.Array)` — [`L510`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L510) — Inserts new kv into kv_pages at [kv_lens - q_lens, kv_lens).
  - `kv_np(self, idx: jax.typing.ArrayLike, per_head_dim: int = 0)` — [`L361`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L361) — Returns the kv for the given idx.
  - `kv_nplist(self, per_head_dim: int = 0)` — [`L379`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L379)
  - `local_num_pages(self)` — [`L316`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L316)
  - `max_available_kv_lens(self)` — [`L385`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L385) — Returns the maximum available KV lens for each sequence.
  - `max_num_pages_per_seq_per_shard(self)` — [`L282`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L282)
  - `num_kv_heads(self)` — [`L304`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L304)
  - `num_shards(self)` — [`L292`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L292)
  - `pad_per_head_dim(self, x: jax.Array)` — [`L321`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L321)
  - `padded_per_head_dim(self)` — [`L308`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L308)
  - `page_indices_np(self, idx: jax.typing.ArrayLike)` — [`L345`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L345) — Returns the page indices for the given idx.
  - `page_indices_nplist(self)` — [`L357`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L357) — Returns the page indices for each sequence as a list of numpy arrays.
  - `page_manage_key(self)` — [`L564`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L564) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `page_size(self)` — [`L300`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L300)
  - `release(self, should_release: jax.Array)` — [`L487`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L487) — Releases the decode state.
  - `release_for_window(self)` — [`L403`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L403) — Releases the decode state for local attention. — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `total_num_pages(self)` — [`L288`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L288)
  - `total_num_pages_per_shard(self)` — [`L296`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L296)
  - `update_decode_state_and_compute_attn(self, q: RaggedArray, k: jax.Array, v: jax.Array, soft_cap: float | None = None, mask_value: float | None = None, update_kv_cache: bool = True, page_manage_cache: MutableMapping[Hashable, Self] | None = None)` — [`L568`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L568) — Updates decode state. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
  - `available_page_indices` — [`L183`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L183) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `head_partition` — [`L191`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L191) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `kv_lens` — [`L186`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L186) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `max_seq_len` — [`L187`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L187)
  - `num_available_pages` — [`L184`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L184) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `num_kv_pages_per_block` — [`L197`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L197)
  - `num_queries_per_block` — [`L200`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L200)
  - `page_indices` — [`L182`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L182) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `pages` — [`L181`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L181) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `seq_partition` — [`L194`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L194) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `window_size` — [`L188`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L188)
- protocol/private: `__post_init__`[`L204`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L204), `_sharded_rpa_fn`[`L711`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L711)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`data`](common.md#RaggedArray.data), [`lens`](common.md#RaggedArray.lens), [`ragged_paged_attention`](../kernels/ragged_paged_attention.md#ragged_paged_attention), [`row_ids`](common.md#RaggedArray.row_ids), [`capacity`](common.md#RaggedArray.capacity), [`cdiv`](../kernels/ragged_paged_attention.md#cdiv), [`get_dtype_packing`](../kernels/ragged_paged_attention.md#get_dtype_packing), [`intra_offset`](common.md#RaggedArray.intra_offset), [`batch_size`](common.md#RaggedArray.batch_size), [`total_length`](common.md#RaggedArray.total_length), [`ref_ragged_paged_attention`](../kernels/ragged_paged_attention.md#ref_ragged_paged_attention), [`autotune_block_sizes`](ragged_paged_attention.md#autotune_block_sizes), [`get_partition_size`](sharding.md#get_partition_size), [`max_num_pages_per_seq_per_shard`](ragged_paged_attention.md#max_num_pages_per_seq_per_shard), [`RaggedArray`](ragged_paged_attention.md#RaggedArray), [`row_starts_with_end`](common.md#RaggedArray.row_starts_with_end)
- used by: [`apply`](../model_lib.md#Attention.apply), [`init`](ragged_paged_attention.md#DecodeStateConfig.init), [`test_update_decode_state_and_compute_attn`](ragged_paged_attention_test.md#DecodeStateTest.test_update_decode_state_and_compute_attn), [`test_release_for_window`](ragged_paged_attention_test.md#DecodeStateTest.test_release_for_window), [`test_allocate`](ragged_paged_attention_test.md#DecodeStateTest.test_allocate), [`test_insert`](ragged_paged_attention_test.md#DecodeStateTest.test_insert), [`desired_issue_lens`](ragged_paged_attention.md#SamplingState.desired_issue_lens), [`create`](ragged_paged_attention.md#SamplingState.create), [`release`](ragged_paged_attention.md#SamplingState.release)

### `DecodeStateConfig`
- def: [`simply/utils/ragged_paged_attention.py:113`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L113)
- doc: Paged KV cache config.
- signature: `class DecodeStateConfig:`
- members:
  - `init(self)` — [`L140`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L140)
  - `max_num_pages_per_seq_per_shard(self)` — [`L134`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L134)
  - `padded_per_head_dim(self)` — [`L130`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L130)
  - `batch_size` — [`L120`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L120)
  - `dtype` — [`L121`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L121)
  - `head_partition` — [`L124`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L124)
  - `max_seq_len` — [`L122`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L122)
  - `n_kv_heads` — [`L118`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L118)
  - `num_kv_pages_per_block` — [`L126`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L126)
  - `num_queries_per_block` — [`L127`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L127)
  - `page_size` — [`L117`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L117)
  - `per_head_dim` — [`L119`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L119)
  - `seq_partition` — [`L125`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L125)
  - `total_num_pages` — [`L116`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L116)
  - `window_size` — [`L123`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L123)
- uses (calls/refs, reference-scoped): [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`kv_lens`](ragged_paged_attention.md#DecodeState.kv_lens), [`get_dtype_packing`](../kernels/ragged_paged_attention.md#get_dtype_packing), [`head_partition`](ragged_paged_attention.md#DecodeState.head_partition), [`page_indices`](ragged_paged_attention.md#DecodeState.page_indices), [`pages`](ragged_paged_attention.md#DecodeState.pages), [`window_size`](ragged_paged_attention.md#DecodeState.window_size), [`available_page_indices`](ragged_paged_attention.md#DecodeState.available_page_indices), [`num_available_pages`](ragged_paged_attention.md#DecodeState.num_available_pages), [`seq_partition`](ragged_paged_attention.md#DecodeState.seq_partition), [`get_partition_size`](sharding.md#get_partition_size), [`max_num_pages_per_seq_per_shard`](ragged_paged_attention.md#max_num_pages_per_seq_per_shard), [`DecodeState`](ragged_paged_attention.md#DecodeState), [`max_seq_len`](ragged_paged_attention.md#DecodeState.max_seq_len), [`num_kv_pages_per_block`](ragged_paged_attention.md#DecodeState.num_kv_pages_per_block), [`num_queries_per_block`](ragged_paged_attention.md#DecodeState.num_queries_per_block)
- used by: [`test_continue_decode`](ragged_paged_attention_test.md#SamplingStateTest.test_continue_decode), [`init_decode_state`](../model_lib.md#Attention.init_decode_state), [`test_ragged_issue_tokens`](ragged_paged_attention_test.md#SamplingStateTest.test_ragged_issue_tokens), [`test_update_decode_state_and_compute_attn`](ragged_paged_attention_test.md#DecodeStateTest.test_update_decode_state_and_compute_attn), [`test_push_and_release`](ragged_paged_attention_test.md#SamplingStateTest.test_push_and_release), [`test_release_for_window`](ragged_paged_attention_test.md#DecodeStateTest.test_release_for_window), [`test_allocate`](ragged_paged_attention_test.md#DecodeStateTest.test_allocate)

### `SamplingState`
- def: [`simply/utils/ragged_paged_attention.py:865`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L865) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
- members:
  - `batch_size(self)` — [`L925`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L925)
  - `continue_decode(self, forward_fn: Callable[..., tuple[jax.Array, common.PyTree]], until_fn: Callable[[Self], jax.Array], params: common.PyTree, extra_inputs: common.PyTree = None, max_num_issue_tokens: int = 1024, temperature: float = 1, top_k: int = -1, top_p: float = 1, scoring_temperature: float = 1, scoring_top_k: int = -1, scoring_top_p: float = 1, intermediate_steps: int = np.iinfo(np.int32).max // 2)` — [`L1270`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1270) — Continues decoding. — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `create(cls, max_total_num_tokens: int, eos_ids: jax.typing.ArrayLike, prng_key: jax.typing.ArrayLike, decode_state: common.PyTree)` — [`L893`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L893) — Creates a sampling state.
  - `current_tokens(self)` — [`L1154`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1154)
  - `desired_issue_lens(self)` — [`L938`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L938)
  - `get(self, mask: jax.typing.ArrayLike)` — [`L1023`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1023) — Returns the tokens, logprobs, and scores for the given mask.
  - `has_ended(self)` — [`L1173`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1173) — Returns whether each sequence in the batch is done with generation.
  - `is_continuable(self)` — [`L1183`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1183)
  - `is_pad_seq(self)` — [`L933`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L933) — This sequence is a padding sequence, in [batch, 1].
  - `issue_lens(self, capacity: int)` — [`L1067`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1067) — Returns the issue lens. — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `lens(self)` — [`L1169`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1169)
  - `max_rank(self)` — [`L973`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L973)
  - `max_seq_len(self)` — [`L929`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L929)
  - `mixed_step(self, forward_fn: Callable[..., jax.Array], params: common.PyTree, extra_inputs: common.PyTree = None, max_num_issue_tokens: int = 128, temperature: float = 1, top_k: int = -1, top_p: float = 1, scoring_temperature: float = 1, scoring_top_k: int = -1, scoring_top_p: float = 1)` — [`L1190`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1190) — Executes a mixed step (prefill+decode). — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `num_available_slots(self)` — [`L981`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L981) — Returns the number of available slots.
  - `num_used_tokens(self)` — [`L1061`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1061) — Returns the number of used tokens.
  - `push(self, input_tokens: jax.typing.ArrayLike, n: int, max_decode_steps: int)` — [`L985`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L985) — Pushes new input tokens.
  - `ragged_issue_tokens(self, capacity: int)` — [`L1100`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1100) — Returns the ragged issue tokens.
  - `rank_indices(self)` — [`L960`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L960)
  - `rank_inv_indices(self)` — [`L977`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L977)
  - `reached_eos(self)` — [`L1158`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1158) — This position is output and eos, in [batch].
  - `release(self, should_release: jax.Array)` — [`L1045`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1045) — Pops and releases the sampling state.
  - `update_with_ragged_output(self, ragged_output_tokens: common.RaggedArray, **kwargs: jax.Array)` — [`L1113`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L1113) — Updates the sampling state with the ragged output tokens.
  - `decode_state` — [`L869`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L869)
  - `eos_ids` — [`L878`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L878)
  - `input_lens` — [`L874`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L874) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `max_decode_steps` — [`L875`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L875)
  - `max_total_num_tokens` — [`L879`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L879)
  - `position` — [`L873`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L873) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
  - `prng_key` — [`L868`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L868)
  - `rank` — [`L876`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L876)
  - `token_logprobs` — [`L871`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L871)
  - `token_scores` — [`L872`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L872)
  - `tokens` — [`L870`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L870) — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
- protocol/private: `__post_init__`[`L881`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L881)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`data`](common.md#RaggedArray.data), [`lens`](common.md#RaggedArray.lens), [`row_ids`](common.md#RaggedArray.row_ids), [`capacity`](common.md#RaggedArray.capacity), [`RaggedArray`](common.md#RaggedArray), [`intra_offset`](common.md#RaggedArray.intra_offset), [`compute_log_likelihood`](sampling_lib.md#compute_log_likelihood), [`sample_from_logits`](sampling_lib.md#sample_from_logits), [`batch_size`](common.md#RaggedArray.batch_size), [`total_length`](common.md#RaggedArray.total_length), [`DecodeState`](ragged_paged_attention.md#DecodeState), [`RaggedArray`](ragged_paged_attention.md#RaggedArray), [`attrs_from_tree`](ragged_paged_attention.md#DecodeState.attrs_from_tree), [`reduce_same`](common.md#reduce_same), [`state`](ragged_paged_attention.md#_StepState.state), [`_StepState`](ragged_paged_attention.md#_StepState), [`step`](ragged_paged_attention.md#_StepState.step)
- used by: [`loop`](../serving/page_batcher.md#Batcher.loop), [`test_continue_decode`](ragged_paged_attention_test.md#SamplingStateTest.test_continue_decode), [`test_ragged_issue_tokens`](ragged_paged_attention_test.md#SamplingStateTest.test_ragged_issue_tokens), [`test_push_and_release`](ragged_paged_attention_test.md#SamplingStateTest.test_push_and_release), [`decode_fn`](../serving/page_batcher.md#Batcher.decode_fn), [`sampling_state`](../serving/page_batcher.md#Batcher.sampling_state), [`compiled_decode_fn`](../serving/page_batcher.md#Batcher.compiled_decode_fn), [`init_sampling_state`](../serving/page_batcher.md#Batcher.init_sampling_state), [`compiled_push_fn`](../serving/page_batcher.md#Batcher.compiled_push_fn), [`compiled_release_fn`](../serving/page_batcher.md#Batcher.compiled_release_fn), [`_StepState`](ragged_paged_attention.md#_StepState)

### `_StepState`
- def: [`simply/utils/ragged_paged_attention.py:858`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L858)
- signature: `class _StepState:`
- members:
  - `state` — [`L860`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L860)
  - `step` — [`L859`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L859)
- uses (calls/refs, reference-scoped): [`SamplingState`](ragged_paged_attention.md#SamplingState)
- used by: [`continue_decode`](ragged_paged_attention.md#SamplingState.continue_decode)

## Functions
- `autotune_block_sizes(*, num_kv_heads: int, num_q_heads: int, page_size: int, max_seq_len: int, per_head_dim: int, window_size: int | None, dtype: jax.typing.DTypeLike, max_num_issue_tokens: int = np.iinfo(np.int32).max, num_shards: int = 1)` — [`L38`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L38) — Autotunes block sizes for ragged paged attention. — documented in [simply-kernels-ragged_paged_attention](../../../concepts/simply-kernels-ragged_paged_attention.md)
- `max_num_pages_per_seq_per_shard(max_seq_len: int, page_size: int, window_size: int | None, num_shards: int = 1)` — [`L94`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L94) — Returns the maximum number of pages per sequence per shard.

## Module values
- `RaggedArray` — [`L35`](../../../../../../raw/code/simply/simply/utils/ragged_paged_attention.py#L35)

