---
title: 'Module: src/maxtext/input_pipeline/packing/prefill_packing.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/packing/prefill_packing.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.packing.prefill_packing`/
symbols:
  BatchedPrefillProcessor._process_bucket: BatchedPrefillProcessor#_process_bucket().
  PrefillBucket.try_add: PrefillBucket#try_add().
  BatchedPrefillProcessor._process_batch_compiled: BatchedPrefillProcessor#_process_batch_compiled().
  BatchedPrefillProcessor.process: BatchedPrefillProcessor#process().
  PrefillProcessor._process_compiled: PrefillProcessor#_process_compiled().
  PrefillBucket.clear: PrefillBucket#clear().
  DecodeState: DecodeState.
  PrefillProcessor._process: PrefillProcessor#_process().
  Params: Params.
  PrefillProcessor.process: PrefillProcessor#process().
  BatchedPrefillProcessor.flush: BatchedPrefillProcessor#flush().
  BatchedPrefillProcessor._process_batch: BatchedPrefillProcessor#_process_batch().
  PrefillBucket.unallocated: PrefillBucket#unallocated().
  jetstream_is_stub: jetstream_is_stub.
  PrefillBucket.add: PrefillBucket#add().
  PrefillBucket.count: PrefillBucket#count.
  PrefillProcessor.engine: PrefillProcessor#engine.
  BatchedPrefillProcessor.engine: BatchedPrefillProcessor#engine.
  BatchedPrefillProcessor.max_batch_size: BatchedPrefillProcessor#max_batch_size.
  PrefillBucket.slots: PrefillBucket#slots.
  PrefillBucket.capacity: PrefillBucket#capacity.
  PrefillProcessor.aot_compile: PrefillProcessor#aot_compile().
  BatchedPrefillProcessor.aot_compile: BatchedPrefillProcessor#aot_compile().
  PrefillBucket.token_ids: PrefillBucket#token_ids.
  PrefillBucket.length: PrefillBucket#length.
  BatchedPrefillProcessor.jitted_process_batch: BatchedPrefillProcessor#jitted_process_batch.
  log: log.
  PrefillProcessor.process_func: PrefillProcessor#process_func.
  BatchedPrefillProcessor.process_batch_func: BatchedPrefillProcessor#process_batch_func.
  BatchedPrefillProcessor.zero_padded: BatchedPrefillProcessor#zero_padded().
  PrefillBucket.is_empty: PrefillBucket#is_empty().
  PRNGKeyType: PRNGKeyType.
  PrefillBucket: PrefillBucket#
  PrefillBucket.row_ids: PrefillBucket#row_ids.
  BatchedPrefillProcessor.buckets: BatchedPrefillProcessor#buckets.
  BatchedPrefillProcessor.auto_layout_supported: BatchedPrefillProcessor#auto_layout_supported.
  config_lib: config_lib.
  engine_api: engine_api.
  Prefix: Prefix.
  PackedPrefix: PackedPrefix.
  PrefillBucket.__init__: PrefillBucket#__init__().
  PrefillProcessor: PrefillProcessor#
  PrefillProcessor.__init__: PrefillProcessor#__init__().
  BatchedPrefillProcessor: BatchedPrefillProcessor#
  BatchedPrefillProcessor.__init__: BatchedPrefillProcessor#__init__().
  token_utils: token_utils.
  tokenizer_api: tokenizer_api.
  token_params_ns: token_params_ns.
---
# Module: [`src/maxtext/input_pipeline/packing/prefill_packing.py`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py)

## Classes
### `BatchedPrefillProcessor`
- def: [`src/maxtext/input_pipeline/packing/prefill_packing.py:183`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L183)
- doc: A wrapper around the APIs used by MaxEngine to do prefill and insert, provides prefill packing feature.
- signature: `class BatchedPrefillProcessor:`
- members:
  - `_process_batch(self, params: Params, tokens: jax.Array, slots: jax.Array, num_prompts: int, decoder_positions: jax.Array, decoder_segment_ids: jax.Array, start_pos: jax.Array, padded_length: int, true_lengths: jax.Array, decode_state: DecodeState, return_prompt_logp: bool = False)` — [`L384`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L384) — Prefill and insert a packed request. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `_process_batch_compiled(self, params: Params, padded_length: int, capacity: int, num_prompts: int, return_prompt_logp: bool)` — [`L339`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L339) — Ahead-of-time compilation wrapper of _process_batch(). — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `_process_bucket(self, model_params: Params, bucket: PrefillBucket, input_padding: int, decode_state: DecodeState, return_prompt_logp: bool = False)` — [`L264`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L264) — Process all items in a bucket. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `aot_compile(self, params: Params, input_padding: int, capacity: int, num_prompts: int, return_prompt_logp: bool = False)` — [`L198`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L198) — Ahead-of-time compile prefill processing routines. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `flush(self, model_params: Params, decode_state: DecodeState, prefill_done: Callable[[list[tuple[Any, int]], list[int], DecodeState], None], return_prompt_logp: bool = False)` — [`L246`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L246) — Process all remaining items in buckets. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `process(self, model_params: Params, decode_state: DecodeState, decode_slot: int, input_id: int, input_prompt: jax.Array, input_padding: int, capacity: int, prefill_done: Callable[[list[tuple[Any, int]], list[int], DecodeState], None], return_prompt_logp: bool = False)` — [`L205`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L205) — Process a new input. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `zero_padded(arr: list[int], padding: int)` — [`L297`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L297) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `auto_layout_supported` — [`L191`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L191) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `buckets` — [`L189`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L189) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `engine` — [`L187`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L187) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `jitted_process_batch` — [`L192`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L192) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `max_batch_size` — [`L190`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L190) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `process_batch_func` — [`L188`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L188) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
- protocol/private: `__init__`[`L186`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L186)
- uses (calls/refs, reference-scoped): [`DecodeState`](prefill_packing.md#DecodeState), [`Params`](prefill_packing.md#Params), [`unallocated`](prefill_packing.md#PrefillBucket.unallocated), [`count`](prefill_packing.md#PrefillBucket.count), [`capacity`](prefill_packing.md#PrefillBucket.capacity), [`slots`](prefill_packing.md#PrefillBucket.slots), [`token_ids`](prefill_packing.md#PrefillBucket.token_ids), [`log`](prefill_packing.md#log), [`PrefillBucket`](prefill_packing.md#PrefillBucket)

### `PrefillBucket`
- def: [`src/maxtext/input_pipeline/packing/prefill_packing.py:46`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L46) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
- doc: Manage a list of prefill requests.
- signature: `class PrefillBucket:`
- members:
  - `add(self, slot: int, row_id: int, token_ids: jax.Array)` — [`L87`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L87) — Adds a prefill prompt to bucket. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `clear(self)` — [`L63`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L63) — Clear bucket.
  - `is_empty(self)` — [`L97`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L97)
  - `try_add(self, slot: int, row_id: int, token_ids: jax.Array)` — [`L71`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L71) — Try to add a prefill prompt to bucket. — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `unallocated(self)` — [`L100`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L100) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `capacity` — [`L59`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L59) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `count` — [`L61`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L61) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `length` — [`L60`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L60) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `row_ids` — [`L53`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L53) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `slots` — [`L51`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L51) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
  - `token_ids` — [`L55`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L55) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
- protocol/private: `__init__`[`L49`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L49)
- used by: [`_process_bucket`](prefill_packing.md#BatchedPrefillProcessor._process_bucket), [`process`](prefill_packing.md#BatchedPrefillProcessor.process)

### `PrefillProcessor`
- def: [`src/maxtext/input_pipeline/packing/prefill_packing.py:104`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L104)
- doc: A wrapper around MaxEngine prefill and insert API.
- signature: `class PrefillProcessor:`
- members:
  - `_process(self, params: Params, tokens: jax.Array, slot: int, true_length: int, decode_state: DecodeState, rng: PRNGKeyType, return_prompt_logp: bool = False)` — [`L162`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L162) — Prefill and insert a request.
  - `_process_compiled(self, params: Params, padded_length: int, return_prompt_logp: bool = False)` — [`L133`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L133) — Ahead-of-time compilation wrapper of _process().
  - `aot_compile(self, params: Params, input_padding: int)` — [`L111`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L111) — Ahead-of-time compile prefill processing routines.
  - `process(self, model_params: Params, decode_state: DecodeState, decode_slot: int, input_tokens_padded: jax.Array, input_true_length: int, rng: PRNGKeyType, return_prompt_logp: bool = False)` — [`L116`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L116) — Process a new input.
  - `engine` — [`L108`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L108)
  - `process_func` — [`L109`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L109)
- protocol/private: `__init__`[`L107`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L107)
- uses (calls/refs, reference-scoped): [`DecodeState`](prefill_packing.md#DecodeState), [`Params`](prefill_packing.md#Params), [`log`](prefill_packing.md#log), [`PRNGKeyType`](prefill_packing.md#PRNGKeyType)

## Module values
- `DecodeState` — [`L37`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L37) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
- `PRNGKeyType` — [`L41`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L41)
- `PackedPrefix` — [`L39`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L39)
- `Params` — [`L40`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L40) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
- `Prefix` — [`L38`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L38)
- `config_lib` — [`L26`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L26)
- `engine_api` — [`L26`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L26)
- `jetstream_is_stub` — [`L28`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L28)
- `log` — [`L43`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L43) — documented in [maxtext-input_pipeline-packing-prefill_packing](../../../../../concepts/maxtext-input_pipeline-packing-prefill_packing.md)
- `token_params_ns` — [`L26`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L26)
- `token_utils` — [`L26`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L26)
- `tokenizer_api` — [`L26`](../../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/packing/prefill_packing.py#L26)

