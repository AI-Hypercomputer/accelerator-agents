---
title: 'Module: simply/utils/sampling_lib.py'
type: catalog
provenance: extracted
module: simply/utils/sampling_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.sampling_lib`/
symbols:
  SamplingParams.get_decoding_schedule: SamplingParams#get_decoding_schedule().
  SamplingParams.max_decode_steps: SamplingParams#max_decode_steps.
  Chunk: Chunk#
  input_as_chunks: input_as_chunks().
  ProcessedInput: ProcessedInput#
  compute_log_likelihood: compute_log_likelihood().
  ChunkSequence: ChunkSequence.
  DecodingSchedule.get_next_length: DecodingSchedule#get_next_length().
  sample_from_logits: sample_from_logits().
  SamplingInput: SamplingInput.
  SamplingParams.temperature: SamplingParams#temperature.
  InputProcessorInterface.encode: InputProcessorInterface#encode().
  SamplingParams.top_k: SamplingParams#top_k.
  SamplingParams.top_p: SamplingParams#top_p.
  chunks_as_text: chunks_as_text().
  create_input_processor: create_input_processor().
  sample_from_logits.masked_sample_fn: sample_from_logits().masked_sample_fn().
  compute_log_likelihood.masked_sample_score_fn: compute_log_likelihood().masked_sample_score_fn().
  ProcessedInputBatch.pad_batch_to: ProcessedInputBatch#pad_batch_to().
  BasicTextInputProcessor.encode: BasicTextInputProcessor#encode().
  SamplingParams.max_seq_len: SamplingParams#max_seq_len.
  BasicTextInputProcessor.vocab: BasicTextInputProcessor#vocab.
  top_k_mask: top_k_mask().
  InputProcessorInterface.eos_ids: InputProcessorInterface#eos_ids.
  BasicTextInputProcessor: BasicTextInputProcessor#
  SamplingParams.num_samples: SamplingParams#num_samples.
  ProcessedInputBatch.repeat: ProcessedInputBatch#repeat().
  BasicTextInputProcessor.decode: BasicTextInputProcessor#decode().
  EmbeddingTextInputProcessor.encode: EmbeddingTextInputProcessor#encode().
  DecodingSchedule.prefill_size: DecodingSchedule#prefill_size.
  InputProcessorInterface.decode: InputProcessorInterface#decode().
  SamplingRegistry: SamplingRegistry#
  ProcessedInputBatch.tokens: ProcessedInputBatch#tokens.
  InputProcessorInterface.input_as_chunks: InputProcessorInterface#input_as_chunks().
  ProcessedInputBatch: ProcessedInputBatch#
  ProcessedInputBatch.from_unpadded_inputs: ProcessedInputBatch#from_unpadded_inputs().
  EmbeddingTextInputProcessor: EmbeddingTextInputProcessor#
  SamplingParams.intermediate_decode_steps: SamplingParams#intermediate_decode_steps.
  ProcessedInputBatch.lengths: ProcessedInputBatch#lengths.
  DecodingSchedule.begin_position: DecodingSchedule#begin_position.
  SamplingParams.sort_by: SamplingParams#sort_by.
  Chunk.Type: Chunk#Type#
  input_as_text: input_as_text().
  sample_from_logits.sample_fn: sample_from_logits().sample_fn().
  compute_log_likelihood.sample_score_fn: compute_log_likelihood().sample_score_fn().
  ProcessedInputBatch.pad_fn: ProcessedInputBatch#pad_fn().
  InputProcessorRegistry: InputProcessorRegistry#
  DecodingSchedule.end_position: DecodingSchedule#end_position.
  DecodingSchedule.chunk_size: DecodingSchedule#chunk_size.
  ProcessedInput.extra_inputs: ProcessedInput#extra_inputs.
  Chunk.Type.ARRAY: Chunk#Type#ARRAY.
  InputProcessorInterface: InputProcessorInterface#
  ProcessedInputBatch.extra_inputs: ProcessedInputBatch#extra_inputs.
  top_p_mask: top_p_mask().
  ProcessedInputBatch.form_batch: ProcessedInputBatch#form_batch().
  ProcessedInputBatch.batch_size: ProcessedInputBatch#batch_size().
  sample_from_logits.simple_sample_fn: sample_from_logits().simple_sample_fn().
  compute_log_likelihood.greedy_score_fn: compute_log_likelihood().greedy_score_fn().
  compute_log_likelihood.simple_sample_score_fn: compute_log_likelihood().simple_sample_score_fn().
  DecodingSchedule: DecodingSchedule#
  SamplingParams.max_input_len: SamplingParams#max_input_len.
  SamplingParams.prefill_size: SamplingParams#prefill_size.
  BasicTextInputProcessor.bos_id: BasicTextInputProcessor#bos_id.
  ProcessedInputBatch.min_length: ProcessedInputBatch#min_length().
  ProcessedInputBatch.max_length: ProcessedInputBatch#max_length().
  ProcessedInputBatch.token_slice: ProcessedInputBatch#token_slice().
  ProcessedInputBatch.pad_to: ProcessedInputBatch#pad_to().
  Chunk.content: Chunk#content.
  SamplingParams: SamplingParams#
  InputProcessorInterface.pad_id: InputProcessorInterface#pad_id.
  _fused_top_k_top_p_mask: _fused_top_k_top_p_mask().
  SamplingParams.min_prefill_size: SamplingParams#min_prefill_size.
  _scatter_mask: _scatter_mask().
  sample_from_logits.greedy_fn: sample_from_logits().greedy_fn().
  SamplingRegistry.namespace: SamplingRegistry#namespace.
  ProcessedInput.tokens: ProcessedInput#tokens.
  Chunk.Type.TEXT: Chunk#Type#TEXT.
  Chunk.role: Chunk#role.
  InputProcessorRegistry.namespace: InputProcessorRegistry#namespace.
  BasicTextInputProcessor.__init__: BasicTextInputProcessor#__init__().
---
# Module: [`simply/utils/sampling_lib.py`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py)

## Classes
### `BasicTextInputProcessor`
- def: [`simply/utils/sampling_lib.py:324`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L324)
- doc: Basic input processor for text.
- signature: `class BasicTextInputProcessor(InputProcessorInterface):`
- members:
  - `decode(self, token_ids: Sequence[int])` — [`L372`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L372)
  - `encode(self, chunks: ChunkSequence, max_input_len: int | None = None)` — [`L356`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L356) — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
  - `bos_id` — [`L352`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L352)
  - `vocab` — [`L335`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L335)
- protocol/private: `__init__`[`L327`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L327)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Chunk`](sampling_lib.md#Chunk), [`ProcessedInput`](sampling_lib.md#ProcessedInput), [`ChunkSequence`](sampling_lib.md#ChunkSequence), [`eos_ids`](sampling_lib.md#InputProcessorInterface.eos_ids), [`InputProcessorRegistry`](sampling_lib.md#InputProcessorRegistry), [`InputProcessorInterface`](sampling_lib.md#InputProcessorInterface), [`pad_id`](sampling_lib.md#InputProcessorInterface.pad_id)
- used by: [`input_processor`](../model_lib.md#LMInterface.input_processor), [`encode`](sampling_lib.md#InputProcessorInterface.encode), [`create_input_processor`](sampling_lib.md#create_input_processor), [`decode`](sampling_lib.md#InputProcessorInterface.decode), [`EmbeddingTextInputProcessor`](sampling_lib.md#EmbeddingTextInputProcessor)

### `Chunk`
- def: [`simply/utils/sampling_lib.py:252`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L252) — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
- signature: `class Chunk:`
- members:
  - `content` — [`L261`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L261)
  - `role` — [`L260`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L260)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Array`](common.md#Array.Array), [`SamplingRegistry`](sampling_lib.md#SamplingRegistry)
- used by: [`generate`](../model_lib.md#LMInterface.generate), [`input_as_chunks`](sampling_lib.md#input_as_chunks), [`ChunkSequence`](sampling_lib.md#ChunkSequence), [`get_sampling_input`](evaluation_lib.md#Evaluation.get_sampling_input), [`chunks_as_text`](sampling_lib.md#chunks_as_text), [`encode`](sampling_lib.md#BasicTextInputProcessor.encode), [`decode`](sampling_lib.md#BasicTextInputProcessor.decode), [`encode`](sampling_lib.md#EmbeddingTextInputProcessor.encode), [`test_chunk_dump_and_load`](sampling_lib_test.md#SamplingLibTest.test_chunk_dump_and_load), [`ARRAY`](sampling_lib.md#Chunk.Type.ARRAY)

### `DecodingSchedule`
- def: [`simply/utils/sampling_lib.py:43`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L43) — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
- doc: Encapsulates the indices used for decoding in chunks.
- signature: `class DecodingSchedule:`
- members:
  - `get_next_length(self, cur_position: int)` — [`L65`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L65)
  - `begin_position` — [`L61`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L61)
  - `chunk_size` — [`L63`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L63)
  - `end_position` — [`L62`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L62)
  - `prefill_size` — [`L60`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L60)
- used by: [`generate`](../model_lib.md#LMInterface.generate), [`get_decoding_schedule`](sampling_lib.md#SamplingParams.get_decoding_schedule), [`test_sampling_params`](sampling_lib_test.md#SamplingLibTest.test_sampling_params), [`test_decoding_schedule`](sampling_lib_test.md#SamplingLibTest.test_decoding_schedule)

### `EmbeddingTextInputProcessor`
- def: [`simply/utils/sampling_lib.py:381`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L381)
- doc: Input processor that allows precomputed embeddings.
- signature: `class EmbeddingTextInputProcessor(BasicTextInputProcessor):`
- members:
  - `encode(self, chunks: ChunkSequence, max_input_len: int | None = None)` — [`L384`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L384) — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Array`](common.md#Array.Array), [`Chunk`](sampling_lib.md#Chunk), [`ProcessedInput`](sampling_lib.md#ProcessedInput), [`ChunkSequence`](sampling_lib.md#ChunkSequence), [`BasicTextInputProcessor`](sampling_lib.md#BasicTextInputProcessor), [`InputProcessorRegistry`](sampling_lib.md#InputProcessorRegistry)

### `InputProcessorInterface`  ·  implements/extends Protocol
- def: [`simply/utils/sampling_lib.py:294`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L294)
- doc: Generic interface for preprocessing input.
- signature: `class InputProcessorInterface(Protocol):`
- members:
  - `decode(self, token_ids: list[int])` — [`L314`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L314)
  - `encode(self, chunks: ChunkSequence, max_input_len: int | None = None)` — [`L309`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L309) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `input_as_chunks(self, sampling_input: SamplingInput)` — [`L317`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L317)
  - `eos_ids` — [`L306`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L306)
  - `pad_id` — [`L307`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L307)
- uses (calls/refs, reference-scoped): [`input_as_chunks`](sampling_lib.md#input_as_chunks), [`ProcessedInput`](sampling_lib.md#ProcessedInput), [`ChunkSequence`](sampling_lib.md#ChunkSequence), [`SamplingInput`](sampling_lib.md#SamplingInput), [`encode`](sampling_lib.md#BasicTextInputProcessor.encode), [`decode`](sampling_lib.md#BasicTextInputProcessor.decode)
- used by: [`generate`](../model_lib.md#LMInterface.generate), [`loop`](../serving/page_batcher.md#Batcher.loop), [`sample_with_tool`](../tool_lib.md#ToolExecutor.sample_with_tool), [`score`](../model_lib.md#LMInterface.score), [`test_batch_sampling_with_max_seq_len`](../model_lib_test.md#ModelLibTest.test_batch_sampling_with_max_seq_len), [`_try_get_request`](../serving/page_batcher.md#Batcher._try_get_request), [`input_processor`](../serving/page_batcher.md#Batcher.input_processor), [`input_processor`](../serving/vanilla_server.md#Batcher.input_processor), [`init_sampling_state`](../serving/page_batcher.md#Batcher.init_sampling_state), [`_encode`](../tool_lib.md#ToolExecutor._encode), [`create_input_processor`](sampling_lib.md#create_input_processor), [`__init__`](../model_lib.md#LMInterface.__init__), [`vocab`](sampling_lib.md#BasicTextInputProcessor.vocab), [`BasicTextInputProcessor`](sampling_lib.md#BasicTextInputProcessor), [`count_num_tokens`](../model_lib.md#LMInterface.count_num_tokens), [`eos_ids`](../model_lib.md#LMInterface.eos_ids)

### `InputProcessorRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/sampling_lib.py:288`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L288)
- doc: Input processor registry.
- signature: `class InputProcessorRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L291`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L291)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`create_input_processor`](sampling_lib.md#create_input_processor), [`BasicTextInputProcessor`](sampling_lib.md#BasicTextInputProcessor), [`EmbeddingTextInputProcessor`](sampling_lib.md#EmbeddingTextInputProcessor)

### `ProcessedInput`
- def: [`simply/utils/sampling_lib.py:139`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L139)
- signature: `class ProcessedInput:`
- members:
  - `extra_inputs` — [`L141`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L141)
  - `tokens` — [`L140`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L140)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`PyTree`](common.md#PyTree.PyTree), [`SamplingRegistry`](sampling_lib.md#SamplingRegistry)
- used by: [`encode`](sampling_lib.md#InputProcessorInterface.encode), [`encode`](sampling_lib.md#BasicTextInputProcessor.encode), [`encode`](sampling_lib.md#EmbeddingTextInputProcessor.encode), [`from_unpadded_inputs`](sampling_lib.md#ProcessedInputBatch.from_unpadded_inputs), [`test_processed_input_batching`](sampling_lib_test.md#SamplingLibTest.test_processed_input_batching), [`processed_input`](../model_lib.md#SamplingOutput.processed_input)

### `ProcessedInputBatch`
- def: [`simply/utils/sampling_lib.py:146`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L146)
- doc: Holder for all sampling input after processing.
- signature: `class ProcessedInputBatch:`
- members:
  - `batch_size(self)` — [`L205`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L205)
  - `form_batch(*xs: common.Array)` — [`L182`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L182)
  - `from_unpadded_inputs(cls, unpadded_inputs: Sequence[ProcessedInput], pad_id: int = 0)` — [`L158`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L158) — Creates a padded version of an unpadded batch of ProcessedInput.
  - `max_length(self)` — [`L213`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L213)
  - `min_length(self)` — [`L209`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L209)
  - `pad_batch_to(self, batch_size: int)` — [`L236`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L236)
  - `pad_fn(x)` — [`L238`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L238)
  - `pad_to(self, length, pad_id=0)` — [`L219`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L219)
  - `repeat(self, n: int)` — [`L228`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L228)
  - `token_slice(self, start, end)` — [`L216`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L216)
  - `extra_inputs` — [`L155`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L155)
  - `lengths` — [`L154`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L154)
  - `tokens` — [`L153`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L153)
- uses (calls/refs, reference-scoped): [`PyTree`](common.md#PyTree.PyTree), [`Array`](common.md#Array.Array), [`ProcessedInput`](sampling_lib.md#ProcessedInput)
- used by: [`generate`](../model_lib.md#LMInterface.generate), [`test_processed_input_batching`](sampling_lib_test.md#SamplingLibTest.test_processed_input_batching)

### `SamplingParams`
- def: [`simply/utils/sampling_lib.py:74`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L74)
- doc: Sampling parameters.
- signature: `class SamplingParams:`
- members:
  - `get_decoding_schedule(self, min_input_length: int, max_input_length: int)` — [`L102`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L102) — Creates DecodingSchedule based on the sampling params. — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
  - `intermediate_decode_steps` — [`L99`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L99)
  - `max_decode_steps` — [`L80`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L80)
  - `max_input_len` — [`L91`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L91)
  - `max_seq_len` — [`L88`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L88) — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
  - `min_prefill_size` — [`L94`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L94)
  - `num_samples` — [`L92`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L92)
  - `prefill_size` — [`L97`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L97)
  - `sort_by` — [`L100`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L100)
  - `temperature` — [`L77`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L77)
  - `top_k` — [`L78`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L78)
  - `top_p` — [`L79`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L79)
- uses (calls/refs, reference-scoped): [`prefill_size`](sampling_lib.md#DecodingSchedule.prefill_size), [`begin_position`](sampling_lib.md#DecodingSchedule.begin_position), [`chunk_size`](sampling_lib.md#DecodingSchedule.chunk_size), [`end_position`](sampling_lib.md#DecodingSchedule.end_position), [`DecodingSchedule`](sampling_lib.md#DecodingSchedule)
- used by: [`run_experiment`](../rl_lib.md#run_experiment), [`main`](../eval/decode_eval.md#main), [`generate`](../model_lib.md#LMInterface.generate), [`sample_with_tool`](../tool_lib.md#ToolExecutor.sample_with_tool), [`test_batch_sampling_with_max_seq_len`](../model_lib_test.md#ModelLibTest.test_batch_sampling_with_max_seq_len), [`test_sampling_token_scores`](../model_lib_test.md#ModelLibTest.test_sampling_token_scores), [`test_batch_sampling`](../model_lib_test.md#ModelLibTest.test_batch_sampling), [`test_lm_interface_generate_with_local_state`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate_with_local_state), [`test_continue_decoding`](../model_lib_test.md#ModelLibTest.test_continue_decoding), [`test_lm_interface_generate_without_scoring`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate_without_scoring), [`test_sampling_max_decode_steps_equals_prefill_size`](../model_lib_test.md#ModelLibTest.test_sampling_max_decode_steps_equals_prefill_size), [`test_sampling_output_logprobs`](../model_lib_test.md#ModelLibTest.test_sampling_output_logprobs), [`lm_interface`](../serving/vanilla_server.md#Batcher.lm_interface), [`test_lm_interface_batch`](../model_lib_test.md#ModelLibTest.test_lm_interface_batch), [`test_lm_interface_generate`](../model_lib_test.md#ModelLibTest.test_lm_interface_generate), [`SamplingParams`](../model_lib.md#SamplingParams), [`test_sampling_params`](sampling_lib_test.md#SamplingLibTest.test_sampling_params), [`from_sampling_params`](../model_lib.md#ScoringParams.from_sampling_params)

### `SamplingRegistry`  ·  implements/extends RootRegistry
- def: [`simply/utils/sampling_lib.py:36`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L36)
- doc: Registry for Sampling related data structures.
- signature: `class SamplingRegistry(registry.RootRegistry):`
- members:
  - `namespace` — [`L39`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L39)
- uses (calls/refs, reference-scoped): [`RootRegistry`](registry.md#RootRegistry)
- used by: [`RootRegistry`](registry.md#RootRegistry), [`Chunk`](sampling_lib.md#Chunk), [`SamplingOutput`](../model_lib.md#SamplingOutput), [`ProcessedInput`](sampling_lib.md#ProcessedInput), [`Type`](sampling_lib.md#Chunk.Type), [`metadata`](../tool_lib.md#ToolOutput.metadata), [`extra_inputs`](sampling_lib.md#ProcessedInput.extra_inputs)

### `Type`
- def: [`simply/utils/sampling_lib.py:255`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L255)
- signature: `class Type(enum.StrEnum):`
- members:
  - `ARRAY` — [`L257`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L257) — documented in [simply-utils-sampling_lib](../../../concepts/simply-utils-sampling_lib.md)
  - `TEXT` — [`L256`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L256)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`Chunk`](sampling_lib.md#Chunk), [`SamplingRegistry`](sampling_lib.md#SamplingRegistry)

## Functions
- `_fused_top_k_top_p_mask(logits: jax.Array, top_k: int, top_p: float)` — [`L444`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L444) — Fused top-k + top-p mask without sorting the full vocabulary.
- `_scatter_mask(shape: tuple[int, ...], indices: jax.Array)` — [`L413`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L413) — Creates a boolean mask with True at the given indices along the last axis.
- `chunks_as_text(chunks: ChunkSequence)` — [`L271`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L271)
- `compute_log_likelihood(logits: jax.Array, tokens: jax.Array, temperature: float = 1, top_k: int = -1, top_p: float = 1)` — [`L540`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L540) — Computes the log likelyhood in float32. — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
- `create_input_processor(config, **kwargs)` — [`L405`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L405) — Creates input processor from config.
- `greedy_fn(logits: jax.Array)` — [`L494`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L494)
- `greedy_score_fn(logits: jax.Array, tokens: jax.Array)` — [`L550`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L550)
- `input_as_chunks(sampling_input: SamplingInput)` — [`L276`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L276) — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)
- `input_as_text(sampling_input: SamplingInput)` — [`L282`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L282)
- `masked_sample_fn(logits: jax.Array)` — [`L506`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L506)
- `masked_sample_score_fn(logits: jax.Array, tokens: jax.Array)` — [`L567`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L567)
- `sample_fn(logits: jax.Array)` — [`L524`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L524)
- `sample_from_logits(prng_key: jax.Array, logits: jax.Array, temperature: float = 1, top_k: int = -1, top_p: float = 1)` — [`L472`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L472) — Samples from the last step of the logits. — documented in [simply-utils-ragged_paged_attention](../../../concepts/simply-utils-ragged_paged_attention.md)
- `sample_score_fn(logits: jax.Array, tokens: jax.Array)` — [`L586`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L586)
- `simple_sample_fn(logits: jax.Array)` — [`L499`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L499)
- `simple_sample_score_fn(logits: jax.Array, tokens: jax.Array)` — [`L560`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L560)
- `top_k_mask(logits: jax.Array, top_k: int)` — [`L427`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L427) — Masks all but the top-k logits.
- `top_p_mask(logits: jax.Array, top_p: float)` — [`L434`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L434)

## Module values
- `ChunkSequence` — [`L264`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L264)
- `SamplingInput` — [`L268`](../../../../../../raw/code/simply/simply/utils/sampling_lib.py#L268)

