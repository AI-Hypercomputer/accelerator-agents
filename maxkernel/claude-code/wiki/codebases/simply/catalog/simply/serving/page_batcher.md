---
title: 'Module: simply/serving/page_batcher.py'
type: catalog
provenance: extracted
module: simply/serving/page_batcher.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.serving.page_batcher`/
symbols:
  Batcher.loop: Batcher#loop().
  Batcher.decode_fn: Batcher#decode_fn().
  Batcher._try_get_request: Batcher#_try_get_request().
  Batcher.input_processor: Batcher#input_processor().
  Batcher.sampling_state: Batcher#sampling_state().
  Batcher.update_params_from_checkpoint_path: Batcher#update_params_from_checkpoint_path().
  Batcher.compiled_decode_fn: Batcher#compiled_decode_fn().
  Batcher.init_sampling_state: Batcher#init_sampling_state().
  Batcher.config: Batcher#config.
  Batcher.compiled_push_fn: Batcher#compiled_push_fn().
  Batcher.compiled_release_fn: Batcher#compiled_release_fn().
  Batcher.lm_format: Batcher#lm_format.
  Batcher.state: Batcher#state.
  Batcher.abstract_sampling_state: Batcher#abstract_sampling_state().
  Batcher.set_mesh: Batcher#set_mesh().
  Batcher.abstract_model_state: Batcher#abstract_model_state().
  Batcher.enqueue: Batcher#enqueue().
  PyTree: PyTree.
  SimplyServiceResponse: SimplyServiceResponse.
  Batcher.model: Batcher#model().
  Batcher.request_queue: Batcher#request_queue().
  Batcher._init_fn: Batcher#_init_fn().
  Batcher.update_params: Batcher#update_params().
  Batcher.thread: Batcher#thread().
  Batcher.max_seq_len: Batcher#max_seq_len.
  Batcher.max_decode_steps: Batcher#max_decode_steps.
  Batcher._maybe_pause: Batcher#_maybe_pause().
  Batcher._batcher_loop: Batcher#_batcher_loop().
  Batcher: Batcher#
  Batcher.max_queue_timeout: Batcher#max_queue_timeout.
  Batcher.temperature: Batcher#temperature.
  Batcher.top_k: Batcher#top_k.
  Batcher.top_p: Batcher#top_p.
  Batcher.intermediate_steps: Batcher#intermediate_steps.
  Batcher.response_asap: Batcher#response_asap.
  Batcher.max_queue_size: Batcher#max_queue_size.
  Batcher.page_size: Batcher#page_size.
---
# Module: [`simply/serving/page_batcher.py`](../../../../../../raw/code/simply/simply/serving/page_batcher.py)

## Classes
### `Batcher`
- def: [`simply/serving/page_batcher.py:52`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L52)
- doc: The batcher.
- signature: `class Batcher:`
- members:
  - `_maybe_pause(self, pause_event: threading.Event, paused_event: threading.Event, resume_event: threading.Event)` — [`L310`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L310) — Pauses the batcher loop if pause_event is set, resuming across hosts.
  - `_try_get_request(self, max_seq_len: int, timeout: float)` — [`L145`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L145) — Tries to get a request from the queue. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `abstract_model_state(self)` — [`L87`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L87) — Returns the abstract model state.
  - `abstract_sampling_state(self)` — [`L224`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L224)
  - `compiled_decode_fn(self)` — [`L237`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L237) — Compiled decode function. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `compiled_push_fn(self)` — [`L260`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L260) — Compiled push function. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `compiled_release_fn(self)` — [`L288`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L288) — Compiled release function. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `decode_fn(self, sampling_state: rpa.SamplingState, params: PyTree)` — [`L183`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L183) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `enqueue(self, request: Any, future: asyncio.Future[SimplyServiceResponse])` — [`L138`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L138)
  - `init_sampling_state(self, prng_key: jax.Array)` — [`L202`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L202) — Initializes sampling state. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `input_processor(self)` — [`L76`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L76) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `loop(self, stop_event: threading.Event, pause_event: threading.Event | None = None, paused_event: threading.Event | None = None, resume_event: threading.Event | None = None)` — [`L325`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L325) — The batcher loop. — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `model(self)` — [`L72`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L72)
  - `request_queue(self)` — [`L131`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L131)
  - `sampling_state(self)` — [`L104`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L104) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `set_mesh(self)` — [`L229`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L229) — Sets the mesh for the current process.
  - `thread(self, stop_event: threading.Event, error_message_queue: queue.Queue[Exception], pause_event: threading.Event | None = None, paused_event: threading.Event | None = None, resume_event: threading.Event | None = None)` — [`L481`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L481) — Starts the batcher thread.
  - `update_params(self, params: PyTree)` — [`L110`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L110)
  - `update_params_from_checkpoint_path(self, ckpt_path: str)` — [`L113`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L113) — Updates the model params from a checkpoint path.
  - `config` — [`L55`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L55) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `intermediate_steps` — [`L68`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L68)
  - `lm_format` — [`L56`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L56)
  - `max_decode_steps` — [`L63`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L63)
  - `max_queue_size` — [`L59`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L59)
  - `max_queue_timeout` — [`L60`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L60)
  - `max_seq_len` — [`L62`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L62)
  - `page_size` — [`L64`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L64)
  - `response_asap` — [`L69`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L69)
  - `state` — [`L57`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L57) — documented in [simply-serving-page_batcher](../../../concepts/simply-serving-page_batcher.md)
  - `temperature` — [`L65`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L65)
  - `top_k` — [`L66`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L66)
  - `top_p` — [`L67`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L67)
- protocol/private: `_batcher_loop`[`L491`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L491), `_init_fn`[`L90`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L90)
- uses (calls/refs, reference-scoped): [`load_checkpoint_from_path`](../utils/checkpoint_lib.md#load_checkpoint_from_path), [`get`](../utils/registry.md#RootRegistry.get), [`TransformerLM`](../model_lib.md#TransformerLM), [`get_instance`](../utils/registry.md#RootRegistry.get_instance), [`push`](../utils/ragged_paged_attention.md#SamplingState.push), [`get_raw_arrays`](../utils/common.md#get_raw_arrays), [`has_ended`](../utils/ragged_paged_attention.md#SamplingState.has_ended), [`issue_lens`](../utils/ragged_paged_attention.md#SamplingState.issue_lens), [`tree_is_sequence`](../utils/pytree.md#tree_is_sequence), [`BaseExperimentConfig`](../config_lib.md#BaseExperimentConfig), [`desired_issue_lens`](../utils/ragged_paged_attention.md#SamplingState.desired_issue_lens), [`get`](../utils/ragged_paged_attention.md#SamplingState.get), [`input_as_chunks`](../utils/sampling_lib.md#input_as_chunks), [`SamplingState`](../utils/ragged_paged_attention.md#SamplingState), [`continue_decode`](../utils/ragged_paged_attention.md#SamplingState.continue_decode), [`position`](../utils/ragged_paged_attention.md#SamplingState.position), [`format`](../utils/lm_format.md#LMFormat.format), [`CheckpointFormatRegistry`](../utils/checkpoint_lib.md#CheckpointFormatRegistry), [`encode`](../utils/sampling_lib.md#InputProcessorInterface.encode), [`chunks_as_text`](../utils/sampling_lib.md#chunks_as_text), [`create`](../utils/ragged_paged_attention.md#SamplingState.create), [`create_input_processor`](../utils/sampling_lib.md#create_input_processor), [`TokenizerRegistry`](../utils/tokenization.md#TokenizerRegistry), [`is_pad_seq`](../utils/ragged_paged_attention.md#SamplingState.is_pad_seq), [`set_mesh`](../utils/sharding.md#set_mesh), [`release`](../utils/ragged_paged_attention.md#SamplingState.release), [`eval_abstract_output`](../utils/common.md#eval_abstract_output), [`max_seq_len`](../utils/ragged_paged_attention.md#SamplingState.max_seq_len), [`input_lens`](../utils/ragged_paged_attention.md#SamplingState.input_lens), [`eos_ids`](../utils/sampling_lib.md#InputProcessorInterface.eos_ids), [`sum_across_hosts`](../utils/sharding.md#sum_across_hosts), [`batch_size`](../utils/ragged_paged_attention.md#SamplingState.batch_size), [`LMFormat`](../utils/lm_format.md#LMFormat), [`decode`](../utils/sampling_lib.md#InputProcessorInterface.decode), [`num_used_tokens`](../utils/ragged_paged_attention.md#SamplingState.num_used_tokens), [`PyTree`](page_batcher.md#PyTree), [`SimplyServiceResponse`](page_batcher.md#SimplyServiceResponse), [`is_primary_task`](../utils/experiment_helper.md#is_primary_task), [`rank`](../utils/ragged_paged_attention.md#SamplingState.rank), [`max_total_num_tokens`](../utils/ragged_paged_attention.md#SamplingState.max_total_num_tokens)  (+7 more)
- used by: [`main`](../eval/page_decode_eval.md#main), [`main`](page_server.md#main), [`Run`](page_server.md#SimplyService.Run), [`query_and_evaluate`](../eval/page_decode_eval.md#main.query_and_evaluate), [`batcher_thread`](page_server.md#SimplyService.batcher_thread), [`batcher`](page_server.md#SimplyService.batcher)

## Module values
- `PyTree` — [`L45`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L45)
- `SimplyServiceResponse` — [`L48`](../../../../../../raw/code/simply/simply/serving/page_batcher.py#L48)

