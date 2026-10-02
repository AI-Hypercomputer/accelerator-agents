---
title: 'Module: simply/serving/vanilla_server.py'
type: catalog
provenance: extracted
module: simply/serving/vanilla_server.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.serving.vanilla_server`/
symbols:
  main: main().
  Batcher.loop: Batcher#loop().
  Batcher.lm_interface: Batcher#lm_interface().
  Batcher.input_processor: Batcher#input_processor().
  SimplyService.Run: SimplyService#Run().
  SimplyService._batcher_loop: SimplyService#_batcher_loop().
  Batcher.enqueue: Batcher#enqueue().
  main._init_fn: main()._init_fn().
  Batcher.config: Batcher#config.
  Batcher.lm_format: Batcher#lm_format.
  SimplyService.batcher: SimplyService#batcher.
  SimplyServiceResponse: SimplyServiceResponse#
  Batcher.model: Batcher#model().
  Batcher.queue: Batcher#queue().
  Batcher.update_params: Batcher#update_params().
  PyTree: PyTree.
  Batcher.model_state: Batcher#model_state.
  SimplyService.batcher_thread: SimplyService#batcher_thread().
  SimplyServiceResponse.code: SimplyServiceResponse#code.
  _BATCH_SIZE: _BATCH_SIZE.
  SimplyServiceResponse.details: SimplyServiceResponse#details.
  SimplyService.stop_event: SimplyService#stop_event().
  SimplyService.error_message_queue: SimplyService#error_message_queue().
  _ACTIVATION_DTYPE: _ACTIVATION_DTYPE.
  SimplyServiceResponse.result: SimplyServiceResponse#result.
  Batcher: Batcher#
  Batcher.batch_size: Batcher#batch_size.
  Batcher.max_queue_timeout: Batcher#max_queue_timeout.
  _SIMPLY_PORT: _SIMPLY_PORT.
  _EXPERIMENT_CONFIG: _EXPERIMENT_CONFIG.
  _MESH_SHAPE: _MESH_SHAPE.
  _CKPT_DIR: _CKPT_DIR.
  _CKPT_STEP: _CKPT_STEP.
  _CKPT_FORMAT: _CKPT_FORMAT.
  _VOCAB_NAME: _VOCAB_NAME.
  _MAX_SEQ_LEN: _MAX_SEQ_LEN.
  _MAX_DECODE_STEPS: _MAX_DECODE_STEPS.
  _INTERMEDIATE_DECODE_STEPS: _INTERMEDIATE_DECODE_STEPS.
  _LM_FORMAT: _LM_FORMAT.
  Batcher.max_queue_size: Batcher#max_queue_size.
  SimplyService: SimplyService#
---
# Module: [`simply/serving/vanilla_server.py`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py)

## Classes
### `Batcher`
- def: [`simply/serving/vanilla_server.py:145`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L145)
- doc: The batcher.
- signature: `class Batcher:`
- members:
  - `enqueue(self, request: Any, future: asyncio.Future[SimplyServiceResponse])` — [`L195`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L195)
  - `input_processor(self)` — [`L161`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L161) — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)
  - `lm_interface(self)` — [`L175`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L175) — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)
  - `loop(self, stop_event: threading.Event)` — [`L202`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L202) — The batcher loop. — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)
  - `model(self)` — [`L157`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L157)
  - `queue(self)` — [`L188`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L188)
  - `update_params(self, params: PyTree)` — [`L171`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L171)
  - `batch_size` — [`L152`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L152)
  - `config` — [`L148`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L148)
  - `lm_format` — [`L149`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L149)
  - `max_queue_size` — [`L153`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L153)
  - `max_queue_timeout` — [`L154`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L154)
  - `model_state` — [`L150`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L150)
- uses (calls/refs, reference-scoped): [`generate`](../model_lib.md#LMInterface.generate), [`TransformerLM`](../model_lib.md#TransformerLM), [`get_instance`](../utils/registry.md#RootRegistry.get_instance), [`tree_is_sequence`](../utils/pytree.md#tree_is_sequence), [`SamplingParams`](../model_lib.md#SamplingParams), [`BaseExperimentConfig`](../config_lib.md#BaseExperimentConfig), [`max_decode_steps`](../utils/sampling_lib.md#SamplingParams.max_decode_steps), [`input_as_chunks`](../utils/sampling_lib.md#input_as_chunks), [`LMInterface`](../model_lib.md#LMInterface), [`format`](../utils/lm_format.md#LMFormat.format), [`create_input_processor`](../utils/sampling_lib.md#create_input_processor), [`TokenizerRegistry`](../utils/tokenization.md#TokenizerRegistry), [`set_mesh`](../utils/sharding.md#set_mesh), [`max_seq_len`](../utils/sampling_lib.md#SamplingParams.max_seq_len), [`LMFormat`](../utils/lm_format.md#LMFormat), [`SimplyServiceResponse`](vanilla_server.md#SimplyServiceResponse), [`intermediate_decode_steps`](../utils/sampling_lib.md#SamplingParams.intermediate_decode_steps), [`InputProcessorInterface`](../utils/sampling_lib.md#InputProcessorInterface), [`PyTree`](vanilla_server.md#PyTree), [`code`](vanilla_server.md#SimplyServiceResponse.code), [`_BATCH_SIZE`](vanilla_server.md#_BATCH_SIZE), [`details`](vanilla_server.md#SimplyServiceResponse.details), [`bos_id`](../utils/lm_format.md#LMFormat.bos_id), [`extra_eos_tokens`](../utils/lm_format.md#LMFormat.extra_eos_tokens), [`pad_id`](../utils/lm_format.md#LMFormat.pad_id), [`result`](vanilla_server.md#SimplyServiceResponse.result), [`_INTERMEDIATE_DECODE_STEPS`](vanilla_server.md#_INTERMEDIATE_DECODE_STEPS), [`_MAX_DECODE_STEPS`](vanilla_server.md#_MAX_DECODE_STEPS), [`_MAX_SEQ_LEN`](vanilla_server.md#_MAX_SEQ_LEN)
- used by: [`main`](vanilla_server.md#main), [`Run`](vanilla_server.md#SimplyService.Run), [`_batcher_loop`](vanilla_server.md#SimplyService._batcher_loop), [`_init_fn`](vanilla_server.md#main._init_fn), [`batcher`](vanilla_server.md#SimplyService.batcher)

### `SimplyService`
- def: [`simply/serving/vanilla_server.py:262`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L262)
- doc: The Simple service with batching.
- signature: `class SimplyService(server_pb2_grpc.SimplyService):`
- members:
  - `Run(self, request: struct_pb2.Value, context: grpc.aio.ServicerContext)` — [`L288`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L288)
  - `batcher_thread(self)` — [`L276`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L276)
  - `error_message_queue(self)` — [`L272`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L272)
  - `stop_event(self)` — [`L268`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L268)
  - `batcher` — [`L265`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L265)
- protocol/private: `_batcher_loop`[`L278`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L278)
- uses (calls/refs, reference-scoped): [`loop`](vanilla_server.md#Batcher.loop), [`enqueue`](vanilla_server.md#Batcher.enqueue), [`py_to_struct_pb`](common.md#py_to_struct_pb), [`struct_pb_to_py`](common.md#struct_pb_to_py), [`SimplyServiceResponse`](vanilla_server.md#SimplyServiceResponse), [`code`](vanilla_server.md#SimplyServiceResponse.code), [`details`](vanilla_server.md#SimplyServiceResponse.details), [`Batcher`](vanilla_server.md#Batcher), [`result`](vanilla_server.md#SimplyServiceResponse.result)
- used by: [`main`](vanilla_server.md#main), [`_init_fn`](vanilla_server.md#main._init_fn)

### `SimplyServiceResponse`  ·  implements/extends NamedTuple
- def: [`simply/serving/vanilla_server.py:138`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L138)
- signature: `class SimplyServiceResponse(NamedTuple):`
- members:
  - `code` — [`L139`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L139)
  - `details` — [`L140`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L140)
  - `result` — [`L141`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L141)
- used by: [`loop`](vanilla_server.md#Batcher.loop), [`Run`](vanilla_server.md#SimplyService.Run), [`enqueue`](vanilla_server.md#Batcher.enqueue), [`queue`](vanilla_server.md#Batcher.queue)

## Functions
- `_init_fn()` — [`L369`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L369)
- `main(argv: Sequence[str])` — [`L316`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L316) — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)

## Module values
- `PyTree` — [`L135`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L135)
- `_ACTIVATION_DTYPE` — [`L106`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L106)
- `_BATCH_SIZE` — [`L102`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L102)
- `_CKPT_DIR` — [`L81`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L81)
- `_CKPT_FORMAT` — [`L91`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L91)
- `_CKPT_STEP` — [`L85`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L85)
- `_EXPERIMENT_CONFIG` — [`L70`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L70)
- `_INTERMEDIATE_DECODE_STEPS` — [`L120`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L120)
- `_LM_FORMAT` — [`L127`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L127)
- `_MAX_DECODE_STEPS` — [`L114`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L114)
- `_MAX_SEQ_LEN` — [`L110`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L110)
- `_MESH_SHAPE` — [`L77`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L77)
- `_SIMPLY_PORT` — [`L66`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L66)
- `_VOCAB_NAME` — [`L95`](../../../../../../raw/code/simply/simply/serving/vanilla_server.py#L95)

