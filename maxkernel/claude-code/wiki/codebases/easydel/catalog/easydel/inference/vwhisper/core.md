---
title: 'Module: easydel/inference/vwhisper/core.py'
type: catalog
provenance: extracted
module: easydel/inference/vwhisper/core.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.vwhisper.core`/vWhisperInference#
symbols:
  vWhisperInference._generate: _generate().
  vWhisperInference.generate: generate().
  vWhisperInference: ''
  vWhisperInference.feature_extractor: feature_extractor.
  vWhisperInference._process_model_outputs: _process_model_outputs().
  vWhisperInference.graphstate: graphstate.
  vWhisperInference._process_model_inputs: _process_model_inputs().
  vWhisperInference.max_length: max_length.
  vWhisperInference.__init__: __init__().
  vWhisperInference.inference_config: inference_config.
  vWhisperInference.generate_function: generate_function.
  vWhisperInference._single_batch_process: _single_batch_process().
  vWhisperInference.model: model.
  vWhisperInference.generation_config: generation_config.
  vWhisperInference.processor: processor.
  vWhisperInference.tokenizer: tokenizer.
  vWhisperInference.graphdef: graphdef.
  vWhisperInference.dtype: dtype.
  vWhisperInference.__call__: __call__.
---
# Module: [`easydel/inference/vwhisper/core.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py)

## Classes
### `vWhisperInference`
- def: [`easydel/inference/vwhisper/core.py:75`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L75)
- doc: High-performance speech-to-text inference engine using Whisper models.
- signature: `class vWhisperInference:`
- members:
  - `__init__(self, model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer, processor: WhisperProcessor, inference_config: vWhisperInferenceConfig | None = None, dtype: jax.typing.DTypeLike = jnp.float32)` — [`L173`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L173) — Initialize the vWhisperInference engine.
  - `_generate(self, input_features: jax.Array, language: str | None = None, task: str | None = None, return_timestamps: bool = False)` — [`L252`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L252) — Generate token sequences from processed audio features.
  - `_process_model_inputs(self, audio_input: str | bytes | np.ndarray | dict[str, np.ndarray | int], chunk_length_s: float = 30, stride_length_s: float | list[float] | None = None, batch_size: int | None = None)` — [`L309`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L309) — Process and chunk audio input for model consumption.
  - `_process_model_outputs(self, model_outputs, return_timestamps: bool | None = None, return_language: str | None = None)` — [`L397`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L397) — Convert raw model outputs to formatted transcription results.
  - `_single_batch_process(self, model_inputs: dict[str, tp.Any], batch_size: int, language: str | None = None, task: str | None = None, return_timestamps: bool = False)` — [`L454`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L454) — Process a single batch of audio chunks through the model.
  - `generate(self, audio_input: str | bytes | np.ndarray | dict[str, np.ndarray | int], chunk_length_s: float = 30, stride_length_s: float | list[float] | None = None, batch_size: int | None = None, language: str | None = None, task: str | None = None, return_timestamps: bool | None = None)` — [`L508`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L508) — Transcribe or translate audio input to text.
  - `dtype` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L237)
  - `feature_extractor` — [`L239`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L239)
  - `generate_function` — [`L250`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L250) — ---
  - `generation_config` — [`L247`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L247)
  - `graphdef` — [`L243`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L243)
  - `graphstate` — [`L244`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L244)
  - `inference_config` — [`L249`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L249)
  - `max_length` — [`L248`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L248)
  - `model` — [`L241`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L241)
  - `processor` — [`L238`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L238)
  - `tokenizer` — [`L240`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L240)
- protocol/private: `__call__`[`L620`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/core.py#L620)
- uses (calls/refs, reference-scoped): [`config`](../../infra/base_module.md#EasyDeLBaseModule.config), [`Registry`](../../utils/registery.md#Registry), [`generation_config`](../../infra/mixins/generation.md#EasyGenerationMixin.generation_config), [`register`](../../utils/registery.md#Registry.register), `max_length`, [`WhisperForConditionalGeneration`](../../modules/whisper/modeling_whisper.md#WhisperForConditionalGeneration), [`vWhisperInferenceConfig`](config.md#vWhisperInferenceConfig), [`_compiled_generate`](generation.md#_compiled_generate), [`chunk_iter_with_batch`](utils.md#chunk_iter_with_batch), [`get_decoder_input_ids`](generation.md#get_decoder_input_ids), [`process_audio_input`](utils.md#process_audio_input), [`generation_config`](config.md#vWhisperInferenceConfig.generation_config), [`batch_size`](config.md#vWhisperInferenceConfig.batch_size), [`language`](config.md#vWhisperInferenceConfig.language), [`max_length`](config.md#vWhisperInferenceConfig.max_length), [`return_timestamps`](config.md#vWhisperInferenceConfig.return_timestamps), [`task`](config.md#vWhisperInferenceConfig.task)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`inference`](server.md#WhisperModel.inference)

