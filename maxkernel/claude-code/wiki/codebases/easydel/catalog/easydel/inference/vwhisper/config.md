---
title: 'Module: easydel/inference/vwhisper/config.py'
type: catalog
provenance: extracted
module: easydel/inference/vwhisper/config.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.inference.vwhisper.config`/vWhisperInferenceConfig#
symbols:
  vWhisperInferenceConfig: ''
  vWhisperInferenceConfig.__hash__: __hash__().
  vWhisperInferenceConfig.generation_config: generation_config.
  vWhisperInferenceConfig.batch_size: batch_size.
  vWhisperInferenceConfig.max_length: max_length.
  vWhisperInferenceConfig.return_timestamps: return_timestamps.
  vWhisperInferenceConfig.task: task.
  vWhisperInferenceConfig.language: language.
  vWhisperInferenceConfig.logits_processor: logits_processor.
  vWhisperInferenceConfig.is_multilingual: is_multilingual.
---
# Module: [`easydel/inference/vwhisper/config.py`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py)

## Classes
### `vWhisperInferenceConfig`
- def: [`easydel/inference/vwhisper/config.py:63`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L63)
- doc: Configuration class for vWhisper inference settings.
- signature: `class vWhisperInferenceConfig:`
- members:
  - `__hash__(self)` — [`L157`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L157) — Compute a hash value for the configuration.
  - `batch_size` — [`L148`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L148)
  - `generation_config` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L150)
  - `is_multilingual` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L155)
  - `language` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L154)
  - `logits_processor` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L151)
  - `max_length` — [`L149`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L149)
  - `return_timestamps` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L152)
  - `task` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/inference/vwhisper/config.py#L153)
- uses (calls/refs, reference-scoped): [`get_safe_hash_int`](../../utils/compiling_utils.md#get_safe_hash_int)
- used by: [`_import_structure`](../../__init__.md#_import_structure), [`generate`](core.md#vWhisperInference.generate), [`graphstate`](core.md#vWhisperInference.graphstate), [`max_length`](core.md#vWhisperInference.max_length), [`__init__`](core.md#vWhisperInference.__init__)

