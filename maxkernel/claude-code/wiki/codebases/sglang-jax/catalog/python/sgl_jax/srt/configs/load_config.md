---
title: 'Module: python/sgl_jax/srt/configs/load_config.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/configs/load_config.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.configs.load_config`/
symbols:
  LoadConfig.load_format: LoadConfig#load_format.
  LoadConfig: LoadConfig#
  LoadConfig.__post_init__: LoadConfig#__post_init__().
  LoadFormat: LoadFormat#
  LoadConfig.ignore_patterns: LoadConfig#ignore_patterns.
  LoadConfig._verify_load_format: LoadConfig#_verify_load_format().
  LoadConfig.model_loader_extra_config: LoadConfig#model_loader_extra_config.
  LoadConfig.download_dir: LoadConfig#download_dir.
  LoadConfig.model_class: LoadConfig#model_class.
  LoadConfig.sub_dir: LoadConfig#sub_dir.
  logger: logger.
  LoadFormat.AUTO: LoadFormat#AUTO.
  LoadFormat.DUMMY: LoadFormat#DUMMY.
  LoadFormat.JAX: LoadFormat#JAX.
  LoadFormat.PT: LoadFormat#PT.
  LoadFormat.SAFETENSORS: LoadFormat#SAFETENSORS.
  LoadFormat.NPCACHE: LoadFormat#NPCACHE.
  LoadFormat.SHARDED_STATE: LoadFormat#SHARDED_STATE.
  LoadFormat.GGUF: LoadFormat#GGUF.
  LoadFormat.BITSANDBYTES: LoadFormat#BITSANDBYTES.
  LoadFormat.MISTRAL: LoadFormat#MISTRAL.
  LoadFormat.LAYERED: LoadFormat#LAYERED.
  LoadFormat.REMOTE: LoadFormat#REMOTE.
  LoadConfig.decryption_key_file: LoadConfig#decryption_key_file.
---
# Module: [`python/sgl_jax/srt/configs/load_config.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py)

## Classes
### `LoadConfig`
- def: [`python/sgl_jax/srt/configs/load_config.py:26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L26)
- doc: download_dir: Directory to download and load the weights, default to the
- signature: `class LoadConfig:`
- members:
  - `decryption_key_file` — [`L56`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L56)
  - `download_dir` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L50)
  - `ignore_patterns` — [`L55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L55)
  - `load_format` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L49)
  - `model_class` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L54)
  - `model_loader_extra_config` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L52)
  - `sub_dir` — [`L51`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L51)
- protocol/private: `__post_init__`[`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L58), `_verify_load_format`[`L72`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L72)
- uses (calls/refs, reference-scoped): [`LoadFormat`](load_config.md#LoadFormat), [`AUTO`](load_config.md#LoadFormat.AUTO), [`logger`](load_config.md#logger)
- used by: [`load_lora_weights`](../lora/lora_manager.md#LoRAManager.load_lora_weights), [`model_loader`](../model_executor/model_runner.md#ModelRunner.model_loader), [`get_model_loader`](../model_loader/loader.md#get_model_loader), [`load_model`](../model_loader/loader.md#JAXModelLoader.load_model), [`_prepare_weights`](../model_loader/loader.md#DefaultModelLoader._prepare_weights), [`__init__`](../model_loader/loader.md#DefaultModelLoader.__init__), [`_initialize_model`](../model_loader/loader.md#JAXModelLoader._initialize_model), [`_maybe_download_from_modelscope`](../model_loader/loader.md#DefaultModelLoader._maybe_download_from_modelscope), [`__init__`](../model_loader/loader.md#JAXDummyModelLoader.__init__), [`__init__`](../lora/lora.md#LoRAAdapter.__init__), [`__init__`](../model_loader/loader.md#JAXModelLoader.__init__), [`__init__`](../model_loader/loader.md#BaseModelLoader.__init__), [`load_config`](../lora/lora.md#LoRAAdapter.load_config)

### `LoadFormat`  ·  implements/extends Enum, str
- def: [`python/sgl_jax/srt/configs/load_config.py:10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L10)
- signature: `class LoadFormat(str, enum.Enum):`
- members:
  - `AUTO` — [`L11`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L11)
  - `BITSANDBYTES` — [`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L18)
  - `DUMMY` — [`L15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L15)
  - `GGUF` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L17)
  - `JAX` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L21)
  - `LAYERED` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L20)
  - `MISTRAL` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L19)
  - `NPCACHE` — [`L14`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L14)
  - `PT` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L12)
  - `REMOTE` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L22)
  - `SAFETENSORS` — [`L13`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L13)
  - `SHARDED_STATE` — [`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L16)
- used by: [`get_model_loader`](../model_loader/loader.md#get_model_loader), [`load_format`](load_config.md#LoadConfig.load_format), [`_verify_load_format`](load_config.md#LoadConfig._verify_load_format)

## Module values
- `logger` — [`L7`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/configs/load_config.py#L7)

