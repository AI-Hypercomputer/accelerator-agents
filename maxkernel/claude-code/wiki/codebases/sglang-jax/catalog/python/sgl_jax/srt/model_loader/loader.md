---
title: 'Module: python/sgl_jax/srt/model_loader/loader.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/model_loader/loader.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.model_loader.loader`/
symbols:
  JAXModelLoader._get_model: JAXModelLoader#_get_model().
  get_model_loader: get_model_loader().
  JAXModelLoader.load_model: JAXModelLoader#load_model().
  JAXModelLoader.download_model: JAXModelLoader#download_model().
  DefaultModelLoader._get_weights_iterator: DefaultModelLoader#_get_weights_iterator().
  DefaultModelLoader._prepare_weights: DefaultModelLoader#_prepare_weights().
  DefaultModelLoader.download_model: DefaultModelLoader#download_model().
  JAXDummyModelLoader.load_model: JAXDummyModelLoader#load_model().
  DefaultModelLoader.__init__: DefaultModelLoader#__init__().
  DefaultModelLoader._maybe_download_from_modelscope: DefaultModelLoader#_maybe_download_from_modelscope().
  JAXModelLoader._warmup_safetensors_cache: JAXModelLoader#_warmup_safetensors_cache().
  JAXModelLoader._initialize_model: JAXModelLoader#_initialize_model().
  JAXDummyModelLoader.__init__: JAXDummyModelLoader#__init__().
  DefaultModelLoader: DefaultModelLoader#
  BaseModelLoader.load_model: BaseModelLoader#load_model().
  JAXModelLoader.JAXSource.init_new: JAXModelLoader#JAXSource#init_new().
  BaseModelLoader: BaseModelLoader#
  BaseModelLoader.load_config: BaseModelLoader#load_config.
  BaseModelLoader.download_model: BaseModelLoader#download_model().
  DefaultModelLoader.Source.init_new: DefaultModelLoader#Source#init_new().
  logger: logger.
  JAXDummyModelLoader._initialize_model: JAXDummyModelLoader#_initialize_model().
  DefaultModelLoader.load_model: DefaultModelLoader#load_model().
  JAXModelLoader.__init__: JAXModelLoader#__init__().
  BaseModelLoader.__init__: BaseModelLoader#__init__().
  JAXModelLoader: JAXModelLoader#
  JAXDummyModelLoader: JAXDummyModelLoader#
  JAXDummyModelLoader.download_model: JAXDummyModelLoader#download_model().
  DefaultModelLoader.Source: DefaultModelLoader#Source#
  DefaultModelLoader.Source.revision: DefaultModelLoader#Source#revision.
  JAXModelLoader.mesh: JAXModelLoader#mesh.
  JAXDummyModelLoader.mesh: JAXDummyModelLoader#mesh.
  DefaultModelLoader.Source.model_or_path: DefaultModelLoader#Source#model_or_path.
  DefaultModelLoader.Source.prefix: DefaultModelLoader#Source#prefix.
  DefaultModelLoader.Source.fall_back_to_pt: DefaultModelLoader#Source#fall_back_to_pt.
  JAXModelLoader.JAXSource: JAXModelLoader#JAXSource#
  JAXModelLoader.JAXSource.model_or_path: JAXModelLoader#JAXSource#model_or_path.
  JAXModelLoader.JAXSource.revision: JAXModelLoader#JAXSource#revision.
  JAXModelLoader._read_file: JAXModelLoader#_read_file().
  DefaultModelLoader.DEFAULT_NUM_THREADS: DefaultModelLoader#DEFAULT_NUM_THREADS.
---
# Module: [`python/sgl_jax/srt/model_loader/loader.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py)

## Classes
### `BaseModelLoader`  ·  implements/extends ABC
- def: [`python/sgl_jax/srt/model_loader/loader.py:27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L27)
- doc: Base class for model loaders.
- signature: `class BaseModelLoader(ABC):`
- members:
  - `download_model(self, model_config: ModelConfig)` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L34) — Download a model so that it can be immediately loaded.
  - `load_model(self, *, model_config: ModelConfig)` — [`L39`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L39) — Load a model with the given configurations.
  - `load_config` — [`L31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L31)
- protocol/private: `__init__`[`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L30)
- uses (calls/refs, reference-scoped): [`ModelConfig`](../configs/model_config.md#ModelConfig), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`download_model`](loader.md#DefaultModelLoader.download_model), [`load_model`](loader.md#JAXDummyModelLoader.load_model), [`DefaultModelLoader`](loader.md#DefaultModelLoader), [`load_model`](loader.md#DefaultModelLoader.load_model), [`JAXDummyModelLoader`](loader.md#JAXDummyModelLoader), [`download_model`](loader.md#JAXDummyModelLoader.download_model)
- used by: [`model`](../model_executor/model_runner.md#ModelRunner.model), [`get_model_loader`](loader.md#get_model_loader), [`load_model`](loader.md#JAXModelLoader.load_model), [`_prepare_weights`](loader.md#DefaultModelLoader._prepare_weights), [`__init__`](loader.md#DefaultModelLoader.__init__), [`_initialize_model`](loader.md#JAXModelLoader._initialize_model), [`_maybe_download_from_modelscope`](loader.md#DefaultModelLoader._maybe_download_from_modelscope), [`__init__`](loader.md#JAXDummyModelLoader.__init__), [`DefaultModelLoader`](loader.md#DefaultModelLoader), [`JAXDummyModelLoader`](loader.md#JAXDummyModelLoader)

### `DefaultModelLoader`  ·  implements/extends BaseModelLoader
- def: [`python/sgl_jax/srt/model_loader/loader.py:48`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L48)
- doc: Model loader that can load different file types from disk.
- signature: `class DefaultModelLoader(BaseModelLoader):`
- members:
  - `_get_weights_iterator(self, source: Source)` — [`L149`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L149) — Get an iterator for the model weights based on the load format.
  - `download_model(self, model_config: ModelConfig)` — [`L92`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L92)
  - `load_model(self, *, model_config: ModelConfig)` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L98)
  - `DEFAULT_NUM_THREADS` — [`L52`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L52)
- protocol/private: `__init__`[`L79`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L79), `_maybe_download_from_modelscope`[`L105`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L105), `_prepare_weights`[`L124`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L124)
- uses (calls/refs, reference-scoped): [`ModelConfig`](../configs/model_config.md#ModelConfig), [`get_bool_env_var`](../utils/common_utils.md#get_bool_env_var), [`load_model`](loader.md#JAXModelLoader.load_model), [`model_path`](../configs/model_config.md#ModelConfig.model_path), [`download_model`](loader.md#JAXModelLoader.download_model), [`load_format`](../configs/load_config.md#LoadConfig.load_format), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`BaseModelLoader`](loader.md#BaseModelLoader), [`load_config`](loader.md#BaseModelLoader.load_config), [`ignore_patterns`](../configs/load_config.md#LoadConfig.ignore_patterns), [`JAXModelLoader`](loader.md#JAXModelLoader), [`__init__`](loader.md#BaseModelLoader.__init__), [`model_loader_extra_config`](../configs/load_config.md#LoadConfig.model_loader_extra_config), [`revision`](../configs/model_config.md#ModelConfig.revision), [`download_dir`](../configs/load_config.md#LoadConfig.download_dir), [`Source`](loader.md#DefaultModelLoader.Source), [`revision`](loader.md#DefaultModelLoader.Source.revision), [`model_or_path`](loader.md#DefaultModelLoader.Source.model_or_path), [`prefix`](loader.md#DefaultModelLoader.Source.prefix)
- used by: [`initialize_weights`](../lora/lora.md#LoRAAdapter.initialize_weights), [`download_model`](loader.md#JAXModelLoader.download_model), [`load_model`](loader.md#BaseModelLoader.load_model), [`BaseModelLoader`](loader.md#BaseModelLoader), [`download_model`](loader.md#BaseModelLoader.download_model), [`__init__`](loader.md#JAXModelLoader.__init__), [`JAXModelLoader`](loader.md#JAXModelLoader)

### `JAXDummyModelLoader`  ·  implements/extends BaseModelLoader
- def: [`python/sgl_jax/srt/model_loader/loader.py:325`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L325)
- doc: Model loader that will set model weights to random values for JAX models.
- signature: `class JAXDummyModelLoader(BaseModelLoader):`
- members:
  - `download_model(self, model_config: ModelConfig)` — [`L337`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L337)
  - `load_model(self, *, model_config: ModelConfig)` — [`L346`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L346)
  - `mesh` — [`L335`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L335)
- protocol/private: `__init__`[`L328`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L328), `_initialize_model`[`L341`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L341)
- uses (calls/refs, reference-scoped): [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`get_model_architecture`](arch.md#get_model_architecture), [`load_format`](../configs/load_config.md#LoadConfig.load_format), [`dtype`](../configs/model_config.md#ModelConfig.dtype), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`BaseModelLoader`](loader.md#BaseModelLoader), [`__init__`](loader.md#BaseModelLoader.__init__), [`model_loader_extra_config`](../configs/load_config.md#LoadConfig.model_loader_extra_config)
- used by: [`get_model_loader`](loader.md#get_model_loader), [`load_model`](loader.md#BaseModelLoader.load_model), [`BaseModelLoader`](loader.md#BaseModelLoader), [`download_model`](loader.md#BaseModelLoader.download_model)

### `JAXModelLoader`  ·  implements/extends DefaultModelLoader
- def: [`python/sgl_jax/srt/model_loader/loader.py:170`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L170)
- signature: `class JAXModelLoader(DefaultModelLoader):`
- members:
  - `_warmup_safetensors_cache(model_config: ModelConfig)` — [`L193`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L193) — Pre-read safetensors files to warm GCSFuse cache.
  - `download_model(self, model_config: ModelConfig)` — [`L187`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L187)
  - `load_model(self, model_config: ModelConfig)` — [`L241`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L241) — documented in [python-sgl_jax-srt-model_executor-model_runner](../../../../../concepts/python-sgl_jax-srt-model_executor-model_runner.md)
  - `mesh` — [`L185`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L185)
- protocol/private: `__init__`[`L183`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L183), `_get_model`[`L281`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L281), `_initialize_model`[`L263`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L263), `_read_file`[`L225`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L225)
- uses (calls/refs, reference-scoped): [`hf_config`](../configs/model_config.md#ModelConfig.hf_config), [`ModelConfig`](../configs/model_config.md#ModelConfig), [`apply_linear_quantization`](../utils/quantization/quantization_utils.md#apply_linear_quantization), [`get_model_architecture`](arch.md#get_model_architecture), [`quantization_config`](../configs/model_config.md#ModelConfig.quantization_config), [`apply_moe_quantization`](../utils/quantization/quantization_utils.md#apply_moe_quantization), [`model_path`](../configs/model_config.md#ModelConfig.model_path), [`dtype`](../configs/model_config.md#ModelConfig.dtype), [`LoadConfig`](../configs/load_config.md#LoadConfig), [`_prepare_weights`](loader.md#DefaultModelLoader._prepare_weights), [`__init__`](loader.md#DefaultModelLoader.__init__), [`print_parameter_shardings`](../utils/debug_utils.md#print_parameter_shardings), [`DefaultModelLoader`](loader.md#DefaultModelLoader), [`init_new`](loader.md#JAXModelLoader.JAXSource.init_new), [`has_moe_quantization`](../configs/quantization_config.md#QuantizationConfig.has_moe_quantization), [`load_config`](loader.md#BaseModelLoader.load_config), [`logger`](loader.md#logger), [`is_static_checkpoint`](../configs/quantization_config.md#QuantizationConfig.is_static_checkpoint), [`get_linear_rules`](../configs/quantization_config.md#QuantizationConfig.get_linear_rules), [`model_class`](../configs/load_config.md#LoadConfig.model_class), [`sub_dir`](../configs/load_config.md#LoadConfig.sub_dir), [`JAXSource`](loader.md#JAXModelLoader.JAXSource), [`model_or_path`](loader.md#JAXModelLoader.JAXSource.model_or_path), [`revision`](loader.md#JAXModelLoader.JAXSource.revision)
- used by: [`get_model_loader`](loader.md#get_model_loader), [`download_model`](loader.md#DefaultModelLoader.download_model), [`DefaultModelLoader`](loader.md#DefaultModelLoader), [`load_model`](loader.md#DefaultModelLoader.load_model)

### `JAXSource`
- def: [`python/sgl_jax/srt/model_loader/loader.py:172`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L172)
- signature: `class JAXSource:`
- members:
  - `init_new(cls, model_config: ModelConfig)` — [`L177`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L177)
  - `model_or_path` — [`L173`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L173)
  - `revision` — [`L174`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L174)
- uses (calls/refs, reference-scoped): [`ModelConfig`](../configs/model_config.md#ModelConfig), [`model_path`](../configs/model_config.md#ModelConfig.model_path), [`revision`](../configs/model_config.md#ModelConfig.revision)
- used by: [`download_model`](loader.md#JAXModelLoader.download_model)

### `Source`
- def: [`python/sgl_jax/srt/model_loader/loader.py:55`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L55)
- doc: A source for weights.
- signature: `class Source:`
- members:
  - `init_new(cls, model_config: ModelConfig, model)` — [`L71`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L71)
  - `fall_back_to_pt` — [`L67`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L67) — ---
  - `model_or_path` — [`L58`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L58) — ---
  - `prefix` — [`L64`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L64) — ---
  - `revision` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L61) — ---
- uses (calls/refs, reference-scoped): [`ModelConfig`](../configs/model_config.md#ModelConfig), [`model_path`](../configs/model_config.md#ModelConfig.model_path), [`revision`](../configs/model_config.md#ModelConfig.revision)
- used by: [`initialize_weights`](../lora/lora.md#LoRAAdapter.initialize_weights), [`_get_weights_iterator`](loader.md#DefaultModelLoader._get_weights_iterator)

## Functions
- `get_model_loader(load_config: LoadConfig, mesh: jax.sharding.Mesh)` — [`L368`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L368) — Get a model loader based on the load format.

## Module values
- `logger` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/model_loader/loader.py#L24)

