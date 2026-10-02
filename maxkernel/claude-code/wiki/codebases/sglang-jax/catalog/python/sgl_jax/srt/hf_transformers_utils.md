---
title: 'Module: python/sgl_jax/srt/hf_transformers_utils.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/hf_transformers_utils.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.hf_transformers_utils`/
symbols:
  get_tokenizer: get_tokenizer().
  _CONFIG_REGISTRY._CONFIG_REGISTRY: _CONFIG_REGISTRY._CONFIG_REGISTRY.
  get_config: get_config().
  cls: cls.
  _ensure_fastokens_patched: _ensure_fastokens_patched().
  download_from_hf: download_from_hf().
  get_processor: get_processor().
  get_generation_config: get_generation_config().
  get_context_length: get_context_length().
  _FASTOKENS_PATCHED: _FASTOKENS_PATCHED.
  check_gguf_file: check_gguf_file().
  logger: logger.
  get_hf_text_config: get_hf_text_config().
  _UNSET: _UNSET.
  attach_additional_stop_token_ids: attach_additional_stop_token_ids().
  GlmMoeDsaConfig: GlmMoeDsaConfig#
  CONTEXT_LENGTH_KEYS: CONTEXT_LENGTH_KEYS.
  _FAST_LLAMA_TOKENIZER: _FAST_LLAMA_TOKENIZER.
  _FASTOKENS_PATCH_LOCK: _FASTOKENS_PATCH_LOCK.
  _validate_tokenizer_backend: _validate_tokenizer_backend().
  _raise_fastokens_load_error: _raise_fastokens_load_error().
  get_tokenizer_from_processor: get_tokenizer_from_processor().
  name: name.
  GlmMoeDsaConfig.model_type: GlmMoeDsaConfig#model_type.
---
# Module: [`python/sgl_jax/srt/hf_transformers_utils.py`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py)

## Classes
### `GlmMoeDsaConfig`  ·  implements/extends PretrainedConfig
- def: [`python/sgl_jax/srt/hf_transformers_utils.py:34`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L34)
- signature: `class GlmMoeDsaConfig(PretrainedConfig):`
- members:
  - `model_type` — [`L37`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L37)
- used by: [`_CONFIG_REGISTRY`](hf_transformers_utils.md#_CONFIG_REGISTRY._CONFIG_REGISTRY)

## Functions
- `_ensure_fastokens_patched()` — [`L223`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L223) — Monkey-patch transformers process-wide to use the fastokens backend once.
- `_raise_fastokens_load_error(tokenizer_name: str, error: Exception)` — [`L246`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L246)
- `_validate_tokenizer_backend(tokenizer_backend: str)` — [`L215`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L215)
- `attach_additional_stop_token_ids(tokenizer)` — [`L522`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L522)
- `check_gguf_file(model: str | os.PathLike)` — [`L530`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L530) — Check if the file is a GGUF model.
- `download_from_hf(model_path: str, allow_patterns: list[str] | None = _UNSET, cache_dir: str | None = None)` — [`L75`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L75)
- `get_config(model: str, trust_remote_code: bool, revision: str | None = None, model_override_args: dict | None = None, **kwargs)` — [`L112`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L112) — documented in [python-sgl_jax-srt-configs-model_config](../../../../concepts/python-sgl_jax-srt-configs-model_config.md)
- `get_context_length(config)` — [`L189`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L189) — Get the context length of a model from a huggingface model configs.
- `get_generation_config(model: str, trust_remote_code: bool, revision: str | None = None, **kwargs)` — [`L162`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L162)
- `get_hf_text_config(config: PretrainedConfig)` — [`L86`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L86) — Get the "sub" config relevant to llm for multi modal models.
- `get_processor(tokenizer_name: str, *args, tokenizer_mode: str = "auto", trust_remote_code: bool = False, tokenizer_revision: str | None = None, use_fast: bool | None = True, **kwargs)` — [`L482`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L482)
- `get_tokenizer(tokenizer_name: str, *args, tokenizer_mode: str = "auto", trust_remote_code: bool = False, tokenizer_revision: str | None = None, tokenizer_backend: str = "huggingface", sub_dir: str = "", download_dir: str | None = None, **kwargs)` — [`L255`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L255) — Gets a tokenizer for the given model name via Huggingface.
- `get_tokenizer_from_processor(processor)` — [`L476`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L476)

## Module values
- `CONTEXT_LENGTH_KEYS` — [`L180`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L180)
- `_CONFIG_REGISTRY` — [`L40`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L40)
- `_FASTOKENS_PATCHED` — [`L211`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L211)
- `_FASTOKENS_PATCH_LOCK` — [`L212`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L212)
- `_FAST_LLAMA_TOKENIZER` — [`L210`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L210)
- `_UNSET` — [`L72`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L72)
- `cls` — [`L60`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L60)
- `logger` — [`L31`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L31)
- `name` — [`L60`](../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/hf_transformers_utils.py#L60)

