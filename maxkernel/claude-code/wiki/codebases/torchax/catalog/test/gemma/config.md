---
title: 'Module: test/gemma/config.py'
type: catalog
provenance: extracted
module: test/gemma/config.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.gemma.config`/
symbols:
  get_config_for_2b: get_config_for_2b().
  GemmaConfig.hidden_size: GemmaConfig#hidden_size.
  GemmaConfig: GemmaConfig#
  get_model_config: get_model_config().
  GemmaConfig.get_dtype: GemmaConfig#get_dtype().
  GemmaConfig.num_hidden_layers: GemmaConfig#num_hidden_layers.
  GemmaConfig.num_key_value_heads: GemmaConfig#num_key_value_heads.
  GemmaConfig.num_attention_heads: GemmaConfig#num_attention_heads.
  GemmaConfig.head_dim: GemmaConfig#head_dim.
  GemmaConfig.quant: GemmaConfig#quant.
  get_config_for_7b: get_config_for_7b().
  GemmaConfig.intermediate_size: GemmaConfig#intermediate_size.
  GemmaConfig.rms_norm_eps: GemmaConfig#rms_norm_eps.
  GemmaConfig.vocab_size: GemmaConfig#vocab_size.
  GemmaConfig.max_position_embeddings: GemmaConfig#max_position_embeddings.
  GemmaConfig.dtype: GemmaConfig#dtype.
  _STR_DTYPE_TO_TORCH_DTYPE: _STR_DTYPE_TO_TORCH_DTYPE.
  GemmaConfig.tokenizer: GemmaConfig#tokenizer.
---
# Module: [`test/gemma/config.py`](../../../../../../raw/code/torchax/test/gemma/config.py)

## Classes
### `GemmaConfig`
- def: [`test/gemma/config.py:35`](../../../../../../raw/code/torchax/test/gemma/config.py#L35)
- signature: `class GemmaConfig:`
- members:
  - `get_dtype(self)` — [`L61`](../../../../../../raw/code/torchax/test/gemma/config.py#L61) — Gets the torch dtype from the config dtype string.
  - `dtype` — [`L55`](../../../../../../raw/code/torchax/test/gemma/config.py#L55)
  - `head_dim` — [`L51`](../../../../../../raw/code/torchax/test/gemma/config.py#L51)
  - `hidden_size` — [`L47`](../../../../../../raw/code/torchax/test/gemma/config.py#L47)
  - `intermediate_size` — [`L49`](../../../../../../raw/code/torchax/test/gemma/config.py#L49)
  - `max_position_embeddings` — [`L39`](../../../../../../raw/code/torchax/test/gemma/config.py#L39)
  - `num_attention_heads` — [`L43`](../../../../../../raw/code/torchax/test/gemma/config.py#L43)
  - `num_hidden_layers` — [`L41`](../../../../../../raw/code/torchax/test/gemma/config.py#L41)
  - `num_key_value_heads` — [`L45`](../../../../../../raw/code/torchax/test/gemma/config.py#L45)
  - `quant` — [`L57`](../../../../../../raw/code/torchax/test/gemma/config.py#L57)
  - `rms_norm_eps` — [`L53`](../../../../../../raw/code/torchax/test/gemma/config.py#L53)
  - `tokenizer` — [`L59`](../../../../../../raw/code/torchax/test/gemma/config.py#L59)
  - `vocab_size` — [`L37`](../../../../../../raw/code/torchax/test/gemma/config.py#L37)
- uses (calls/refs, reference-scoped): (1 test-only callers)
- used by: (18 test-only callers)

## Functions
- `get_config_for_2b()` — [`L70`](../../../../../../raw/code/torchax/test/gemma/config.py#L70)
- `get_config_for_7b()` — [`L66`](../../../../../../raw/code/torchax/test/gemma/config.py#L66)
- `get_model_config(variant: str)` — [`L80`](../../../../../../raw/code/torchax/test/gemma/config.py#L80)

## Module values
- `_STR_DTYPE_TO_TORCH_DTYPE` — [`L24`](../../../../../../raw/code/torchax/test/gemma/config.py#L24)

