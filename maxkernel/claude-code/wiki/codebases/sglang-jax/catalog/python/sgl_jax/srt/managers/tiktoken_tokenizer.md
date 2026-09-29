---
title: 'Module: python/sgl_jax/srt/managers/tiktoken_tokenizer.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/managers/tiktoken_tokenizer.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.managers.tiktoken_tokenizer`/
symbols:
  TiktokenTokenizer.__init__: TiktokenTokenizer#__init__().
  DEFAULT_CONTROL_TOKENS: DEFAULT_CONTROL_TOKENS.
  DEFAULT_SPECIAL_TOKENS: DEFAULT_SPECIAL_TOKENS.
  TiktokenTokenizer.tokenizer: TiktokenTokenizer#tokenizer.
  TiktokenTokenizer.eos_token_id: TiktokenTokenizer#eos_token_id.
  TiktokenTokenizer.all_special_tokens: TiktokenTokenizer#all_special_tokens().
  TiktokenTokenizer.convert_tokens_to_ids: TiktokenTokenizer#convert_tokens_to_ids().
  TiktokenTokenizer: TiktokenTokenizer#
  EOS: EOS.
  TiktokenTokenizer.apply_chat_template: TiktokenTokenizer#apply_chat_template().
  TiktokenTokenizer.all_special_ids: TiktokenTokenizer#all_special_ids().
  TiktokenTokenizer.encode: TiktokenTokenizer#encode().
  TiktokenTokenizer.chat_template_jinja: TiktokenTokenizer#chat_template_jinja.
  TiktokenProcessor.__init__: TiktokenProcessor#__init__().
  TiktokenTokenizer.decode: TiktokenTokenizer#decode().
  TiktokenTokenizer.batch_decode: TiktokenTokenizer#batch_decode().
  TiktokenTokenizer.__call__: TiktokenTokenizer#__call__().
  PAD: PAD.
  SEP: SEP.
  RESERVED_TOKEN_TEXTS: RESERVED_TOKEN_TEXTS.
  CONTROL_TOKEN_TEXTS: CONTROL_TOKEN_TEXTS.
  PAT_STR_B: PAT_STR_B.
  TiktokenTokenizer.encode_patched: TiktokenTokenizer#encode_patched().
  TiktokenTokenizer.chat_template: TiktokenTokenizer#chat_template.
  TiktokenProcessor: TiktokenProcessor#
  TiktokenProcessor.tokenizer: TiktokenProcessor#tokenizer.
  TiktokenProcessor.image_processor: TiktokenProcessor#image_processor().
  TiktokenTokenizer.bos_token_id: TiktokenTokenizer#bos_token_id.
  TiktokenTokenizer.vocab_size: TiktokenTokenizer#vocab_size.
  TiktokenTokenizer.additional_stop_token_ids: TiktokenTokenizer#additional_stop_token_ids.
---
# Module: [`python/sgl_jax/srt/managers/tiktoken_tokenizer.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py)

## Classes
### `TiktokenProcessor`
- def: [`python/sgl_jax/srt/managers/tiktoken_tokenizer.py:8`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L8)
- signature: `class TiktokenProcessor:`
- members:
  - `image_processor(self, image)` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L12)
  - `tokenizer` — [`L10`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L10)
- protocol/private: `__init__`[`L9`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L9)
- uses (calls/refs, reference-scoped): [`TiktokenTokenizer`](tiktoken_tokenizer.md#TiktokenTokenizer)

### `TiktokenTokenizer`
- def: [`python/sgl_jax/srt/managers/tiktoken_tokenizer.py:31`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L31)
- members:
  - `all_special_ids(self)` — [`L153`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L153) — `list[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
  - `all_special_tokens(self)` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L134) — `list[str]`: List all special tokens in the tokenizer.
  - `apply_chat_template(self, messages, tokenize, add_generation_prompt, tools=None, reasoning_effort=None)` — [`L115`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L115)
  - `batch_decode(self, batch, skip_special_tokens=True, spaces_between_special_tokens=False)` — [`L110`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L110)
  - `convert_tokens_to_ids(self, tokens: list[str])` — [`L140`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L140) — Convert a list of tokens to token IDs.
  - `decode(self, x, *args, **kwargs)` — [`L107`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L107)
  - `encode(self, x, add_special_tokens=False)` — [`L104`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L104)
  - `encode_patched(self, text: str, *, allowed_special: Literal["all"] | Set[str] = set(), disallowed_special: Literal["all"] | Collection[str] = "all")` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L73)
  - `additional_stop_token_ids` — [`L102`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L102)
  - `bos_token_id` — [`L97`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L97)
  - `chat_template` — [`L100`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L100)
  - `chat_template_jinja` — [`L101`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L101)
  - `eos_token_id` — [`L98`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L98)
  - `tokenizer` — [`L96`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L96)
  - `vocab_size` — [`L99`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L99)
- protocol/private: `__call__`[`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L128), `__init__`[`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L32)
- uses (calls/refs, reference-scoped): [`DEFAULT_CONTROL_TOKENS`](tiktoken_tokenizer.md#DEFAULT_CONTROL_TOKENS), [`EOS`](tiktoken_tokenizer.md#EOS), `_special_tokens`, [`CONTROL_TOKEN_TEXTS`](tiktoken_tokenizer.md#CONTROL_TOKEN_TEXTS), [`PAT_STR_B`](tiktoken_tokenizer.md#PAT_STR_B), [`RESERVED_TOKEN_TEXTS`](tiktoken_tokenizer.md#RESERVED_TOKEN_TEXTS)
- used by: [`get_tokenizer`](../hf_transformers_utils.md#get_tokenizer), [`get_guidance_backend`](../constrained/llguidance_backend.md#get_guidance_backend), [`__init__`](tiktoken_tokenizer.md#TiktokenProcessor.__init__)

## Module values
- `CONTROL_TOKEN_TEXTS` — [`L17`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L17)
- `DEFAULT_CONTROL_TOKENS` — [`L25`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L25)
- `DEFAULT_SPECIAL_TOKENS` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L24)
- `EOS` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L21)
- `PAD` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L20)
- `PAT_STR_B` — [`L28`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L28)
- `RESERVED_TOKEN_TEXTS` — [`L16`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L16)
- `SEP` — [`L22`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/managers/tiktoken_tokenizer.py#L22)

