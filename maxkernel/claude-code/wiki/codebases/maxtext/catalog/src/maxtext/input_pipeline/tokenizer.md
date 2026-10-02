---
title: 'Module: src/maxtext/input_pipeline/tokenizer.py'
type: catalog
provenance: extracted
module: src/maxtext/input_pipeline/tokenizer.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.input_pipeline.tokenizer`/
symbols:
  TikTokenTokenizer.encode: TikTokenTokenizer#encode().
  SentencePieceTokenizer.encode: SentencePieceTokenizer#encode().
  SentencePieceTokenizer._tokenizer_model: SentencePieceTokenizer#_tokenizer_model.
  TikTokenTokenizer.stop_tokens: TikTokenTokenizer#stop_tokens.
  TikTokenTokenizer.model: TikTokenTokenizer#model.
  build_tokenizer: build_tokenizer().
  TikTokenTokenizer.special_tokens: TikTokenTokenizer#special_tokens.
  HFTokenizer.tokenizer: HFTokenizer#tokenizer.
  TikTokenTokenizer.__init__: TikTokenTokenizer#__init__().
  TikTokenTokenizer.bos_id: TikTokenTokenizer#bos_id.
  TikTokenTokenizer.eos_id: TikTokenTokenizer#eos_id.
  TikTokenTokenizer.n_words: TikTokenTokenizer#n_words.
  SentencePieceTokenizer.bos_id: SentencePieceTokenizer#bos_id.
  SentencePieceTokenizer.eos_id: SentencePieceTokenizer#eos_id.
  TikTokenTokenizer.decode: TikTokenTokenizer#decode().
  SentencePieceTokenizer.pad_id: SentencePieceTokenizer#pad_id.
  SentencePieceTokenizer.unk_id: SentencePieceTokenizer#unk_id.
  SentencePieceTokenizer.decode: SentencePieceTokenizer#decode().
  HFTokenizer.pad_id: HFTokenizer#pad_id.
  HFTokenizer.unk_id: HFTokenizer#unk_id.
  HFTokenizer.bos_id: HFTokenizer#bos_id.
  HFTokenizer.eos_id: HFTokenizer#eos_id.
  HFTokenizer.encode: HFTokenizer#encode().
  HFTokenizer.decode: HFTokenizer#decode().
  TikTokenTokenizer: TikTokenTokenizer#
  TikTokenTokenizer.num_reserved_special_tokens: TikTokenTokenizer#num_reserved_special_tokens.
  TikTokenTokenizer.pat_str: TikTokenTokenizer#pat_str.
  TikTokenTokenizer.eos: TikTokenTokenizer#eos.
  TikTokenTokenizer.bos: TikTokenTokenizer#bos.
  TikTokenTokenizer._split_whitespaces_or_nonwhitespaces: TikTokenTokenizer#_split_whitespaces_or_nonwhitespaces().
  SentencePieceTokenizer: SentencePieceTokenizer#
  SentencePieceTokenizer.add_bos: SentencePieceTokenizer#add_bos.
  SentencePieceTokenizer.add_eos: SentencePieceTokenizer#add_eos.
  HFTokenizer: HFTokenizer#
  TikTokenTokenizer.pad_id: TikTokenTokenizer#pad_id.
  SentencePieceTokenizer.__init__: SentencePieceTokenizer#__init__().
  HFTokenizer.__init__: HFTokenizer#__init__().
---
# Module: [`src/maxtext/input_pipeline/tokenizer.py`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py)

## Classes
### `HFTokenizer`
- def: [`src/maxtext/input_pipeline/tokenizer.py:221`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L221)
- doc: Tokenizing using huggingface tokenizer
- signature: `class HFTokenizer:`
- members:
  - `decode(self, t: Sequence[int])` — [`L245`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L245)
  - `encode(self, s: str)` — [`L242`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L242)
  - `bos_id` — [`L239`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L239)
  - `eos_id` — [`L240`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L240)
  - `pad_id` — [`L237`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L237)
  - `tokenizer` — [`L229`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L229)
  - `unk_id` — [`L238`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L238)
- protocol/private: `__init__`[`L226`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L226)
- used by: [`build_tokenizer`](tokenizer.md#build_tokenizer)

### `SentencePieceTokenizer`
- def: [`src/maxtext/input_pipeline/tokenizer.py:183`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L183)
- doc: Tokenizing and encoding/decoding text using the native sentencepiece library.
- signature: `class SentencePieceTokenizer:`
- members:
  - `decode(self, t: Sequence[int])` — [`L217`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L217)
  - `encode(self, s: str)` — [`L209`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L209)
  - `add_bos` — [`L206`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L206)
  - `add_eos` — [`L207`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L207)
  - `bos_id` — [`L204`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L204)
  - `eos_id` — [`L205`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L205)
  - `pad_id` — [`L202`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L202)
  - `unk_id` — [`L203`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L203)
- protocol/private: `__init__`[`L189`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L189), `_tokenizer_model`[`L191`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L191)
- used by: [`build_tokenizer`](tokenizer.md#build_tokenizer)

### `TikTokenTokenizer`
- def: [`src/maxtext/input_pipeline/tokenizer.py:26`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L26)
- doc: Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
- signature: `class TikTokenTokenizer:`
- members:
  - `__init__(self, model_path: str, add_bos: bool, add_eos: bool)` — [`L37`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L37) — Initializes the Tokenizer with a Tiktoken model.
  - `_split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int)` — [`L159`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L159) — Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
  - `decode(self, t)` — [`L145`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L145) — Decodes a list of token IDs into a string.
  - `encode(self, s: str, *, allowed_special: Literal["all"] | Collection[str] = (), disallowed_special: Literal["all"] | Collection[str] = ())` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L84) — Encodes a string into a list of token IDs.
  - `bos` — [`L70`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L70)
  - `bos_id` — [`L75`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L75)
  - `eos` — [`L69`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L69)
  - `eos_id` — [`L76`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L76)
  - `model` — [`L63`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L63)
  - `n_words` — [`L73`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L73)
  - `num_reserved_special_tokens` — [`L33`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L33)
  - `pad_id` — [`L77`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L77)
  - `pat_str` — [`L35`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L35)
  - `special_tokens` — [`L31`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L31)
  - `stop_tokens` — [`L78`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L78)
- used by: [`build_tokenizer`](tokenizer.md#build_tokenizer)

## Functions
- `build_tokenizer(tokenizer_path, tokenizer_type, add_bos, add_eos, hf_access_token)` — [`L249`](../../../../../../../raw/code/maxtext/src/maxtext/input_pipeline/tokenizer.py#L249) — Loads the tokenizer at `tokenizer_path`

