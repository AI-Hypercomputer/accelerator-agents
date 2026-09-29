---
title: 'Module: python/sgl_jax/srt/constrained/llguidance_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/constrained/llguidance_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.constrained.llguidance_backend`/
symbols:
  get_guidance_backend: get_guidance_backend().
  GuidanceBackend.dispatch_json: GuidanceBackend#dispatch_json().
  GuidanceBackend._from_serialized: GuidanceBackend#_from_serialized().
  GuidanceGrammar.fill_vocab_mask: GuidanceGrammar#fill_vocab_mask().
  GuidanceGrammar.ll_matcher: GuidanceGrammar#ll_matcher.
  GuidanceBackend.dispatch_ebnf: GuidanceBackend#dispatch_ebnf().
  GuidanceGrammar: GuidanceGrammar#
  GuidanceGrammar.finished: GuidanceGrammar#finished.
  GuidanceGrammar.accept_token: GuidanceGrammar#accept_token().
  GuidanceGrammar.copy: GuidanceGrammar#copy().
  GuidanceBackend.dispatch_structural_tag: GuidanceBackend#dispatch_structural_tag().
  logger: logger.
  GuidanceGrammar.bitmask: GuidanceGrammar#bitmask.
  GuidanceGrammar.allocate_vocab_mask: GuidanceGrammar#allocate_vocab_mask().
  GuidanceBackend.dispatch_regex: GuidanceBackend#dispatch_regex().
  GuidanceGrammar.eos_token: GuidanceGrammar#eos_token.
  GuidanceGrammar._check_err: GuidanceGrammar#_check_err().
  GuidanceBackend.llguidance_tokenizer: GuidanceBackend#llguidance_tokenizer.
  GuidanceBackend: GuidanceBackend#
  GuidanceGrammar.llguidance_tokenizer: GuidanceGrammar#llguidance_tokenizer.
  GuidanceGrammar.__init__: GuidanceGrammar#__init__().
  GuidanceGrammar.is_terminated: GuidanceGrammar#is_terminated().
  GuidanceBackend.__init__: GuidanceBackend#__init__().
  GuidanceGrammar.serialized_grammar: GuidanceGrammar#serialized_grammar.
  GuidanceBackend.any_whitespace: GuidanceBackend#any_whitespace.
  GuidanceBackend.whitespace_pattern: GuidanceBackend#whitespace_pattern.
---
# Module: [`python/sgl_jax/srt/constrained/llguidance_backend.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py)

## Classes
### `GuidanceBackend`
- def: [`python/sgl_jax/srt/constrained/llguidance_backend.py:131`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L131)
- members:
  - `__init__(self, tokenizer, any_whitespace: bool = True, whitespace_pattern: str | None = None, num_threads: int = 4)` — [`L134`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L134) — Initialize the llguidance backend.
  - `dispatch_ebnf(self, key_string: str)` — [`L184`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L184)
  - `dispatch_json(self, key_string: str)` — [`L166`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L166)
  - `dispatch_regex(self, key_string: str)` — [`L180`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L180)
  - `dispatch_structural_tag(self, key_string: str)` — [`L192`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L192)
  - `any_whitespace` — [`L150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L150)
  - `llguidance_tokenizer` — [`L152`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L152)
  - `whitespace_pattern` — [`L151`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L151)
- protocol/private: `_from_serialized`[`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L156)
- uses (calls/refs, reference-scoped): [`INVALID_GRAMMAR_OBJ`](base_grammar_backend.md#INVALID_GRAMMAR_OBJ), [`GuidanceGrammar`](llguidance_backend.md#GuidanceGrammar), [`logger`](llguidance_backend.md#logger), [`BaseGrammarBackend`](base_grammar_backend.md#BaseGrammarBackend), [`__init__`](base_grammar_backend.md#BaseGrammarBackend.__init__)
- used by: [`get_guidance_backend`](llguidance_backend.md#get_guidance_backend), [`dispatch_ebnf`](base_grammar_backend.md#BaseGrammarBackend.dispatch_ebnf), [`dispatch_json`](base_grammar_backend.md#BaseGrammarBackend.dispatch_json), [`dispatch_regex`](base_grammar_backend.md#BaseGrammarBackend.dispatch_regex), [`dispatch_structural_tag`](base_grammar_backend.md#BaseGrammarBackend.dispatch_structural_tag)

### `GuidanceGrammar`  ·  implements/extends BaseGrammarObject
- def: [`python/sgl_jax/srt/constrained/llguidance_backend.py:23`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L23)
- doc: Grammar object using llguidance library.
- signature: `class GuidanceGrammar(BaseGrammarObject):`
- members:
  - `accept_token(self, token: int)` — [`L50`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L50)
  - `allocate_vocab_mask(self, vocab_size: int, batch_size: int)` — [`L61`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L61) — Allocate a vocabulary bitmask.
  - `copy(self)` — [`L87`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L87)
  - `fill_vocab_mask(self, vocab_mask: np.ndarray, idx: int)` — [`L70`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L70) — Fill the vocabulary bitmask for this grammar at batch index.
  - `is_terminated(self)` — [`L84`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L84)
  - `bitmask` — [`L44`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L44)
  - `eos_token` — [`L34`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L34)
  - `finished` — [`L54`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L54)
  - `ll_matcher` — [`L36`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L36)
  - `llguidance_tokenizer` — [`L32`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L32)
  - `serialized_grammar` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L33)
- protocol/private: `__init__`[`L26`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L26), `_check_err`[`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L46)
- uses (calls/refs, reference-scoped): [`BaseGrammarObject`](base_grammar_backend.md#BaseGrammarObject), [`logger`](llguidance_backend.md#logger), [`allocate_token_bitmask`](bitmask_ops.md#allocate_token_bitmask), [`fill_token_bitmask`](bitmask_ops.md#fill_token_bitmask), [`__init__`](base_grammar_backend.md#BaseGrammarObject.__init__)
- used by: [`dispatch_json`](llguidance_backend.md#GuidanceBackend.dispatch_json), [`_from_serialized`](llguidance_backend.md#GuidanceBackend._from_serialized), [`BaseGrammarObject`](base_grammar_backend.md#BaseGrammarObject), [`dispatch_ebnf`](llguidance_backend.md#GuidanceBackend.dispatch_ebnf), [`dispatch_structural_tag`](llguidance_backend.md#GuidanceBackend.dispatch_structural_tag), [`dispatch_regex`](llguidance_backend.md#GuidanceBackend.dispatch_regex), [`accept_token`](base_grammar_backend.md#BaseGrammarObject.accept_token), [`allocate_vocab_mask`](base_grammar_backend.md#BaseGrammarObject.allocate_vocab_mask), [`copy`](base_grammar_backend.md#BaseGrammarObject.copy), [`fill_vocab_mask`](base_grammar_backend.md#BaseGrammarObject.fill_vocab_mask), [`is_terminated`](base_grammar_backend.md#BaseGrammarObject.is_terminated)

## Functions
- `get_guidance_backend(tokenizer, any_whitespace: bool = True, whitespace_pattern: str | None = None, n_vocab: int = 0, num_threads: int = 4)` — [`L94`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L94)

## Module values
- `logger` — [`L20`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/llguidance_backend.py#L20)

