---
title: 'Module: python/sgl_jax/srt/constrained/base_grammar_backend.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/constrained/base_grammar_backend.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.constrained.base_grammar_backend`/
symbols:
  create_grammar_backend: create_grammar_backend().
  BaseGrammarBackend._dispatch: BaseGrammarBackend#_dispatch().
  BaseGrammarObject: BaseGrammarObject#
  INVALID_GRAMMAR_OBJ: INVALID_GRAMMAR_OBJ.
  BaseGrammarBackend.get_cached_or_future_value: BaseGrammarBackend#get_cached_or_future_value().
  BaseGrammarBackend.set_cache: BaseGrammarBackend#set_cache().
  BaseGrammarBackend.dispatch_json: BaseGrammarBackend#dispatch_json().
  BaseGrammarBackend.dispatch_regex: BaseGrammarBackend#dispatch_regex().
  BaseGrammarBackend.dispatch_ebnf: BaseGrammarBackend#dispatch_ebnf().
  BaseGrammarBackend.dispatch_structural_tag: BaseGrammarBackend#dispatch_structural_tag().
  BaseGrammarBackend.cache: BaseGrammarBackend#cache.
  BaseGrammarBackend.reset: BaseGrammarBackend#reset().
  BaseGrammarBackend: BaseGrammarBackend#
  BaseGrammarObject.accept_token: BaseGrammarObject#accept_token().
  BaseGrammarObject.allocate_vocab_mask: BaseGrammarObject#allocate_vocab_mask().
  BaseGrammarObject.fill_vocab_mask: BaseGrammarObject#fill_vocab_mask().
  BaseGrammarObject.is_terminated: BaseGrammarObject#is_terminated().
  BaseGrammarObject.copy: BaseGrammarObject#copy().
  BaseGrammarBackend.shutdown: BaseGrammarBackend#shutdown().
  BaseGrammarBackend.executor: BaseGrammarBackend#executor.
  BaseGrammarObject.__init__: BaseGrammarObject#__init__().
  BaseGrammarBackend.__init__: BaseGrammarBackend#__init__().
  logger: logger.
  BaseGrammarObject.finished: BaseGrammarObject#finished.
---
# Module: [`python/sgl_jax/srt/constrained/base_grammar_backend.py`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py)

## Classes
### `BaseGrammarBackend`
- def: [`python/sgl_jax/srt/constrained/base_grammar_backend.py:37`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L37)
- doc: Base class for grammar backends with async compilation support.
- signature: `class BaseGrammarBackend:`
- members:
  - `__init__(self, num_threads: int = 4)` — [`L40`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L40) — Initialize the grammar backend.
  - `_dispatch(self, key_type: str, key_string: str)` — [`L85`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L85) — Dispatch grammar creation based on type.
  - `dispatch_ebnf(self, key_string: str)` — [`L128`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L128) — Create a grammar from EBNF definition.
  - `dispatch_json(self, key_string: str)` — [`L106`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L106) — Create a grammar from JSON schema.
  - `dispatch_regex(self, key_string: str)` — [`L117`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L117) — Create a grammar from regex pattern.
  - `dispatch_structural_tag(self, key_string: str)` — [`L139`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L139) — Create a grammar from structural tag configuration.
  - `get_cached_or_future_value(self, key: tuple[str, str])` — [`L49`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L49) — Get a cached grammar object or submit async compilation.
  - `reset(self)` — [`L82`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L82)
  - `set_cache(self, key: tuple[str, str], value: BaseGrammarObject)` — [`L73`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L73) — Store a compiled grammar in the cache.
  - `shutdown(self)` — [`L150`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L150) — Shutdown the thread pool executor.
  - `cache` — [`L47`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L47)
  - `executor` — [`L46`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L46)
- uses (calls/refs, reference-scoped): [`dispatch_json`](llguidance_backend.md#GuidanceBackend.dispatch_json), [`BaseGrammarObject`](base_grammar_backend.md#BaseGrammarObject), [`dispatch_ebnf`](llguidance_backend.md#GuidanceBackend.dispatch_ebnf), [`dispatch_structural_tag`](llguidance_backend.md#GuidanceBackend.dispatch_structural_tag), [`dispatch_regex`](llguidance_backend.md#GuidanceBackend.dispatch_regex)
- used by: [`handle_generate_request`](../managers/scheduler.md#Scheduler.handle_generate_request), [`flush_cache`](../managers/scheduler.md#Scheduler.flush_cache), [`move_ready_grammar_requests`](../managers/scheduler.md#Scheduler.move_ready_grammar_requests), [`create_grammar_backend`](base_grammar_backend.md#create_grammar_backend), [`GuidanceBackend`](llguidance_backend.md#GuidanceBackend), [`__init__`](llguidance_backend.md#GuidanceBackend.__init__)

### `BaseGrammarObject`
- def: [`python/sgl_jax/srt/constrained/base_grammar_backend.py:15`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L15)
- doc: Base class for grammar objects that maintain state during generation.
- signature: `class BaseGrammarObject:`
- members:
  - `accept_token(self, token: int)` — [`L21`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L21)
  - `allocate_vocab_mask(self, vocab_size: int, batch_size: int)` — [`L24`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L24)
  - `copy(self)` — [`L33`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L33)
  - `fill_vocab_mask(self, vocab_mask: np.ndarray, idx: int)` — [`L27`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L27)
  - `is_terminated(self)` — [`L30`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L30)
  - `finished` — [`L19`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L19)
- protocol/private: `__init__`[`L18`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L18)
- uses (calls/refs, reference-scoped): [`fill_vocab_mask`](llguidance_backend.md#GuidanceGrammar.fill_vocab_mask), [`GuidanceGrammar`](llguidance_backend.md#GuidanceGrammar), [`accept_token`](llguidance_backend.md#GuidanceGrammar.accept_token), [`copy`](llguidance_backend.md#GuidanceGrammar.copy), [`allocate_vocab_mask`](llguidance_backend.md#GuidanceGrammar.allocate_vocab_mask), [`is_terminated`](llguidance_backend.md#GuidanceGrammar.is_terminated)
- used by: [`_dispatch`](base_grammar_backend.md#BaseGrammarBackend._dispatch), [`INVALID_GRAMMAR_OBJ`](base_grammar_backend.md#INVALID_GRAMMAR_OBJ), [`GuidanceGrammar`](llguidance_backend.md#GuidanceGrammar), [`set_cache`](base_grammar_backend.md#BaseGrammarBackend.set_cache), [`dispatch_ebnf`](base_grammar_backend.md#BaseGrammarBackend.dispatch_ebnf), [`dispatch_json`](base_grammar_backend.md#BaseGrammarBackend.dispatch_json), [`dispatch_regex`](base_grammar_backend.md#BaseGrammarBackend.dispatch_regex), [`dispatch_structural_tag`](base_grammar_backend.md#BaseGrammarBackend.dispatch_structural_tag), [`__init__`](llguidance_backend.md#GuidanceGrammar.__init__)

## Functions
- `create_grammar_backend(server_args: ServerArgs, tokenizer, vocab_size: int, eos_token_ids: set | None = None)` — [`L159`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L159)

## Module values
- `INVALID_GRAMMAR_OBJ` — [`L156`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L156)
- `logger` — [`L12`](../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/constrained/base_grammar_backend.py#L12)

