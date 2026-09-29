---
title: 'Module: test/gemma/tokenizer.py'
type: catalog
provenance: extracted
module: test/gemma/tokenizer.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `test.gemma.tokenizer`/Tokenizer#
symbols:
  Tokenizer.sp_model: sp_model.
  Tokenizer.encode: encode().
  Tokenizer.bos_id: bos_id.
  Tokenizer.eos_id: eos_id.
  Tokenizer.n_words: n_words.
  Tokenizer.pad_id: pad_id.
  Tokenizer.decode: decode().
  Tokenizer: ''
  Tokenizer.__init__: __init__().
---
# Module: [`test/gemma/tokenizer.py`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py)

## Classes
### `Tokenizer`
- def: [`test/gemma/tokenizer.py:21`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L21)
- signature: `class Tokenizer:`
- members:
  - `decode(self, t: list[int])` — [`L44`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L44) — Converts a list of tokens into a string.
  - `encode(self, s: str, bos: bool = True, eos: bool = False)` — [`L34`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L34) — Converts a string into a list of tokens.
  - `bos_id` — [`L29`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L29)
  - `eos_id` — [`L30`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L30)
  - `n_words` — [`L28`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L28)
  - `pad_id` — [`L31`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L31)
  - `sp_model` — [`L25`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L25)
- protocol/private: `__init__`[`L22`](../../../../../../raw/code/torchax/test/gemma/tokenizer.py#L22)

