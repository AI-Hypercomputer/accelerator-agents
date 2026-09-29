---
title: 'Module: prepare.py'
type: catalog
provenance: extracted
module: prepare.py
status: fresh
symbol_base: scip-python python autoresearch 0.0.0 prepare/
symbols:
  train_tokenizer: train_tokenizer().
  num_shards: num_shards.
  evaluate_bpb: evaluate_bpb().
  download_data: download_data().
  DATA_DIR: DATA_DIR.
  TOKENIZER_DIR: TOKENIZER_DIR.
  _document_batches: _document_batches().
  make_dataloader: make_dataloader().
  MAX_SEQ_LEN: MAX_SEQ_LEN.
  download_single_shard: download_single_shard().
  text_iterator: text_iterator().
  VAL_SHARD: VAL_SHARD.
  list_parquet_files: list_parquet_files().
  parser: parser.
  args: args.
  TIME_BUDGET: TIME_BUDGET.
  Tokenizer.enc: Tokenizer#enc.
  VAL_FILENAME: VAL_FILENAME.
  Tokenizer.from_directory: Tokenizer#from_directory().
  Tokenizer.get_vocab_size: Tokenizer#get_vocab_size().
  get_token_bytes: get_token_bytes().
  CACHE_DIR: CACHE_DIR.
  MAX_SHARD: MAX_SHARD.
  SPECIAL_TOKENS: SPECIAL_TOKENS.
  Tokenizer.__init__: Tokenizer#__init__().
  Tokenizer.get_bos_token_id: Tokenizer#get_bos_token_id().
  Tokenizer.encode: Tokenizer#encode().
  Tokenizer.decode: Tokenizer#decode().
  Tokenizer: Tokenizer#
  EVAL_TOKENS: EVAL_TOKENS.
  BASE_URL: BASE_URL.
  VOCAB_SIZE: VOCAB_SIZE.
  SPLIT_PATTERN: SPLIT_PATTERN.
  BOS_TOKEN: BOS_TOKEN.
  Tokenizer.bos_token_id: Tokenizer#bos_token_id.
  make_dataloader.refill_buffer: make_dataloader().refill_buffer().
---
# Module: [`prepare.py`](../../../../raw/code/autoresearch/prepare.py)

## Classes
### `Tokenizer`
- def: [`prepare.py:209`](../../../../raw/code/autoresearch/prepare.py#L209)
- doc: Minimal tokenizer wrapper. Training is handled above.
- signature: `class Tokenizer:`
- members:
  - `decode(self, ids)` — [`L244`](../../../../raw/code/autoresearch/prepare.py#L244)
  - `encode(self, text, prepend=None, num_threads=8)` — [`L228`](../../../../raw/code/autoresearch/prepare.py#L228)
  - `from_directory(cls, tokenizer_dir=TOKENIZER_DIR)` — [`L217`](../../../../raw/code/autoresearch/prepare.py#L217)
  - `get_bos_token_id(self)` — [`L225`](../../../../raw/code/autoresearch/prepare.py#L225)
  - `get_vocab_size(self)` — [`L222`](../../../../raw/code/autoresearch/prepare.py#L222)
  - `bos_token_id` — [`L214`](../../../../raw/code/autoresearch/prepare.py#L214)
  - `enc` — [`L213`](../../../../raw/code/autoresearch/prepare.py#L213)
- protocol/private: `__init__`[`L212`](../../../../raw/code/autoresearch/prepare.py#L212)
- uses (calls/refs, reference-scoped): [`TOKENIZER_DIR`](prepare.md#TOKENIZER_DIR), [`BOS_TOKEN`](prepare.md#BOS_TOKEN)
- used by: [`fa3`](train.md#fa3), [`vocab_size`](train.md#vocab_size), [`tokenizer`](train.md#tokenizer)

## Functions
- `_document_batches(split, tokenizer_batch_size=128)` — [`L254`](../../../../raw/code/autoresearch/prepare.py#L254) — Infinite iterator over document batches from parquet files. — documented in [prepare](../concepts/prepare.md)
- `download_data(num_shards, download_workers=8)` — [`L91`](../../../../raw/code/autoresearch/prepare.py#L91) — Download training shards + pinned validation shard. — documented in [prepare](../concepts/prepare.md)
- `download_single_shard(index)` — [`L57`](../../../../raw/code/autoresearch/prepare.py#L57) — Download one parquet shard with retries. Returns True on success. — documented in [prepare](../concepts/prepare.md)
- `evaluate_bpb(model, tokenizer, batch_size)` — [`L344`](../../../../raw/code/autoresearch/prepare.py#L344) — Bits per byte (BPB): vocab size-independent evaluation metric. — documented in [prepare](../concepts/prepare.md)
- `get_token_bytes(device="cpu")` — [`L248`](../../../../raw/code/autoresearch/prepare.py#L248) — documented in [prepare](../concepts/prepare.md)
- `list_parquet_files()` — [`L119`](../../../../raw/code/autoresearch/prepare.py#L119) — Return sorted list of parquet file paths in the data directory. — documented in [prepare](../concepts/prepare.md)
- `make_dataloader(tokenizer, B, T, split, buffer_size=1000)` — [`L276`](../../../../raw/code/autoresearch/prepare.py#L276) — BOS-aligned dataloader with best-fit packing. — documented in [prepare](../concepts/prepare.md)
- `refill_buffer()` — [`L290`](../../../../raw/code/autoresearch/prepare.py#L290) — documented in [prepare](../concepts/prepare.md)
- `text_iterator(max_chars=1000000000, doc_cap=10000)` — [`L125`](../../../../raw/code/autoresearch/prepare.py#L125) — Yield documents from training split (all shards except pinned val shard). — documented in [prepare](../concepts/prepare.md)
- `train_tokenizer()` — [`L141`](../../../../raw/code/autoresearch/prepare.py#L141) — Train BPE tokenizer using rustbpe, save as tiktoken pickle. — documented in [prepare](../concepts/prepare.md)

## Module values
- `BASE_URL` — [`L41`](../../../../raw/code/autoresearch/prepare.py#L41) — documented in [prepare](../concepts/prepare.md)
- `BOS_TOKEN` — [`L51`](../../../../raw/code/autoresearch/prepare.py#L51)
- `CACHE_DIR` — [`L38`](../../../../raw/code/autoresearch/prepare.py#L38) — documented in [prepare](../concepts/prepare.md)
- `DATA_DIR` — [`L39`](../../../../raw/code/autoresearch/prepare.py#L39) — documented in [prepare](../concepts/prepare.md)
- `EVAL_TOKENS` — [`L32`](../../../../raw/code/autoresearch/prepare.py#L32) — documented in [prepare](../concepts/prepare.md)
- `MAX_SEQ_LEN` — [`L30`](../../../../raw/code/autoresearch/prepare.py#L30) — documented in [prepare](../concepts/prepare.md)
- `MAX_SHARD` — [`L42`](../../../../raw/code/autoresearch/prepare.py#L42) — documented in [prepare](../concepts/prepare.md)
- `SPECIAL_TOKENS` — [`L50`](../../../../raw/code/autoresearch/prepare.py#L50) — documented in [prepare](../concepts/prepare.md)
- `SPLIT_PATTERN` — [`L48`](../../../../raw/code/autoresearch/prepare.py#L48) — documented in [prepare](../concepts/prepare.md)
- `TIME_BUDGET` — [`L31`](../../../../raw/code/autoresearch/prepare.py#L31)
- `TOKENIZER_DIR` — [`L40`](../../../../raw/code/autoresearch/prepare.py#L40) — documented in [prepare](../concepts/prepare.md)
- `VAL_FILENAME` — [`L44`](../../../../raw/code/autoresearch/prepare.py#L44) — documented in [prepare](../concepts/prepare.md)
- `VAL_SHARD` — [`L43`](../../../../raw/code/autoresearch/prepare.py#L43) — documented in [prepare](../concepts/prepare.md)
- `VOCAB_SIZE` — [`L45`](../../../../raw/code/autoresearch/prepare.py#L45) — documented in [prepare](../concepts/prepare.md)
- `args` — [`L375`](../../../../raw/code/autoresearch/prepare.py#L375) — documented in [prepare](../concepts/prepare.md)
- `num_shards` — [`L377`](../../../../raw/code/autoresearch/prepare.py#L377) — documented in [prepare](../concepts/prepare.md)
- `parser` — [`L372`](../../../../raw/code/autoresearch/prepare.py#L372) — documented in [prepare](../concepts/prepare.md)

