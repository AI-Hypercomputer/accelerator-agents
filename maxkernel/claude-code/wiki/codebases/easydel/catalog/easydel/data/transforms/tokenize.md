---
title: 'Module: easydel/data/transforms/tokenize.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/tokenize.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.tokenize`/
symbols:
  TokenizeStage.process: TokenizeStage#process().
  tokenize_dataset_config: tokenize_dataset_config().
  TokenizedShardedSource._tokenize_example: TokenizedShardedSource#_tokenize_example().
  TokenizerManager.tokenize_text: TokenizerManager#tokenize_text().
  TokenizerManager.tokenize_batch: TokenizerManager#tokenize_batch().
  batched_tokenize_iterator: batched_tokenize_iterator().
  TokenizerManager.get_tokenizer: TokenizerManager#get_tokenizer().
  TokenizedShardedSource.__repr__: TokenizedShardedSource#__repr__().
  TokenizerManager._make_cache_key: TokenizerManager#_make_cache_key().
  TokenizedShardedSource.open_shard: TokenizedShardedSource#open_shard().
  TokenizedShardedSource.open_shard_at_row: TokenizedShardedSource#open_shard_at_row().
  TokenizedShardedSource: TokenizedShardedSource#
  TokenizeStage: TokenizeStage#
  TokenizerManager: TokenizerManager#
  TokenizedShardedSource._source: TokenizedShardedSource#_source.
  TokenizedShardedSource._transform_example: TokenizedShardedSource#_transform_example().
  TokenizedShardedSource.__init__: TokenizedShardedSource#__init__().
  TokenizedShardedSource.shard_names: TokenizedShardedSource#shard_names().
  TokenizedShardedSource.num_shards: TokenizedShardedSource#num_shards().
  TokenizeStage.__init__: TokenizeStage#__init__().
  batched_tokenize_iterator.flush_batch: batched_tokenize_iterator().flush_batch().
  logger: logger.
  TokenizerManager._cache: TokenizerManager#_cache.
  TokenizedShardedSource._manager: TokenizedShardedSource#_manager.
  TokenizeStage._stage_config: TokenizeStage#_stage_config.
  TokenizeStage._tokenizer_manager: TokenizeStage#_tokenizer_manager.
  TokenizedShardedSource._tokenizer_config: TokenizedShardedSource#_tokenizer_config.
  TokenizedShardedSource._content_field: TokenizedShardedSource#_content_field.
  TokenizerManager.__init__: TokenizerManager#__init__().
  TokenizedShardedSource.__len__: TokenizedShardedSource#__len__().
  TokenizedShardedSource._format_callback: TokenizedShardedSource#_format_callback.
  TokenizedShardedSource._format_fields: TokenizedShardedSource#_format_fields.
  TokenizedShardedSource._tokenizer: TokenizedShardedSource#_tokenizer.
  TokenizedShardedSource._additional_fields: TokenizedShardedSource#_additional_fields.
  compute_tokenizer_hash: compute_tokenizer_hash().
  TokenizeStage.name: TokenizeStage#name().
---
# Module: [`easydel/data/transforms/tokenize.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py)

## Classes
### `TokenizeStage`  ·  implements/extends BaseStage
- def: [`easydel/data/transforms/tokenize.py:326`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L326)
- doc: Pipeline stage for tokenizing text data in each dataset.
- signature: `class TokenizeStage(BaseStage):`
- members:
  - `__init__(self, config: TokenizeStageConfig | None = None)` — [`L335`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L335) — Initialize TokenizeStage.
  - `name(self)` — [`L346`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L346)
  - `process(self, data: dict[str, ShardedDataSource], context: PipelineContext)` — [`L349`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L349) — Process datasets through tokenization.
- protocol/private: `_stage_config`[`L342`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L342), `_tokenizer_manager`[`L343`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L343)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`BaseStage`](../core/protocols.md#BaseStage), [`PipelineContext`](../core/protocols.md#PipelineContext), [`merge_tokenizer_config`](../core/config.md#merge_tokenizer_config), [`get_tokenizer`](tokenize.md#TokenizerManager.get_tokenizer), [`TokenizeStageConfig`](../core/config.md#TokenizeStageConfig), [`get_dataset_by_name`](../core/config.md#PipelineConfig.get_dataset_by_name), [`config`](../core/protocols.md#PipelineContext.config), [`name_or_path`](../core/config.md#TokenizerConfig.name_or_path), [`TokenizedShardedSource`](tokenize.md#TokenizedShardedSource), [`_update_metric`](../core/protocols.md#BaseStage._update_metric), [`TokenizerManager`](tokenize.md#TokenizerManager), [`__init__`](../core/protocols.md#BaseStage.__init__), [`logger`](tokenize.md#logger), [`content_field`](../core/config.md#DatasetConfig.content_field), [`default_tokenizer`](../core/config.md#PipelineConfig.default_tokenizer), [`additional_fields`](../core/config.md#DatasetConfig.additional_fields), [`format_callback`](../core/config.md#DatasetConfig.format_callback), [`format_fields`](../core/config.md#DatasetConfig.format_fields), [`tokenizer_kwargs`](../core/config.md#DatasetConfig.tokenizer_kwargs)
- used by: [`tokenize`](../execution/pipeline.md#Pipeline.tokenize), [`BaseStage`](../core/protocols.md#BaseStage), [`process`](../core/protocols.md#BaseStage.process), [`name`](../core/protocols.md#BaseStage.name)

### `TokenizedShardedSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/transforms/tokenize.py:145`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L145)
- doc: Sharded source that wraps another source with on-the-fly tokenization.
- signature: `class TokenizedShardedSource(ShardedDataSource[dict]):`
- members:
  - `__init__(self, source: ShardedDataSource[dict], tokenizer: PreTrainedTokenizer, tokenizer_config: TokenizerConfig, content_field: str = "text", additional_fields: list[str] | None = None, format_callback: Callable[[dict], dict] | None = None, format_fields: dict[str, str] | None = None)` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L153) — Initialize TokenizedShardedSource.
  - `__len__(self)` — [`L255`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L255) — Return length of underlying source.
  - `_tokenize_example(self, example: dict)` — [`L204`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L204) — Tokenize a single example.
  - `_transform_example(self, example: dict)` — [`L190`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L190) — Apply format transformation to an example.
  - `num_shards(self)` — [`L187`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L187)
  - `open_shard(self, shard_name: str)` — [`L229`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L229) — Open a shard and tokenize examples on the fly.
  - `open_shard_at_row(self, shard_name: str, row: int)` — [`L242`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L242) — Open a shard at a specific row and tokenize from that position.
  - `shard_names(self)` — [`L184`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L184)
- protocol/private: `__repr__`[`L259`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L259), `_additional_fields`[`L178`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L178), `_content_field`[`L177`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L177), `_format_callback`[`L179`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L179), `_format_fields`[`L180`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L180), `_manager`[`L181`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L181), `_source`[`L174`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L174), `_tokenizer`[`L175`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L175), `_tokenizer_config`[`L176`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L176)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`TokenizerConfig`](../core/config.md#TokenizerConfig), [`tokenize_text`](tokenize.md#TokenizerManager.tokenize_text), [`TokenizerManager`](tokenize.md#TokenizerManager), [`logger`](tokenize.md#logger), [`max_length`](../core/config.md#TokenizerConfig.max_length)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`process`](tokenize.md#TokenizeStage.process), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`open_shard_at_row`](../core/protocols.md#ShardedDataSource.open_shard_at_row), [`tokenize_dataset_config`](tokenize.md#tokenize_dataset_config)

### `TokenizerManager`
- def: [`easydel/data/transforms/tokenize.py:43`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L43)
- doc: Manages tokenizers with caching and configuration.
- signature: `class TokenizerManager:`
- members:
  - `__init__(self)` — [`L52`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L52) — Initialize TokenizerManager with an empty cache.
  - `_make_cache_key(self, config: TokenizerConfig)` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L84) — Create a cache key from tokenizer config.
  - `get_tokenizer(self, config: TokenizerConfig, **extra_kwargs)` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L56) — Get or create a tokenizer from configuration.
  - `tokenize_batch(self, tokenizer: PreTrainedTokenizer, texts: list[str], config: TokenizerConfig)` — [`L118`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L118) — Tokenize a batch of texts.
  - `tokenize_text(self, tokenizer: PreTrainedTokenizer, text: str, config: TokenizerConfig)` — [`L92`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L92) — Tokenize a single text string.
- protocol/private: `_cache`[`L50`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L50)
- uses (calls/refs, reference-scoped): [`TokenizerConfig`](../core/config.md#TokenizerConfig), [`name_or_path`](../core/config.md#TokenizerConfig.name_or_path), [`max_length`](../core/config.md#TokenizerConfig.max_length), [`add_special_tokens`](../core/config.md#TokenizerConfig.add_special_tokens), [`padding`](../core/config.md#TokenizerConfig.padding), [`return_attention_mask`](../core/config.md#TokenizerConfig.return_attention_mask), [`truncation`](../core/config.md#TokenizerConfig.truncation), [`trust_remote_code`](../core/config.md#TokenizerConfig.trust_remote_code)
- used by: [`process`](tokenize.md#TokenizeStage.process), [`tokenize_dataset_config`](tokenize.md#tokenize_dataset_config), [`_tokenize_example`](tokenize.md#TokenizedShardedSource._tokenize_example), [`batched_tokenize_iterator`](tokenize.md#batched_tokenize_iterator), [`flush_batch`](tokenize.md#batched_tokenize_iterator.flush_batch), [`_manager`](tokenize.md#TokenizedShardedSource._manager), [`_tokenizer_manager`](tokenize.md#TokenizeStage._tokenizer_manager)

## Functions
- `batched_tokenize_iterator(source: ShardedDataSource[dict], tokenizer: PreTrainedTokenizer, tokenizer_config: TokenizerConfig, content_field: str = "text", batch_size: int = 1000, additional_fields: list[str] | None = None, format_callback: Callable[[dict], dict] | None = None)` — [`L264`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L264) — Iterate over a source with batched tokenization for efficiency.
- `compute_tokenizer_hash(tokenizer_name: str)` — [`L444`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L444) — Compute a hash for a tokenizer for cache invalidation.
- `flush_batch()` — [`L292`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L292) — Tokenize and yield the current batch.
- `tokenize_dataset_config(source: ShardedDataSource, config: DatasetConfig, global_tokenizer: str | None = None)` — [`L406`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L406) — Tokenize a source based on dataset configuration.

## Module values
- `logger` — [`L39`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/tokenize.py#L39)

