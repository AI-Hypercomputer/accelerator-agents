---
title: 'Module: easydel/data/transforms/pack.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/pack.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.pack`/
symbols:
  PackStage.process: PackStage#process().
  PackedShardedSource.open_shard: PackedShardedSource#open_shard().
  GreedyPacker.flush_final: GreedyPacker#flush_final().
  GreedyPacker._flush: GreedyPacker#_flush().
  FirstFitPacker._pack_buffer: FirstFitPacker#_pack_buffer().
  GreedyPacker.add: GreedyPacker#add().
  PackedShardedSource._create_packer: PackedShardedSource#_create_packer().
  PackedSequence: PackedSequence#
  GreedyPacker._buffer: GreedyPacker#_buffer.
  PoolPacker.add: PoolPacker#add().
  PackStage._stage_config: PackStage#_stage_config.
  PackedShardedSource.emit: PackedShardedSource#emit().
  PackedShardedSource: PackedShardedSource#
  FirstFitPacker.add: FirstFitPacker#add().
  pack_constant_length: pack_constant_length().
  PackedSequence.to_dict: PackedSequence#to_dict().
  PoolPacker.flush_all: PoolPacker#flush_all().
  GreedyPacker.seq_length: GreedyPacker#seq_length.
  GreedyPacker._current_segment: GreedyPacker#_current_segment.
  PackedShardedSource.__repr__: PackedShardedSource#__repr__().
  pack_pre_tokenized: pack_pre_tokenized().
  GreedyPacker._segment_ids: GreedyPacker#_segment_ids.
  GreedyPacker._source_ids: GreedyPacker#_source_ids.
  PackStage: PackStage#
  PoolPacker._packers: PoolPacker#_packers.
  PackedSequence.segment_ids: PackedSequence#segment_ids.
  GreedyPacker.include_segment_ids: GreedyPacker#include_segment_ids.
  PoolPacker: PoolPacker#
  FirstFitPacker: FirstFitPacker#
  FirstFitPacker._pending: FirstFitPacker#_pending.
  FirstFitPacker.flush_all: FirstFitPacker#flush_all().
  PackStage.__init__: PackStage#__init__().
  PackedSequence.input_ids: PackedSequence#input_ids.
  PackedSequence.attention_mask: PackedSequence#attention_mask.
  GreedyPacker: GreedyPacker#
  PackedShardedSource._source: PackedShardedSource#_source.
  PackedShardedSource._seq_length: PackedShardedSource#_seq_length.
  PackedSequence.source_ids: PackedSequence#source_ids.
  PackedSequence.num_segments: PackedSequence#num_segments.
  FirstFitPacker.seq_length: FirstFitPacker#seq_length.
  PackedShardedSource._eos_token_id: PackedShardedSource#_eos_token_id.
  PackedShardedSource._pad_token_id: PackedShardedSource#_pad_token_id.
  PackedShardedSource._strategy: PackedShardedSource#_strategy.
  PackedShardedSource._include_segment_ids: PackedShardedSource#_include_segment_ids.
  PackedShardedSource.__init__: PackedShardedSource#__init__().
  PackedShardedSource.__len__: PackedShardedSource#__len__().
  FirstFitPacker.eos_token_id: FirstFitPacker#eos_token_id.
  PackedShardedSource._shuffle: PackedShardedSource#_shuffle.
  PackedShardedSource._seed: PackedShardedSource#_seed.
  logger: logger.
  GreedyPacker.eos_token_id: GreedyPacker#eos_token_id.
  GreedyPacker.pad_token_id: GreedyPacker#pad_token_id.
  PoolPacker.seq_length: PoolPacker#seq_length.
  FirstFitPacker.pad_token_id: FirstFitPacker#pad_token_id.
  FirstFitPacker.include_segment_ids: FirstFitPacker#include_segment_ids.
  FirstFitPacker.buffer_size: FirstFitPacker#buffer_size.
  PackedShardedSource._num_packers: PackedShardedSource#_num_packers.
  PackedShardedSource._input_field: PackedShardedSource#_input_field.
  PackedShardedSource._shuffle_buffer_factor: PackedShardedSource#_shuffle_buffer_factor.
  pack_pre_tokenized.gen: pack_pre_tokenized().gen().
  pack_constant_length.token_iter: pack_constant_length().token_iter().
  GreedyPacker.__init__: GreedyPacker#__init__().
  PoolPacker.__init__: PoolPacker#__init__().
  PoolPacker.num_packers: PoolPacker#num_packers.
  FirstFitPacker.__init__: FirstFitPacker#__init__().
  PackedShardedSource.shard_names: PackedShardedSource#shard_names().
  PackedShardedSource.num_shards: PackedShardedSource#num_shards().
  PackStage.name: PackStage#name().
---
# Module: [`easydel/data/transforms/pack.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py)

## Classes
### `FirstFitPacker`
- def: [`easydel/data/transforms/pack.py:288`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L288)
- doc: First-fit decreasing bin-packing packer.
- signature: `class FirstFitPacker:`
- members:
  - `__init__(self, seq_length: int, eos_token_id: int, pad_token_id: int = 0, include_segment_ids: bool = True, buffer_size: int = 1000)` — [`L295`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L295) — Initialize FirstFitPacker.
  - `_pack_buffer(self)` — [`L337`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L337) — Pack the pending buffer using first-fit decreasing.
  - `add(self, tokens: list[int], source_id: str | None = None)` — [`L320`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L320) — Add tokens to the pending buffer.
  - `flush_all(self)` — [`L401`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L401) — Flush remaining pending sequences through first-fit packing.
  - `buffer_size` — [`L316`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L316)
  - `eos_token_id` — [`L313`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L313)
  - `include_segment_ids` — [`L315`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L315)
  - `pad_token_id` — [`L314`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L314)
  - `seq_length` — [`L312`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L312)
- protocol/private: `_pending`[`L318`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L318)
- uses (calls/refs, reference-scoped): [`PackedSequence`](pack.md#PackedSequence), [`segment_ids`](pack.md#PackedSequence.segment_ids), [`attention_mask`](pack.md#PackedSequence.attention_mask), [`input_ids`](pack.md#PackedSequence.input_ids), [`num_segments`](pack.md#PackedSequence.num_segments), [`source_ids`](pack.md#PackedSequence.source_ids)
- used by: [`open_shard`](pack.md#PackedShardedSource.open_shard), [`_create_packer`](pack.md#PackedShardedSource._create_packer)

### `GreedyPacker`
- def: [`easydel/data/transforms/pack.py:77`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L77)
- doc: Simple greedy packer that concatenates sequences.
- signature: `class GreedyPacker:`
- members:
  - `__init__(self, seq_length: int, eos_token_id: int, pad_token_id: int = 0, include_segment_ids: bool = True)` — [`L84`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L84) — Initialize GreedyPacker.
  - `_flush(self)` — [`L147`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L147) — Create a packed sequence from the current buffer.
  - `add(self, tokens: list[int], source_id: str | None = None)` — [`L110`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L110) — Add tokens to the packer.
  - `flush_final(self)` — [`L172`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L172) — Flush any remaining tokens in the buffer with padding.
  - `eos_token_id` — [`L100`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L100)
  - `include_segment_ids` — [`L102`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L102)
  - `pad_token_id` — [`L101`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L101)
  - `seq_length` — [`L99`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L99)
- protocol/private: `_buffer`[`L105`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L105), `_current_segment`[`L107`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L107), `_segment_ids`[`L106`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L106), `_source_ids`[`L108`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L108)
- uses (calls/refs, reference-scoped): [`PackedSequence`](pack.md#PackedSequence), [`segment_ids`](pack.md#PackedSequence.segment_ids), [`attention_mask`](pack.md#PackedSequence.attention_mask), [`input_ids`](pack.md#PackedSequence.input_ids), [`num_segments`](pack.md#PackedSequence.num_segments), [`source_ids`](pack.md#PackedSequence.source_ids)
- used by: [`open_shard`](pack.md#PackedShardedSource.open_shard), [`_create_packer`](pack.md#PackedShardedSource._create_packer), [`add`](pack.md#PoolPacker.add), [`flush_all`](pack.md#PoolPacker.flush_all), [`_packers`](pack.md#PoolPacker._packers)

### `PackStage`  ·  implements/extends BaseStage
- def: [`easydel/data/transforms/pack.py:586`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L586)
- doc: Pipeline stage for packing tokenized sequences into fixed-length chunks.
- signature: `class PackStage(BaseStage):`
- members:
  - `__init__(self, config: PackStageConfig | None = None)` — [`L594`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L594) — Initialize PackStage.
  - `name(self)` — [`L604`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L604)
  - `process(self, data: dict[str, ShardedDataSource], context: PipelineContext)` — [`L607`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L607) — Pack sequences in all datasets.
- protocol/private: `_stage_config`[`L601`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L601)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`BaseStage`](../core/protocols.md#BaseStage), [`PipelineContext`](../core/protocols.md#PipelineContext), [`PackedShardedSource`](pack.md#PackedShardedSource), [`PackStageConfig`](../core/config.md#PackStageConfig), [`__init__`](../core/protocols.md#BaseStage.__init__), [`seed`](../core/protocols.md#PipelineContext.seed), [`strategy`](../core/config.md#PackStageConfig.strategy), [`enabled`](../core/config.md#PackStageConfig.enabled), [`eos_token_id`](../core/config.md#PackStageConfig.eos_token_id), [`include_segment_ids`](../core/config.md#PackStageConfig.include_segment_ids), [`logger`](pack.md#logger), [`num_packers`](../core/config.md#PackStageConfig.num_packers), [`pad_token_id`](../core/config.md#PackStageConfig.pad_token_id), [`seq_length`](../core/config.md#PackStageConfig.seq_length), [`shuffle_buffer_factor`](../core/config.md#PackStageConfig.shuffle_buffer_factor), [`shuffle_packed`](../core/config.md#PackStageConfig.shuffle_packed)
- used by: [`pack`](../execution/pipeline.md#Pipeline.pack), [`BaseStage`](../core/protocols.md#BaseStage), [`process`](../core/protocols.md#BaseStage.process), [`name`](../core/protocols.md#BaseStage.name)

### `PackedSequence`
- def: [`easydel/data/transforms/pack.py:44`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L44)
- doc: A packed sequence combining multiple examples with metadata.
- signature: `class PackedSequence:`
- members:
  - `to_dict(self)` — [`L62`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L62) — Convert to a dictionary suitable for training loops.
  - `attention_mask` — [`L57`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L57)
  - `input_ids` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L56)
  - `num_segments` — [`L60`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L60)
  - `segment_ids` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L58)
  - `source_ids` — [`L59`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L59)
- used by: [`flush_final`](pack.md#GreedyPacker.flush_final), [`_flush`](pack.md#GreedyPacker._flush), [`_pack_buffer`](pack.md#FirstFitPacker._pack_buffer), [`add`](pack.md#GreedyPacker.add), [`add`](pack.md#PoolPacker.add), [`emit`](pack.md#PackedShardedSource.emit), [`add`](pack.md#FirstFitPacker.add), [`flush_all`](pack.md#PoolPacker.flush_all), [`flush_all`](pack.md#FirstFitPacker.flush_all)

### `PackedShardedSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/transforms/pack.py:410`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L410)
- doc: Sharded source that packs sequences from another source.
- signature: `class PackedShardedSource(ShardedDataSource[dict]):`
- members:
  - `__init__(self, source: ShardedDataSource[dict], seq_length: int, eos_token_id: int, pad_token_id: int = 0, strategy: str = "greedy", num_packers: int = 4, include_segment_ids: bool = True, input_field: str = "input_ids", shuffle: bool = True, shuffle_buffer_factor: int = 10, seed: int | None = None)` — [`L419`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L419) — Initialize PackedShardedSource.
  - `__len__(self)` — [`L565`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L565) — Return estimated number of packed sequences.
  - `_create_packer(self)` — [`L467`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L467) — Create a packer based on strategy.
  - `emit(packed: PackedSequence)` — [`L511`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L511) — Emit a packed sequence, handling shuffle.
  - `num_shards(self)` — [`L464`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L464)
  - `open_shard(self, shard_name: str)` — [`L492`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L492) — Open the packed shard and iterate over packed sequences.
  - `shard_names(self)` — [`L461`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L461)
- protocol/private: `__repr__`[`L580`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L580), `_eos_token_id`[`L450`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L450), `_include_segment_ids`[`L454`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L454), `_input_field`[`L455`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L455), `_num_packers`[`L453`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L453), `_pad_token_id`[`L451`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L451), `_seed`[`L458`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L458), `_seq_length`[`L449`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L449), `_shuffle`[`L456`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L456), `_shuffle_buffer_factor`[`L457`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L457), `_source`[`L448`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L448), `_strategy`[`L452`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L452)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`flush_final`](pack.md#GreedyPacker.flush_final), [`add`](pack.md#GreedyPacker.add), [`PackedSequence`](pack.md#PackedSequence), [`add`](pack.md#PoolPacker.add), [`flush_all`](pack.md#PoolPacker.flush_all), [`to_dict`](pack.md#PackedSequence.to_dict), [`FirstFitPacker`](pack.md#FirstFitPacker), [`PoolPacker`](pack.md#PoolPacker), [`GreedyPacker`](pack.md#GreedyPacker)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`process`](pack.md#PackStage.process), [`build_sharded_source`](../../infra/elarge/builders.md#build_sharded_source), [`_apply_preprocess_transforms`](../../trainers/supervised_fine_tuning_trainer/sft_trainer.md#SFTTrainer._apply_preprocess_transforms)

### `PoolPacker`
- def: [`easydel/data/transforms/pack.py:212`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L212)
- doc: Pool of packers for more efficient bin-packing.
- signature: `class PoolPacker:`
- members:
  - `__init__(self, seq_length: int, eos_token_id: int, pad_token_id: int = 0, num_packers: int = 4, include_segment_ids: bool = True)` — [`L219`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L219) — Initialize PoolPacker.
  - `add(self, tokens: list[int], source_id: str | None = None)` — [`L242`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L242) — Add tokens to the best-fit packer.
  - `flush_all(self)` — [`L274`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L274) — Flush all packers in the pool, returning their remaining sequences.
  - `num_packers` — [`L237`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L237)
  - `seq_length` — [`L236`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L236)
- protocol/private: `_packers`[`L238`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L238)
- uses (calls/refs, reference-scoped): [`flush_final`](pack.md#GreedyPacker.flush_final), [`add`](pack.md#GreedyPacker.add), [`PackedSequence`](pack.md#PackedSequence), [`_buffer`](pack.md#GreedyPacker._buffer), [`GreedyPacker`](pack.md#GreedyPacker)
- used by: [`open_shard`](pack.md#PackedShardedSource.open_shard), [`_create_packer`](pack.md#PackedShardedSource._create_packer)

## Functions
- `gen()` — [`L663`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L663)
- `pack_constant_length(stream, tokenize_fn, seq_length: int, eos_token_id: int, batch_size: int, shuffle: bool, buffer_factor: int)` — [`L695`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L695) — Pack sequences with on-the-fly tokenization into constant-length chunks.
- `pack_pre_tokenized(stream, seq_length: int, eos_token_id: int, batch_size: int, shuffle: bool, buffer_factor: int)` — [`L644`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L644) — Pack pre-tokenized sequences into constant-length chunks.
- `token_iter()` — [`L723`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L723)

## Module values
- `logger` — [`L40`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/pack.py#L40)

