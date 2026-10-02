---
title: 'Module: easydel/data/transforms/mixture.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/mixture.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.mixture`/
symbols:
  MixStage.process: MixStage#process().
  MixedShardedSource.open_shard: MixedShardedSource#open_shard().
  WeightScheduler._schedule: WeightScheduler#_schedule.
  WeightScheduler.get_weights: WeightScheduler#get_weights().
  WeightScheduler._dataset_names: WeightScheduler#_dataset_names.
  MixStage._stage_config: MixStage#_stage_config.
  MixedShardedSource._chain_shards: MixedShardedSource#_chain_shards().
  MixedShardedSource._names: MixedShardedSource#_names.
  MixedShardedSource._get_weights_for_step: MixedShardedSource#_get_weights_for_step().
  MixedShardedSource._weights: MixedShardedSource#_weights.
  MixedShardedSource: MixedShardedSource#
  MixedShardedSource.__repr__: MixedShardedSource#__repr__().
  MixStage: MixStage#
  block_mixture_interleave: block_mixture_interleave().
  MixedShardedSource._compute_counts: MixedShardedSource#_compute_counts().
  MixedShardedSource.__init__: MixedShardedSource#__init__().
  MixStage.__init__: MixStage#__init__().
  MixedShardedSource._weight_scheduler: MixedShardedSource#_weight_scheduler.
  WeightScheduler: WeightScheduler#
  MixedShardedSource._sources: MixedShardedSource#_sources.
  MixedShardedSource._block_size: MixedShardedSource#_block_size.
  WeightScheduler.__init__: WeightScheduler#__init__().
  WeightScheduler.dataset_names: WeightScheduler#dataset_names().
  MixedShardedSource.__len__: MixedShardedSource#__len__().
  logger: logger.
  WeightScheduler._interpolation: WeightScheduler#_interpolation.
  MixedShardState: MixedShardState#
  MixedShardedSource._seed: MixedShardedSource#_seed.
  MixedShardedSource._stop_strategy: MixedShardedSource#_stop_strategy.
  block_mixture_interleave.gen: block_mixture_interleave().gen().
  MixedShardState.source_index: MixedShardState#source_index.
  MixedShardState.shard_index: MixedShardState#shard_index.
  MixedShardState.row_index: MixedShardState#row_index.
  MixedShardState.block_index: MixedShardState#block_index.
  MixedShardState.examples_yielded: MixedShardState#examples_yielded.
  MixedShardedSource.shard_names: MixedShardedSource#shard_names().
  MixedShardedSource.num_shards: MixedShardedSource#num_shards().
  MixStage.name: MixStage#name().
---
# Module: [`easydel/data/transforms/mixture.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py)

## Classes
### `MixStage`  ·  implements/extends BaseStage
- def: [`easydel/data/transforms/mixture.py:317`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L317)
- doc: Pipeline stage for mixing multiple datasets into one.
- signature: `class MixStage(BaseStage):`
- members:
  - `__init__(self, config: MixStageConfig | None = None)` — [`L326`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L326) — Initialize MixStage.
  - `name(self)` — [`L336`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L336)
  - `process(self, data: dict[str, ShardedDataSource], context: PipelineContext)` — [`L339`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L339) — Mix multiple datasets into one.
- protocol/private: `_stage_config`[`L333`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L333)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`BaseStage`](../core/protocols.md#BaseStage), [`PipelineContext`](../core/protocols.md#PipelineContext), [`MixStageConfig`](../core/config.md#MixStageConfig), [`MixedShardedSource`](mixture.md#MixedShardedSource), [`weight_schedule`](../core/config.md#MixStageConfig.weight_schedule), [`__init__`](../core/protocols.md#BaseStage.__init__), [`weights`](../core/config.md#MixStageConfig.weights), [`WeightScheduler`](mixture.md#WeightScheduler), [`seed`](../core/protocols.md#PipelineContext.seed), [`block_size`](../core/config.md#MixStageConfig.block_size), [`logger`](mixture.md#logger), [`seed`](../core/config.md#MixStageConfig.seed), [`stop_strategy`](../core/config.md#MixStageConfig.stop_strategy), [`weight_schedule_type`](../core/config.md#MixStageConfig.weight_schedule_type)
- used by: [`mix`](../execution/pipeline.md#Pipeline.mix), [`BaseStage`](../core/protocols.md#BaseStage), [`process`](../core/protocols.md#BaseStage.process), [`name`](../core/protocols.md#BaseStage.name)

### `MixedShardState`
- def: [`easydel/data/transforms/mixture.py:139`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L139)
- doc: State for tracking position in mixed iteration.
- signature: `class MixedShardState:`
- members:
  - `block_index` — [`L153`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L153)
  - `examples_yielded` — [`L154`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L154)
  - `row_index` — [`L152`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L152)
  - `shard_index` — [`L151`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L151)
  - `source_index` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L150)

### `MixedShardedSource`  ·  implements/extends ShardedDataSource
- def: [`easydel/data/transforms/mixture.py:157`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L157)
- doc: Sharded source that mixes multiple sources with weights.
- signature: `class MixedShardedSource(ShardedDataSource[dict]):`
- members:
  - `__init__(self, sources: dict[str, ShardedDataSource], weights: dict[str, float] | None = None, block_size: int = 1000, seed: int | None = None, stop_strategy: str = "restart", weight_scheduler: WeightScheduler | None = None)` — [`L163`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L163) — Initialize MixedShardedSource.
  - `__len__(self)` — [`L302`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L302) — Return total number of examples across all sources.
  - `_chain_shards(self, source: ShardedDataSource)` — [`L297`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L297) — Chain all shards of a source into a single iterator.
  - `_compute_counts(self, weights: dict[str, float])` — [`L219`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L219) — Compute per-dataset counts for a block.
  - `_get_weights_for_step(self, step: int)` — [`L213`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L213) — Get weights for a specific step (uses scheduler if available).
  - `num_shards(self)` — [`L210`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L210)
  - `open_shard(self, _shard_name: str)` — [`L231`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L231) — Open the mixed shard and iterate over interleaved examples.
  - `shard_names(self)` — [`L206`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L206)
- protocol/private: `__repr__`[`L310`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L310), `_block_size`[`L184`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L184), `_names`[`L183`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L183), `_seed`[`L185`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L185), `_sources`[`L182`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L182), `_stop_strategy`[`L186`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L186), `_weight_scheduler`[`L187`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L187), `_weights`[`L192`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L192)
- uses (calls/refs, reference-scoped): [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`get_weights`](mixture.md#WeightScheduler.get_weights), [`WeightScheduler`](mixture.md#WeightScheduler), [`logger`](mixture.md#logger)
- used by: [`ShardedDataSource`](../core/protocols.md#ShardedDataSource), [`open_shard`](../core/protocols.md#ShardedDataSource.open_shard), [`shard_names`](../core/protocols.md#ShardedDataSource.shard_names), [`num_shards`](../core/protocols.md#ShardedDataSource.num_shards), [`process`](mixture.md#MixStage.process), [`build_sharded_source`](../../infra/elarge/builders.md#build_sharded_source)

### `WeightScheduler`
- def: [`easydel/data/transforms/mixture.py:42`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L42)
- doc: Dynamic weight scheduler for dataset mixing.
- signature: `class WeightScheduler:`
- members:
  - `__init__(self, schedule: list[WeightSchedulePoint], interpolation: str = "step")` — [`L58`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L58) — Initialize WeightScheduler.
  - `dataset_names(self)` — [`L133`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L133) — Get the list of dataset names in the schedule.
  - `get_weights(self, step: int)` — [`L82`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L82) — Get weights for a specific training step.
- protocol/private: `_dataset_names`[`L75`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L75), `_interpolation`[`L74`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L74), `_schedule`[`L73`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L73)
- uses (calls/refs, reference-scoped): [`weights`](../core/config.md#WeightSchedulePoint.weights), [`step`](../core/config.md#WeightSchedulePoint.step), [`WeightSchedulePoint`](../core/config.md#WeightSchedulePoint)
- used by: [`process`](mixture.md#MixStage.process), [`_get_weights_for_step`](mixture.md#MixedShardedSource._get_weights_for_step), [`__init__`](mixture.md#MixedShardedSource.__init__)

## Functions
- `block_mixture_interleave(datasets: dict[str, tp.Any], weights: dict[str, float] | None = None, block_size: int = 1000, seed: int | None = 42, stop: str = "restart")` — [`L382`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L382) — Create a deterministic block-based mixture of multiple datasets.
- `gen()` — [`L457`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L457)

## Module values
- `logger` — [`L39`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/mixture.py#L39)

