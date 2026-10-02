---
title: 'Module: axlearn/common/input_base.py'
type: catalog
provenance: extracted
module: axlearn/common/input_base.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.input_base`/
symbols:
  Input.dispatch_global_batch: Input#dispatch_global_batch().
  Input: Input#
  Input.batches: Input#batches().
  Input.input_dispatcher: Input#input_dispatcher.
  Input.dataset: Input#dataset().
  Input.Config: Input#Config#
  partition_by_path_rank.fn: partition_by_path_rank().fn().
  Input.__init__: Input#__init__().
  Input.partition_spec: Input#partition_spec().
  Input.constrain_batch_axis: Input#constrain_batch_axis().
  Input._partition_spec: Input#_partition_spec.
  Input.Config.input_dispatcher: Input#Config#input_dispatcher.
  Input.__iter__: Input#__iter__().
  Input._input_partitioner: Input#_input_partitioner.
  partition_by_path_rank: partition_by_path_rank().
  partition_by_path_rank.fn.maybe_constrain: partition_by_path_rank().fn().maybe_constrain().
  Input.Config.input_partitioner: Input#Config#input_partitioner.
  Input.element_spec: Input#element_spec().
  InputPartitionFn.__call__: InputPartitionFn#__call__().
  InputPartitionFn: InputPartitionFn#
  Input.Config.partition_spec: Input#Config#partition_spec.
  PathAndRank: PathAndRank#
  PathAndRank.path: PathAndRank#path.
  PathAndRank.rank: PathAndRank#rank.
---
# Module: [`axlearn/common/input_base.py`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py)

## Classes
### `Config`  ·  implements/extends Config
- def: [`axlearn/common/input_base.py:152`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L152)
- doc: Configures Input.
- signature: `class Config(Module.Config):`
- members:
  - `input_dispatcher` — [`L173`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L173)
  - `input_partitioner` — [`L174`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L174)
  - `partition_spec` — [`L172`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L172)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`Module`](module.md#Module), [`ConfigOr`](config.md#ConfigOr), [`Config`](module.md#Module.Config), [`Config`](input_dispatch.md#InputDispatcher.Config), [`InputDispatcher`](input_dispatch.md#InputDispatcher), [`InputPartitionFn`](input_base.md#InputPartitionFn)
- used by: [`__init__`](elastic_input.md#ElasticInput.__init__), [`input_dispatcher`](input_base.md#Input.input_dispatcher), [`elastic_input`](elastic_input.md#ElasticInput.elastic_input), [`input`](elastic_input.md#ElasticInput.Config.input), [`__init__`](input_base.md#Input.__init__), [`input`](trainer.md#SpmdTrainer.Config.input), [`input`](evaler.md#SpmdEvaler.Config.input), [`_partition_spec`](input_base.md#Input._partition_spec), [`Config`](input_fake.md#EmptyInput.Config), [`Config`](input_tf_data.md#Input.Config), [`_input_partitioner`](input_base.md#Input._input_partitioner), [`Config`](elastic_input.md#ElasticInput.Config)

### `Input`  ·  implements/extends Module
- def: [`axlearn/common/input_base.py:122`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L122)
- doc: A Module to generate input batches.
- signature: `class Input(Module):`
- members:
  - `__iter__(self)` — [`L203`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L203) — Iterates over the input dataset.
  - `batches(self, it: Iterator[Nested[Tensor]])` — [`L213`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L213) — Yields per-feed physical input batches (using `input_dispatcher` if configured).
  - `constrain_batch_axis(path: str, value: Tensor)` — [`L238`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L238)
  - `dataset(self)` — [`L191`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L191) — Returns the input dataset, which should produce per-feed logical batches.
  - `dispatch_global_batch(self, global_physical_batch: Nested[Tensor])` — [`L227`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L227) — Converts a global physical batch to a global logical batch.
  - `element_spec(self)` — [`L278`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L278) — Returns the per-feed logical batch spec.
  - `partition_spec(self)` — [`L286`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L286) — Returns the input partition spec for `host_to_global_device_array` and for `jit`.
  - `input_dispatcher` — [`L181`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L181)
- protocol/private: `__init__`[`L176`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L176), `_input_partitioner`[`L187`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L187), `_partition_spec`[`L179`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L179)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`maybe_instantiate`](config.md#maybe_instantiate), [`__init__`](module.md#Module.__init__), [`with_sharding_constraint`](utils.md#with_sharding_constraint), [`tree_paths`](utils.md#tree_paths), [`maybe_set_config`](config.md#maybe_set_config), [`children`](module.md#Module.children), [`batches`](elastic_input.md#ElasticInput.batches), [`dataset`](elastic_input.md#ElasticInput.dataset), `env`, [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`EmptyInput`](input_fake.md#EmptyInput), [`Config`](input_base.md#Input.Config), [`logical_to_physical_batch`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_batch), [`physical_to_logical_batch`](input_dispatch.md#BaseInputDispatcher.physical_to_logical_batch), [`input_partition_spec`](utils.md#input_partition_spec), [`as_numpy_array`](utils.md#as_numpy_array), [`input_dispatcher`](input_base.md#Input.Config.input_dispatcher), [`dispatch_input_batch`](utils.md#dispatch_input_batch), [`dataset`](input_tf_data.md#Input.dataset), [`partition_spec`](input_dispatch.md#BaseInputDispatcher.partition_spec), [`batches`](input_fake.md#EmptyInput.batches), [`input_partitioner`](input_base.md#Input.Config.input_partitioner), [`element_spec`](input_tf_data.md#Input.element_spec), [`ElasticInput`](elastic_input.md#ElasticInput), [`Input`](input_tf_data.md#Input), [`InputPartitionFn`](input_base.md#InputPartitionFn), [`dataset`](input_fake.md#EmptyInput.dataset), [`partition_spec`](input_base.md#Input.Config.partition_spec)
- used by: [`Module`](module.md#Module), [`run`](trainer.md#SpmdTrainer.run), [`_train_step`](trainer.md#SpmdTrainer._train_step), [`eval_step`](evaler.md#SpmdEvaler.eval_step), [`compile_train_step`](trainer.md#SpmdTrainer.compile_train_step), [`restore_checkpoint`](trainer.md#SpmdTrainer.restore_checkpoint), [`input`](evaler.md#SpmdEvaler.input), [`__init__`](elastic_input.md#ElasticInput.__init__), [`__init__`](input_tf_data.md#Input.__init__), [`input`](trainer.md#SpmdTrainer.input), [`EmptyInput`](input_fake.md#EmptyInput), [`input`](elastic_input.md#ElasticInput.Config.input), [`_input_partition_spec`](evaler.md#BaseMetricCalculator._input_partition_spec), [`_dispatch_global_batch`](evaler.md#BaseMetricCalculator._dispatch_global_batch), [`input`](trainer.md#SpmdTrainer.Config.input), [`__init__`](input_fake.md#EmptyInput.__init__), [`input`](evaler.md#SpmdEvaler.Config.input), [`Config`](input_fake.md#EmptyInput.Config), [`Config`](input_tf_data.md#Input.Config), [`Config`](elastic_input.md#ElasticInput.Config), [`_train_step_input_partition_specs`](trainer.md#SpmdTrainer._train_step_input_partition_specs), [`ElasticInput`](elastic_input.md#ElasticInput), [`Input`](input_tf_data.md#Input)

### `InputPartitionFn`  ·  implements/extends Protocol
- def: [`axlearn/common/input_base.py:28`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L28)
- doc: Partitions the input batch.
- signature: `class InputPartitionFn(Protocol):`
- members:
  - `__call__(self, input_batch: Nested[Tensor])` — [`L31`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L31) — Applies sharding constraints to `input_batch` and returns the modified batch.
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested)
- used by: [`_input_partitioner`](input_base.md#Input._input_partitioner), [`partition_by_path_rank`](input_base.md#partition_by_path_rank), [`input_partitioner`](input_base.md#Input.Config.input_partitioner)

### `PathAndRank`  ·  implements/extends NamedTuple
- def: [`axlearn/common/input_base.py:38`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L38)
- doc: A tuple (path, rank) used for matching against inputs in a batch.
- signature: `class PathAndRank(NamedTuple):`
- members:
  - `path` — [`L46`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L46)
  - `rank` — [`L47`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L47)
- used by: [`partition_by_path_rank`](input_base.md#partition_by_path_rank)

## Functions
- `fn(input_batch: Nested[Tensor])` — [`L89`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L89)
- `maybe_constrain(path: str, value: Tensor)` — [`L94`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L94)
- `partition_by_path_rank(path_rank_to_partition: dict[PathAndRank, PartitionSpec])` — [`L50`](../../../../../../raw/code/axlearn/axlearn/common/input_base.py#L50) — Partitions the paths in the input batch by regex and rank (ndim).

