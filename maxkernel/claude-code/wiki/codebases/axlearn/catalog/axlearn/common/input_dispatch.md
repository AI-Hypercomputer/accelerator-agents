---
title: 'Module: axlearn/common/input_dispatch.py'
type: catalog
provenance: extracted
module: axlearn/common/input_dispatch.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.input_dispatch`/
symbols:
  InputDispatcher.logical_to_physical_batch: InputDispatcher#logical_to_physical_batch().
  InputDispatcher.__init__: InputDispatcher#__init__().
  BaseInputDispatcher.Config.global_logical_batch_size: BaseInputDispatcher#Config#global_logical_batch_size.
  BaseInputDispatcher: BaseInputDispatcher#
  SpmdInputDispatcher.__init__: SpmdInputDispatcher#__init__().
  BaseInputDispatcher.logical_to_physical_batch: BaseInputDispatcher#logical_to_physical_batch().
  BaseInputDispatcher.physical_to_logical_batch: BaseInputDispatcher#physical_to_logical_batch().
  SpmdInputDispatcher.feed_logical_batch_size: SpmdInputDispatcher#feed_logical_batch_size().
  InputDispatcher.traverse_and_dispatch: InputDispatcher#traverse_and_dispatch().
  SpmdInputDispatcher.Config.partition_spec: SpmdInputDispatcher#Config#partition_spec.
  BaseInputDispatcher.logical_to_physical_shapes: BaseInputDispatcher#logical_to_physical_shapes().
  InputDispatcher.Config.num_physical_feeds: InputDispatcher#Config#num_physical_feeds.
  BaseInputDispatcher.Config: BaseInputDispatcher#Config#
  InputDispatcher.Config: InputDispatcher#Config#
  SpmdInputDispatcher.Config: SpmdInputDispatcher#Config#
  InputDispatcher.Config.logical_feed_indices: InputDispatcher#Config#logical_feed_indices.
  InputDispatcher.physical_to_logical_batch: InputDispatcher#physical_to_logical_batch().
  BaseInputDispatcher.feed_logical_batch_size: BaseInputDispatcher#feed_logical_batch_size().
  BaseInputDispatcher.partition_spec: BaseInputDispatcher#partition_spec().
  BaseInputDispatcher.feed_read_config: BaseInputDispatcher#feed_read_config().
  InputDispatcher.num_logical_feeds: InputDispatcher#num_logical_feeds().
  InputDispatcher.pad_to_physical_batch_size: InputDispatcher#pad_to_physical_batch_size().
  InputDispatcher.feed_logical_batch_size: InputDispatcher#feed_logical_batch_size().
  BaseInputDispatcher.num_logical_feeds: BaseInputDispatcher#num_logical_feeds().
  BaseInputDispatcher.logical_feed_index: BaseInputDispatcher#logical_feed_index().
  InputDispatcher.feed_read_config: InputDispatcher#feed_read_config().
  InputDispatcher.logical_to_physical_shapes: InputDispatcher#logical_to_physical_shapes().
  InputDispatcher.Config.physical_feed_index: InputDispatcher#Config#physical_feed_index.
  _validate_logical_feed_shapes: _validate_logical_feed_shapes().
  InputDispatcher.feed_physical_batch_size: InputDispatcher#feed_physical_batch_size().
  SpmdInputDispatcher._partition_spec: SpmdInputDispatcher#_partition_spec.
  InputDispatcher: InputDispatcher#
  InputDispatcher.Config.global_physical_batch_size: InputDispatcher#Config#global_physical_batch_size.
  SpmdInputDispatcher.feed_read_config: SpmdInputDispatcher#feed_read_config().
  SpmdInputDispatcher.logical_to_physical_batch: SpmdInputDispatcher#logical_to_physical_batch().
  SpmdInputDispatcher.physical_to_logical_batch: SpmdInputDispatcher#physical_to_logical_batch().
  SpmdInputDispatcher.logical_to_physical_shapes: SpmdInputDispatcher#logical_to_physical_shapes().
  InputDispatcher.logical_feed_index: InputDispatcher#logical_feed_index().
  SpmdInputDispatcher: SpmdInputDispatcher#
  SpmdInputDispatcher._num_logical_feeds: SpmdInputDispatcher#_num_logical_feeds.
  InputDispatcher.partition_spec: InputDispatcher#partition_spec().
  SpmdInputDispatcher.num_logical_feeds: SpmdInputDispatcher#num_logical_feeds().
  SpmdInputDispatcher.logical_feed_index: SpmdInputDispatcher#logical_feed_index().
  SpmdInputDispatcher.partition_spec: SpmdInputDispatcher#partition_spec().
  SpmdInputDispatcher._logical_feed_index: SpmdInputDispatcher#_logical_feed_index.
---
# Module: [`axlearn/common/input_dispatch.py`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py)

## Classes
### `BaseInputDispatcher`  ·  implements/extends Module
- def: [`axlearn/common/input_dispatch.py:28`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L28)
- doc: Base input dispatch interface.
- signature: `class BaseInputDispatcher(Module):`
- members:
  - `feed_logical_batch_size(self)` — [`L46`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L46)
  - `feed_read_config(self)` — [`L53`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L53)
  - `logical_feed_index(self)` — [`L42`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L42)
  - `logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor])` — [`L56`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L56)
  - `logical_to_physical_shapes(self, logical_feed_shapes: Nested[jax.ShapeDtypeStruct])` — [`L62`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L62) — Maps per-feed logical shapes to per-feed physical shapes for AOT compilation.
  - `num_logical_feeds(self)` — [`L38`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L38)
  - `partition_spec(self)` — [`L50`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L50)
  - `physical_to_logical_batch(self, global_physical_batch: Nested[Tensor])` — [`L59`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L59)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`Module`](module.md#Module), [`logical_to_physical_batch`](input_dispatch.md#InputDispatcher.logical_to_physical_batch), [`feed_read_config`](elastic_input.md#ElasticSpmdInputDispatcher.feed_read_config), [`feed_logical_batch_size`](input_dispatch.md#SpmdInputDispatcher.feed_logical_batch_size), [`partition_spec`](elastic_input.md#ElasticSpmdInputDispatcher.partition_spec), [`physical_to_logical_batch`](input_dispatch.md#InputDispatcher.physical_to_logical_batch), [`num_logical_feeds`](input_dispatch.md#InputDispatcher.num_logical_feeds), [`feed_logical_batch_size`](input_dispatch.md#InputDispatcher.feed_logical_batch_size), [`ElasticSpmdInputDispatcher`](elastic_input.md#ElasticSpmdInputDispatcher), [`feed_read_config`](input_dispatch.md#InputDispatcher.feed_read_config), [`logical_to_physical_shapes`](input_dispatch.md#InputDispatcher.logical_to_physical_shapes), [`InputDispatcher`](input_dispatch.md#InputDispatcher), [`feed_read_config`](input_dispatch.md#SpmdInputDispatcher.feed_read_config), [`logical_to_physical_batch`](elastic_input.md#ElasticSpmdInputDispatcher.logical_to_physical_batch), [`logical_to_physical_batch`](input_dispatch.md#SpmdInputDispatcher.logical_to_physical_batch), [`logical_to_physical_shapes`](elastic_input.md#ElasticSpmdInputDispatcher.logical_to_physical_shapes), [`logical_to_physical_shapes`](input_dispatch.md#SpmdInputDispatcher.logical_to_physical_shapes), [`physical_to_logical_batch`](elastic_input.md#ElasticSpmdInputDispatcher.physical_to_logical_batch), [`physical_to_logical_batch`](input_dispatch.md#SpmdInputDispatcher.physical_to_logical_batch), [`SpmdInputDispatcher`](input_dispatch.md#SpmdInputDispatcher), [`logical_feed_index`](input_dispatch.md#InputDispatcher.logical_feed_index), [`num_logical_feeds`](elastic_input.md#ElasticSpmdInputDispatcher.num_logical_feeds), [`feed_logical_batch_size`](elastic_input.md#ElasticSpmdInputDispatcher.feed_logical_batch_size), [`logical_feed_index`](elastic_input.md#ElasticSpmdInputDispatcher.logical_feed_index), [`logical_feed_index`](input_dispatch.md#SpmdInputDispatcher.logical_feed_index), [`num_logical_feeds`](input_dispatch.md#SpmdInputDispatcher.num_logical_feeds), [`partition_spec`](input_dispatch.md#InputDispatcher.partition_spec), [`partition_spec`](input_dispatch.md#SpmdInputDispatcher.partition_spec)
- used by: [`Module`](module.md#Module), [`compile_train_step`](trainer.md#SpmdTrainer.compile_train_step), [`dispatch_global_batch`](input_base.md#Input.dispatch_global_batch), [`batches`](input_base.md#Input.batches), [`__init__`](input_tf_data.md#Input.__init__), [`input_dispatcher`](input_base.md#Input.input_dispatcher), [`partition_spec`](input_base.md#Input.partition_spec), [`Config`](elastic_input.md#ElasticSpmdInputDispatcher.Config), [`Config`](input_dispatch.md#InputDispatcher.Config), [`Config`](input_dispatch.md#SpmdInputDispatcher.Config), [`ElasticSpmdInputDispatcher`](elastic_input.md#ElasticSpmdInputDispatcher), [`InputDispatcher`](input_dispatch.md#InputDispatcher), [`SpmdInputDispatcher`](input_dispatch.md#SpmdInputDispatcher)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/input_dispatch.py:340`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L340)
- doc: Configuration for SpmdInputDispatcher.
- signature: `class Config(BaseInputDispatcher.Config):`
- members:
  - `global_logical_batch_size` — [`L35`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L35)
  - `global_physical_batch_size` — [`L103`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L103)
  - `logical_feed_indices` — [`L115`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L115)
  - `num_physical_feeds` — [`L106`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L106)
  - `partition_spec` — [`L347`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L347)
  - `physical_feed_index` — [`L110`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L110)
- uses (calls/refs, reference-scoped): [`config_class`](config.md#config_class), [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`Config`](input_dispatch.md#BaseInputDispatcher.Config)
- used by: [`logical_to_physical_batch`](input_dispatch.md#InputDispatcher.logical_to_physical_batch), [`__init__`](input_dispatch.md#InputDispatcher.__init__), [`__init__`](input_dispatch.md#SpmdInputDispatcher.__init__), [`_feed_logical_batch_size`](elastic_input.md#ElasticSpmdInputDispatcher._feed_logical_batch_size), [`_partition_spec`](elastic_input.md#ElasticSpmdInputDispatcher._partition_spec), [`feed_logical_batch_size`](input_dispatch.md#SpmdInputDispatcher.feed_logical_batch_size), [`feed_index`](elastic_input.md#ElasticSpmdInputDispatcher.feed_index), [`_device_physical_batch_size`](elastic_input.md#ElasticSpmdInputDispatcher._device_physical_batch_size), [`pad_to_physical_batch_size`](input_dispatch.md#InputDispatcher.pad_to_physical_batch_size), [`_partition_spec`](input_dispatch.md#SpmdInputDispatcher._partition_spec)

### `InputDispatcher`  ·  implements/extends BaseInputDispatcher
- def: [`axlearn/common/input_dispatch.py:79`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L79)
- doc: A Module to dispatch per-feed logical input batches to global logical batches on device.
- signature: `class InputDispatcher(BaseInputDispatcher):`
- members:
  - `feed_logical_batch_size(self)` — [`L170`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L170)
  - `feed_physical_batch_size(self)` — [`L174`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L174)
  - `feed_read_config(self)` — [`L182`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L182) — Generates the read configuration for the local physical feed.
  - `logical_feed_index(self)` — [`L163`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L163)
  - `logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor])` — [`L205`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L205) — Converts a per-feed logical batch to a per-feed physical batch.
  - `logical_to_physical_shapes(self, logical_feed_shapes: Nested[jax.ShapeDtypeStruct])` — [`L316`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L316)
  - `num_logical_feeds(self)` — [`L159`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L159)
  - `pad_to_physical_batch_size(x: Tensor)` — [`L231`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L231)
  - `partition_spec(self)` — [`L179`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L179)
  - `physical_to_logical_batch(self, global_physical_batch: Nested[Tensor])` — [`L284`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L284) — Converts a global physical batch to a global logical batch.
  - `traverse_and_dispatch(data: Nested[Tensor])` — [`L297`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L297)
- protocol/private: `__init__`[`L117`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L117)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`__init__`](module.md#Module.__init__), [`global_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.Config.global_logical_batch_size), [`clone`](config.md#ConfigBase.clone), [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`input_partition_spec`](utils.md#input_partition_spec), [`PHYSICAL_TO_LOGICAL_DISPATCH_KEY`](utils.md#PHYSICAL_TO_LOGICAL_DISPATCH_KEY), [`num_physical_feeds`](input_dispatch.md#InputDispatcher.Config.num_physical_feeds), [`Config`](input_dispatch.md#InputDispatcher.Config), [`logical_feed_indices`](input_dispatch.md#InputDispatcher.Config.logical_feed_indices), [`physical_feed_index`](input_dispatch.md#InputDispatcher.Config.physical_feed_index), [`_validate_logical_feed_shapes`](input_dispatch.md#_validate_logical_feed_shapes), [`global_physical_batch_size`](input_dispatch.md#InputDispatcher.Config.global_physical_batch_size)
- used by: [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`logical_to_physical_batch`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_batch), [`physical_to_logical_batch`](input_dispatch.md#BaseInputDispatcher.physical_to_logical_batch), [`logical_to_physical_shapes`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_shapes), [`input_dispatcher`](input_base.md#Input.Config.input_dispatcher), [`feed_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.feed_logical_batch_size), [`feed_read_config`](input_dispatch.md#BaseInputDispatcher.feed_read_config), [`partition_spec`](input_dispatch.md#BaseInputDispatcher.partition_spec), [`logical_feed_index`](input_dispatch.md#BaseInputDispatcher.logical_feed_index), [`num_logical_feeds`](input_dispatch.md#BaseInputDispatcher.num_logical_feeds)

### `SpmdInputDispatcher`  ·  implements/extends BaseInputDispatcher
- def: [`axlearn/common/input_dispatch.py:328`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L328)
- doc: A variant of InputDispatcher which is mesh/topology aware.
- signature: `class SpmdInputDispatcher(BaseInputDispatcher):`
- members:
  - `feed_logical_batch_size(self)` — [`L399`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L399)
  - `feed_read_config(self)` — [`L407`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L407)
  - `logical_feed_index(self)` — [`L395`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L395)
  - `logical_to_physical_batch(self, logical_feed_batch: Nested[Tensor])` — [`L410`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L410)
  - `logical_to_physical_shapes(self, logical_feed_shapes: Nested[jax.ShapeDtypeStruct])` — [`L416`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L416)
  - `num_logical_feeds(self)` — [`L391`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L391)
  - `partition_spec(self)` — [`L404`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L404)
  - `physical_to_logical_batch(self, global_physical_batch: Nested[Tensor])` — [`L413`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L413)
- protocol/private: `__init__`[`L349`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L349), `_logical_feed_index`[`L388`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L388), `_num_logical_feeds`[`L387`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L387), `_partition_spec`[`L370`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L370)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`Nested`](utils.md#Nested), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`__init__`](module.md#Module.__init__), [`global_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.Config.global_logical_batch_size), `env`, [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`partition_spec`](input_dispatch.md#SpmdInputDispatcher.Config.partition_spec), [`Config`](input_dispatch.md#SpmdInputDispatcher.Config), [`_validate_logical_feed_shapes`](input_dispatch.md#_validate_logical_feed_shapes)
- used by: [`BaseInputDispatcher`](input_dispatch.md#BaseInputDispatcher), [`logical_to_physical_batch`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_batch), [`physical_to_logical_batch`](input_dispatch.md#BaseInputDispatcher.physical_to_logical_batch), [`logical_to_physical_shapes`](input_dispatch.md#BaseInputDispatcher.logical_to_physical_shapes), [`feed_logical_batch_size`](input_dispatch.md#BaseInputDispatcher.feed_logical_batch_size), [`feed_read_config`](input_dispatch.md#BaseInputDispatcher.feed_read_config), [`partition_spec`](input_dispatch.md#BaseInputDispatcher.partition_spec), [`logical_feed_index`](input_dispatch.md#BaseInputDispatcher.logical_feed_index), [`num_logical_feeds`](input_dispatch.md#BaseInputDispatcher.num_logical_feeds)

## Functions
- `_validate_logical_feed_shapes(logical_feed_shapes: Nested[jax.ShapeDtypeStruct])` — [`L69`](../../../../../../raw/code/axlearn/axlearn/common/input_dispatch.py#L69) — Validates that leaves have `dtype`, `shape` with ndim >= 1, and consistent batch dims.

