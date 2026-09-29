---
title: 'Module: axlearn/common/summary_writer.py'
type: catalog
provenance: extracted
module: axlearn/common/summary_writer.py
status: fresh
symbol_base: scip-python python axlearn 0.0.0 `axlearn.common.summary_writer`/
symbols:
  BaseWriter: BaseWriter#
  _match_summary_type: _match_summary_type().
  SummaryWriter.write: SummaryWriter#write().
  CompositeWriter._writers: CompositeWriter#_writers.
  SummaryWriter.__init__: SummaryWriter#__init__().
  WandBWriter.log_config: WandBWriter#log_config().
  WandBWriter.convert: WandBWriter#convert().
  WandBWriter.format_config: WandBWriter#format_config().
  BaseWriter.Config: BaseWriter#Config#
  BaseWriter.log_config: BaseWriter#log_config().
  SummaryWriter.__call__: SummaryWriter#__call__().
  WandBWriter.__init__: WandBWriter#__init__().
  WandBWriter.__call__: WandBWriter#__call__().
  BaseWriter.Config.dir: BaseWriter#Config#dir.
  CompositeWriter.Config.writers: CompositeWriter#Config#writers.
  SummaryWriter.Config: SummaryWriter#Config#
  SummaryWriter.summary_writer: SummaryWriter#summary_writer.
  CompositeWriter.__init__: CompositeWriter#__init__().
  CompositeWriter.log_config: CompositeWriter#log_config().
  CompositeWriter.log_checkpoint: CompositeWriter#log_checkpoint().
  SummaryWriter._time_to_write: SummaryWriter#_time_to_write().
  WandBWriter._time_to_write: WandBWriter#_time_to_write().
  SummaryWriter: SummaryWriter#
  BaseWriter.log_checkpoint: BaseWriter#log_checkpoint().
  CompositeWriter.Config: CompositeWriter#Config#
  SummaryWriter.log_config: SummaryWriter#log_config().
  WandBWriter.Config: WandBWriter#Config#
  WandBWriter: WandBWriter#
  WandBWriter._initialize_run: WandBWriter#_initialize_run().
  processor_zero_only: processor_zero_only().
  CheckpointerAction: CheckpointerAction#
  CompositeWriter.writers: CompositeWriter#writers().
  CompositeWriter.__call__: CompositeWriter#__call__().
  SummaryKind: SummaryKind.
  WandBWriter.default_config: WandBWriter#default_config().
  SummaryWriter.is_leaf: SummaryWriter#is_leaf().
  WandBWriter.is_leaf: WandBWriter#is_leaf().
  SummaryWriter.as_default: SummaryWriter#as_default().
  WandBWriter._FLAT_CONFIG_KEY: WandBWriter#_FLAT_CONFIG_KEY.
  WandBWriter._upload: WandBWriter#_upload().
  CompositeWriter: CompositeWriter#
  NoOpWriter: NoOpWriter#
  NoOpWriter.log_config: NoOpWriter#log_config().
  SummaryWriter.Config.write_every_n_steps_map: SummaryWriter#Config#write_every_n_steps_map.
  WandBWriter.Config.write_every_n_steps_map: WandBWriter#Config#write_every_n_steps_map.
  CheckpointerAction.RESTORE: CheckpointerAction#RESTORE.
  CheckpointerAction.SAVE: CheckpointerAction#SAVE.
  processor_zero_only.wrapper: processor_zero_only().wrapper().
  SummaryWriter.Config.max_queue: SummaryWriter#Config#max_queue.
  SummaryWriter.Config.flush_ms: SummaryWriter#Config#flush_ms.
  BaseWriter.__call__: BaseWriter#__call__().
  NoOpWriter.__call__: NoOpWriter#__call__().
  SummaryWriter.Config.write_every_n_steps: SummaryWriter#Config#write_every_n_steps.
  WandBWriter.Config.write_every_n_steps: WandBWriter#Config#write_every_n_steps.
  WandBWriter.Config.prefix: WandBWriter#Config#prefix.
  WandBWriter.Config.exp_name: WandBWriter#Config#exp_name.
  WandBWriter.Config.project: WandBWriter#Config#project.
  WandBWriter.Config.entity: WandBWriter#Config#entity.
  WandBWriter.Config.group: WandBWriter#Config#group.
  WandBWriter.Config.tags: WandBWriter#Config#tags.
  WandBWriter.Config.notes: WandBWriter#Config#notes.
  WandBWriter.Config.mode: WandBWriter#Config#mode.
  WandBWriter.Config.resume: WandBWriter#Config#resume.
  WandBWriter.Config.convert_2d_to_image: WandBWriter#Config#convert_2d_to_image.
  WandBWriter.Config.wandb_dir: WandBWriter#Config#wandb_dir.
  WandBWriter.Config.wandb_settings_kwargs: WandBWriter#Config#wandb_settings_kwargs.
---
# Module: [`axlearn/common/summary_writer.py`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py)

## Classes
### `BaseWriter`  ·  implements/extends Module
- def: [`axlearn/common/summary_writer.py:57`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L57)
- doc: Base summary writer.
- signature: `class BaseWriter(Module):`
- members:
  - `__call__(self, step: int, values: dict[str, Any])` — [`L95`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L95) — Log data to disk.
  - `log_checkpoint(self, ckpt_dir: str, *, state: NestedTensor, action: CheckpointerAction, step: int = 0)` — [`L75`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L75) — Log a checkpoint. The default implementation is no-op.
  - `log_config(self, config: ConfigBase, step: int = 0)` — [`L66`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L66) — Log the config.
- uses (calls/refs, reference-scoped): [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`ConfigBase`](config.md#ConfigBase), [`log_config`](summary_writer.md#WandBWriter.log_config), [`log_checkpoint`](summary_writer.md#CompositeWriter.log_checkpoint), [`log_config`](summary_writer.md#CompositeWriter.log_config), [`SummaryWriter`](summary_writer.md#SummaryWriter), [`log_config`](summary_writer.md#SummaryWriter.log_config), [`WandBWriter`](summary_writer.md#WandBWriter), [`CheckpointerAction`](summary_writer.md#CheckpointerAction), [`CompositeWriter`](summary_writer.md#CompositeWriter), [`NoOpWriter`](summary_writer.md#NoOpWriter), [`log_config`](summary_writer.md#NoOpWriter.log_config)
- used by: [`Module`](module.md#Module), [`__init__`](summary_writer.md#SummaryWriter.__init__), [`_writers`](summary_writer.md#CompositeWriter._writers), [`summary_writer`](trainer.md#SpmdTrainer.Config.summary_writer), [`writers`](summary_writer.md#CompositeWriter.Config.writers), [`Config`](summary_writer.md#SummaryWriter.Config), [`log_checkpoint`](summary_writer.md#CompositeWriter.log_checkpoint), [`log_config`](summary_writer.md#CompositeWriter.log_config), [`SummaryWriter`](summary_writer.md#SummaryWriter), [`Config`](summary_writer.md#CompositeWriter.Config), [`Config`](summary_writer.md#WandBWriter.Config), [`WandBWriter`](summary_writer.md#WandBWriter), [`__call__`](summary_writer.md#CompositeWriter.__call__), [`writers`](summary_writer.md#CompositeWriter.writers), [`CompositeWriter`](summary_writer.md#CompositeWriter), [`NoOpWriter`](summary_writer.md#NoOpWriter)

### `CheckpointerAction`  ·  implements/extends Enum, str
- def: [`axlearn/common/summary_writer.py:30`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L30)
- doc: Represents the checkpointer action corresponding to a checkpoint summary.
- signature: `class CheckpointerAction(str, enum.Enum):`
- members:
  - `RESTORE` — [`L38`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L38)
  - `SAVE` — [`L39`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L39)
- used by: [`save`](checkpointer.md#Checkpointer.save), [`validate_and_restore`](checkpointer.md#Checkpointer.validate_and_restore), [`log_checkpoint`](summary_writer.md#CompositeWriter.log_checkpoint), [`log_checkpoint`](summary_writer.md#BaseWriter.log_checkpoint)

### `CompositeWriter`  ·  implements/extends BaseWriter
- def: [`axlearn/common/summary_writer.py:105`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L105)
- doc: A collection of named writers.
- signature: `class CompositeWriter(BaseWriter):`
- members:
  - `log_checkpoint(self, ckpt_dir: str, *, state: NestedTensor, action: CheckpointerAction, step: int = 0)` — [`L140`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L140)
  - `log_config(self, config: ConfigBase, step: int = 0)` — [`L130`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L130)
  - `writers(self)` — [`L126`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L126) — A list of writers.
- protocol/private: `__call__`[`L135`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L135), `__init__`[`L115`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L115), `_writers`[`L119`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L119)
- uses (calls/refs, reference-scoped): [`config`](config.md#Configurable.config), [`NestedTensor`](utils.md#NestedTensor), [`Module`](module.md#Module), [`_add_child`](module.md#Module._add_child), [`__init__`](module.md#Module.__init__), [`ConfigBase`](config.md#ConfigBase), [`BaseWriter`](summary_writer.md#BaseWriter), [`log_config`](summary_writer.md#BaseWriter.log_config), [`dir`](summary_writer.md#BaseWriter.Config.dir), [`writers`](summary_writer.md#CompositeWriter.Config.writers), [`Config`](summary_writer.md#CompositeWriter.Config), [`log_checkpoint`](summary_writer.md#BaseWriter.log_checkpoint), [`CheckpointerAction`](summary_writer.md#CheckpointerAction)
- used by: [`BaseWriter`](summary_writer.md#BaseWriter), [`log_config`](summary_writer.md#BaseWriter.log_config), [`log_checkpoint`](summary_writer.md#BaseWriter.log_checkpoint)

### `Config`  ·  implements/extends Config
- def: [`axlearn/common/summary_writer.py:362`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L362)
- doc: Configures WandBWriter.
- signature: `class Config(BaseWriter.Config):`
- members:
  - `convert_2d_to_image` — [`L404`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L404)
  - `dir` — [`L64`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L64)
  - `entity` — [`L386`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L386)
  - `exp_name` — [`L382`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L382)
  - `flush_ms` — [`L243`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L243)
  - `group` — [`L388`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L388)
  - `max_queue` — [`L242`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L242)
  - `mode` — [`L397`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L397)
  - `notes` — [`L392`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L392)
  - `prefix` — [`L372`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L372)
  - `project` — [`L384`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L384)
  - `resume` — [`L401`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L401)
  - `tags` — [`L390`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L390)
  - `wandb_dir` — [`L411`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L411)
  - `wandb_settings_kwargs` — [`L415`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L415)
  - `write_every_n_steps` — [`L240`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L240)
  - `write_every_n_steps` — [`L365`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L365)
  - `write_every_n_steps_map` — [`L241`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L241)
  - `write_every_n_steps_map` — [`L371`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L371)
  - `writers` — [`L113`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L113)
- uses (calls/refs, reference-scoped): [`REQUIRED`](config.md#REQUIRED), [`Required`](config.md#Required), [`BaseWriter`](summary_writer.md#BaseWriter), [`Config`](summary_writer.md#BaseWriter.Config), [`SummaryKind`](summary_writer.md#SummaryKind)
- used by: [`_input_iter`](trainer.md#SpmdTrainer._input_iter), [`_save_policy`](checkpointer.md#Checkpointer._save_policy), [`_writers`](summary_writer.md#CompositeWriter._writers), [`summary_writer`](summary_writer.md#SummaryWriter.summary_writer), [`default_config`](summary_writer.md#WandBWriter.default_config)

### `NoOpWriter`  ·  implements/extends BaseWriter
- def: [`axlearn/common/summary_writer.py:152`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L152)
- doc: A writer that does nothing. Used by testing.
- signature: `class NoOpWriter(BaseWriter):`
- members:
  - `log_config(self, config: ConfigBase, step: int = 0)` — [`L155`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L155)
- protocol/private: `__call__`[`L158`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L158)
- uses (calls/refs, reference-scoped): [`ConfigBase`](config.md#ConfigBase), [`BaseWriter`](summary_writer.md#BaseWriter)
- used by: [`BaseWriter`](summary_writer.md#BaseWriter), [`log_config`](summary_writer.md#BaseWriter.log_config)

### `SummaryWriter`  ·  implements/extends BaseWriter
- def: [`axlearn/common/summary_writer.py:217`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L217)
- doc: Tensorflow summary writer.
- signature: `class SummaryWriter(BaseWriter):`
- members:
  - `as_default(self)` — [`L257`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L257)
  - `is_leaf(x)` — [`L341`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L341)
  - `log_config(self, config: ConfigBase, step: int = 0)` — [`L261`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L261)
  - `write(path: str, value: jax.Array)` — [`L285`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L285)
  - `summary_writer` — [`L248`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L248)
- protocol/private: `__call__`[`L278`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L278), `__init__`[`L245`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L245), `_time_to_write`[`L270`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L270)
- uses (calls/refs, reference-scoped): [`Tensor`](utils.md#Tensor), [`config`](config.md#Configurable.config), [`Module`](module.md#Module), [`vlog`](module.md#Module.vlog), [`Summary`](summary.md#Summary), [`__init__`](module.md#Module.__init__), [`path`](module.md#Module.path), [`ConfigBase`](config.md#ConfigBase), [`tree_paths`](utils.md#tree_paths), [`BaseWriter`](summary_writer.md#BaseWriter), [`_match_summary_type`](summary_writer.md#_match_summary_type), [`value`](summary.md#Summary.value), [`Config`](summary_writer.md#BaseWriter.Config), [`dir`](summary_writer.md#BaseWriter.Config.dir), [`Config`](summary_writer.md#SummaryWriter.Config), [`debug_string`](config.md#ConfigBase.debug_string), [`SummaryKind`](summary_writer.md#SummaryKind), [`flush_ms`](summary_writer.md#SummaryWriter.Config.flush_ms), [`max_queue`](summary_writer.md#SummaryWriter.Config.max_queue)
- used by: [`BaseWriter`](summary_writer.md#BaseWriter), [`log_config`](summary_writer.md#BaseWriter.log_config), [`summary_writer`](trainer.md#SpmdTrainer.Config.summary_writer), [`__init__`](summary_writer.md#WandBWriter.__init__), [`summary_writer`](evaler.md#SpmdEvaler.Config.summary_writer), [`summary_writer`](checkpointer.md#Checkpointer.Config.summary_writer)

### `WandBWriter`  ·  implements/extends BaseWriter
- def: [`axlearn/common/summary_writer.py:349`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L349)
- doc: Utility for logging with Weights and Biases.
- signature: `class WandBWriter(BaseWriter):`
- members:
  - `__call__(self, step: int, values: dict[str, Any])` — [`L519`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L519) — Convert nested summary values to wandb acceptable format and upload run data.
  - `convert(path: str, value: Any)` — [`L525`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L525)
  - `default_config(cls: Config)` — [`L418`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L418)
  - `format_config(val)` — [`L480`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L480) — Helper function to format config for wandb logging.
  - `is_leaf(x)` — [`L573`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L573)
  - `log_config(self, config: ConfigBase, step: int = 0)` — [`L498`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L498)
- protocol/private: `_FLAT_CONFIG_KEY`[`L359`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L359), `__init__`[`L431`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L431), `_initialize_run`[`L442`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L442), `_time_to_write`[`L511`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L511), `_upload`[`L597`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L597)
- uses (calls/refs, reference-scoped): [`Nested`](utils.md#Nested), [`default_config`](config.md#Configurable.default_config), [`config`](config.md#Configurable.config), [`config_class`](config.md#config_class), [`Module`](module.md#Module), [`vlog`](module.md#Module.vlog), [`Summary`](summary.md#Summary), [`__init__`](module.md#Module.__init__), [`path`](module.md#Module.path), [`ConfigBase`](config.md#ConfigBase), [`tree_paths`](utils.md#tree_paths), [`BaseWriter`](summary_writer.md#BaseWriter), [`_match_summary_type`](summary_writer.md#_match_summary_type), [`value`](summary.md#Summary.value), [`RequiredFieldValue`](config.md#RequiredFieldValue), [`Config`](summary_writer.md#SummaryWriter.Config), [`to_flat_dict`](config.md#ConfigBase.to_flat_dict), [`SummaryWriter`](summary_writer.md#SummaryWriter), [`Config`](summary_writer.md#WandBWriter.Config), [`to_dict`](config.md#ConfigBase.to_dict), [`processor_zero_only`](summary_writer.md#processor_zero_only), [`SummaryKind`](summary_writer.md#SummaryKind)
- used by: [`default_config`](config.md#Configurable.default_config), [`BaseWriter`](summary_writer.md#BaseWriter), [`log_config`](summary_writer.md#BaseWriter.log_config)

## Functions
- `_match_summary_type(kind: SummaryKind, *, value: Union[Summary, Tensor], raw_value: Union[np.ndarray, numbers.Number, str])` — [`L175`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L175) — Checks whether a given value is appropriate for the specified summary kind.
- `processor_zero_only(fn: Callable)` — [`L42`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L42) — Decorator to use for operations that should only happen on the main process.
- `wrapper(*args: Any, **kwargs: Any)` — [`L49`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L49)

## Module values
- `SummaryKind` — [`L172`](../../../../../../raw/code/axlearn/axlearn/common/summary_writer.py#L172)

