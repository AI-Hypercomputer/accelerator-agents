---
title: 'Module: simply/utils/checkpoint_lib.py'
type: catalog
provenance: extracted
module: simply/utils/checkpoint_lib.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.checkpoint_lib`/
symbols:
  load_checkpoint_from_path: load_checkpoint_from_path().
  PyTree: PyTree.
  load_checkpoint_from_dir: load_checkpoint_from_dir().
  save_checkpoint: save_checkpoint().
  get_abstract_params: get_abstract_params().
  CheckpointFormatRegistry: CheckpointFormatRegistry#
  construct_restore_item: construct_restore_item().
  Gemma3pFormat.transforms: Gemma3pFormat#transforms().
  Qwen2Format.transforms: Qwen2Format#transforms().
  Gemma3pFormat.prefix_mapping: Gemma3pFormat#prefix_mapping().
  CheckpointFormat.transforms: CheckpointFormat#transforms().
  Qwen2Format._split_head: Qwen2Format#_split_head().
  Qwen2Format: Qwen2Format#
  resolve_checkpoint_handler_from_json: resolve_checkpoint_handler_from_json().
  load_checkpoint_from_path.transform_state_fn: load_checkpoint_from_path().transform_state_fn().
  CheckpointFormat: CheckpointFormat#
  V2Format: V2Format#
  DefaultFormat: DefaultFormat.
  Gemma3pLegacyFormat: Gemma3pLegacyFormat#
  Gemma2Format: Gemma2Format#
  CheckpointFormat.__post_init__: CheckpointFormat#__post_init__().
  LegacyFormat: LegacyFormat#
  load_checkpoint_from_manager: load_checkpoint_from_manager().
  load_data_state_from_dir: load_data_state_from_dir().
  get_checkpoint_path: get_checkpoint_path().
  CheckpointFormatRegistry.namespace: CheckpointFormatRegistry#namespace.
  Gemma3pLegacyFormat.transpose_ffn_weights: Gemma3pLegacyFormat#transpose_ffn_weights.
  Gemma3pFormat: Gemma3pFormat#
  Gemma2TransposeFormat: Gemma2TransposeFormat#
  resolve_checkpoint_handler_from_path: resolve_checkpoint_handler_from_path().
  load_checkpoint_from_path.transform_state_fn._get_regularized_value: load_checkpoint_from_path().transform_state_fn()._get_regularized_value().
  DATA_ITEM_NAME: DATA_ITEM_NAME.
  Qwen2Format._gather_experts: Qwen2Format#_gather_experts().
  LegacyFormat.transforms: LegacyFormat#transforms().
  CHECKPOINT_FORMAT_KEY: CHECKPOINT_FORMAT_KEY.
  Gemma3pFormat.ln_mapping: Gemma3pFormat#ln_mapping().
  CheckpointFormat.restore_dtype: CheckpointFormat#restore_dtype.
  Gemma3pFormat.transpose_ffn_weights: Gemma3pFormat#transpose_ffn_weights.
  last_checkpoint_step: last_checkpoint_step().
  construct_restore_item._restore_leaf_dtype: construct_restore_item()._restore_leaf_dtype().
  Gemma2TransposeFormat.prefix_mapping: Gemma2TransposeFormat#prefix_mapping().
  Gemma2Format.prefix_mapping: Gemma2Format#prefix_mapping().
  readonly_checkpoint_manager: readonly_checkpoint_manager().
---
# Module: [`simply/utils/checkpoint_lib.py`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py)

## Classes
### `CheckpointFormat`  ·  implements/extends ABC
- def: [`simply/utils/checkpoint_lib.py:46`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L46) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- doc: Checkpoint formats for Simply.
- signature: `class CheckpointFormat(abc.ABC):`
- members:
  - `transforms(self, stored_state: PyTree, target_abstract_state: PyTree = None)` — [`L65`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L65) — Transforms the stored state. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
  - `restore_dtype` — [`L49`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L49)
- protocol/private: `__post_init__`[`L52`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L52)
- uses (calls/refs, reference-scoped): [`get`](registry.md#RootRegistry.get), [`PyTree`](checkpoint_lib.md#PyTree), [`CheckpointFormatRegistry`](checkpoint_lib.md#CheckpointFormatRegistry), [`transforms`](checkpoint_lib.md#Gemma3pFormat.transforms), [`transforms`](checkpoint_lib.md#Qwen2Format.transforms), [`fullname`](registry.md#RootRegistry.fullname), [`transforms`](checkpoint_lib.md#LegacyFormat.transforms)
- used by: [`load_checkpoint_from_path`](checkpoint_lib.md#load_checkpoint_from_path), [`load_checkpoint_from_dir`](checkpoint_lib.md#load_checkpoint_from_dir), [`save_checkpoint`](checkpoint_lib.md#save_checkpoint), [`Qwen2Format`](checkpoint_lib.md#Qwen2Format), [`transform_state_fn`](checkpoint_lib.md#load_checkpoint_from_path.transform_state_fn), [`V2Format`](checkpoint_lib.md#V2Format), [`LegacyFormat`](checkpoint_lib.md#LegacyFormat), [`load_checkpoint_from_manager`](checkpoint_lib.md#load_checkpoint_from_manager), [`Gemma3pFormat`](checkpoint_lib.md#Gemma3pFormat)

### `CheckpointFormatRegistry`
- def: [`simply/utils/checkpoint_lib.py:73`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L73) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- members:
  - `namespace` — [`L76`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L76)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`FunctionRegistry`](registry.md#FunctionRegistry)
- used by: [`load_checkpoint_from_path`](checkpoint_lib.md#load_checkpoint_from_path), [`update_params_from_checkpoint_path`](../serving/page_batcher.md#Batcher.update_params_from_checkpoint_path), [`main`](../tools/hf_to_orbax.md#main), [`Qwen2Format`](checkpoint_lib.md#Qwen2Format), [`Gemma2Format`](checkpoint_lib.md#Gemma2Format), [`Gemma3pLegacyFormat`](checkpoint_lib.md#Gemma3pLegacyFormat), [`V2Format`](checkpoint_lib.md#V2Format), [`DefaultFormat`](checkpoint_lib.md#DefaultFormat), [`__post_init__`](checkpoint_lib.md#CheckpointFormat.__post_init__), [`transpose_ffn_weights`](checkpoint_lib.md#Gemma3pLegacyFormat.transpose_ffn_weights)

### `Gemma2Format`
- def: [`simply/utils/checkpoint_lib.py:255`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L255)
- doc: Gemma2 checkpoint format.
- signature: `class Gemma2Format(Gemma3pLegacyFormat):`
- members:
  - `prefix_mapping(self)` — [`L259`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L259)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`CheckpointFormatRegistry`](checkpoint_lib.md#CheckpointFormatRegistry), [`Gemma3pLegacyFormat`](checkpoint_lib.md#Gemma3pLegacyFormat)
- used by: [`test_restore_gemma2_format`](checkpoint_lib_test.md#CheckpointFormatTest.test_restore_gemma2_format)

### `Gemma2TransposeFormat`
- def: [`simply/utils/checkpoint_lib.py:241`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L241)
- doc: Gemma2 checkpoint format with transposed ffn weights.
- signature: `class Gemma2TransposeFormat(Gemma3pFormat):`
- members:
  - `prefix_mapping(self)` — [`L245`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L245)
- uses (calls/refs, reference-scoped): [`Gemma3pFormat`](checkpoint_lib.md#Gemma3pFormat)
- used by: [`test_restore_gemma2_transpose_format`](checkpoint_lib_test.md#CheckpointFormatTest.test_restore_gemma2_transpose_format)

### `Gemma3pFormat`
- def: [`simply/utils/checkpoint_lib.py:131`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L131)
- doc: Gemma third-party checkpoint format with transposed ffn weights.
- signature: `class Gemma3pFormat(CheckpointFormat):`
- members:
  - `ln_mapping(self)` — [`L141`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L141)
  - `prefix_mapping(self)` — [`L137`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L137)
  - `transforms(self, stored_state: PyTree, target_abstract_state: PyTree = None)` — [`L149`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L149) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
  - `transpose_ffn_weights` — [`L134`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L134)
- uses (calls/refs, reference-scoped): [`PyTree`](checkpoint_lib.md#PyTree), [`CheckpointFormat`](checkpoint_lib.md#CheckpointFormat), [`LegacyFormat`](checkpoint_lib.md#LegacyFormat)
- used by: [`transforms`](checkpoint_lib.md#CheckpointFormat.transforms), [`Gemma3pLegacyFormat`](checkpoint_lib.md#Gemma3pLegacyFormat), [`Gemma2TransposeFormat`](checkpoint_lib.md#Gemma2TransposeFormat)

### `Gemma3pLegacyFormat`
- def: [`simply/utils/checkpoint_lib.py:233`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L233)
- doc: Gemma third-party checkpoint format without transposed ffn weights.
- signature: `class Gemma3pLegacyFormat(Gemma3pFormat):`
- members:
  - `transpose_ffn_weights` — [`L236`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L236)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`CheckpointFormatRegistry`](checkpoint_lib.md#CheckpointFormatRegistry), [`Gemma3pFormat`](checkpoint_lib.md#Gemma3pFormat)
- used by: [`Gemma2Format`](checkpoint_lib.md#Gemma2Format)

### `LegacyFormat`
- def: [`simply/utils/checkpoint_lib.py:81`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L81) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- doc: The legacy checkpoint format this project is using at the beginning.
- signature: `class LegacyFormat(CheckpointFormat):`
- members:
  - `transforms(self, stored_state: PyTree, target_abstract_state: PyTree = None)` — [`L84`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L84)
- uses (calls/refs, reference-scoped): [`PyTree`](checkpoint_lib.md#PyTree), [`CheckpointFormat`](checkpoint_lib.md#CheckpointFormat)
- used by: [`load_checkpoint_from_path`](checkpoint_lib.md#load_checkpoint_from_path), [`test_restore_with_format`](checkpoint_lib_test.md#CheckpointFormatTest.test_restore_with_format), [`transforms`](checkpoint_lib.md#Gemma3pFormat.transforms), [`transforms`](checkpoint_lib.md#Qwen2Format.transforms), [`transforms`](checkpoint_lib.md#CheckpointFormat.transforms), [`test_dump_format`](checkpoint_lib_test.md#CheckpointLibTest.test_dump_format)

### `Qwen2Format`
- def: [`simply/utils/checkpoint_lib.py:269`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L269)
- doc: Qwen2 checkpoint format.
- signature: `class Qwen2Format(CheckpointFormat):`
- members:
  - `_split_head(self, v: jax.Array, per_head_dim: int, axis: int = 0)` — [`L282`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L282) — Splits combined head into multiple heads.
  - `transforms(self, stored_state: PyTree, target_abstract_state: PyTree = None)` — [`L307`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L307) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- protocol/private: `_gather_experts`[`L272`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L272)
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`with_sharding_constraint`](sharding.md#with_sharding_constraint), [`PyTree`](checkpoint_lib.md#PyTree), [`tree_value`](pytree.md#tree_value), [`CheckpointFormatRegistry`](checkpoint_lib.md#CheckpointFormatRegistry), [`CheckpointFormat`](checkpoint_lib.md#CheckpointFormat), [`partition_with_minimum_redundancy`](sharding.md#partition_with_minimum_redundancy), [`LegacyFormat`](checkpoint_lib.md#LegacyFormat), `axis_sizes`
- used by: [`test_restore_qwen3_moe_format`](checkpoint_lib_test.md#QwenFormatTest.test_restore_qwen3_moe_format), [`test_restore_qwen2_format`](checkpoint_lib_test.md#QwenFormatTest.test_restore_qwen2_format), [`transforms`](checkpoint_lib.md#CheckpointFormat.transforms)

### `V2Format`
- def: [`simply/utils/checkpoint_lib.py:122`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L122) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- doc: Current format that modulizes a lot of model components.
- signature: `class V2Format(CheckpointFormat):`
- uses (calls/refs, reference-scoped): [`register`](registry.md#RootRegistry.register), [`CheckpointFormatRegistry`](checkpoint_lib.md#CheckpointFormatRegistry), [`CheckpointFormat`](checkpoint_lib.md#CheckpointFormat)
- used by: [`DefaultFormat`](checkpoint_lib.md#DefaultFormat)

## Functions
- `_get_regularized_value(path: jax.tree_util.KeyPath, abstract: jax.ShapeDtypeStruct | jax.Array)` — [`L631`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L631)
- `_restore_leaf_dtype(leaf_dtype: jax.typing.DTypeLike)` — [`L551`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L551)
- `construct_restore_item(x: PyTree, restore_dtype: jax.typing.DTypeLike | None = None)` — [`L535`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L535) — Constructs a restore item from a PyTree. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `get_abstract_params(model: module.SimplyModule)` — [`L741`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L741) — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `get_checkpoint_path(ckpt_dir: str, ckpt_step: int = -1)` — [`L443`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L443) — Returns the checkpoint path for the given ckpt_dir and ckpt_step.
- `last_checkpoint_step(ckpt_dir: epath.PathLike)` — [`L431`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L431)
- `load_checkpoint_from_dir(ckpt_dir: str, abstract_state: PyTree, ckpt_step: int = -1, ckpt_format: CheckpointFormat | str = '')` — [`L463`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L463) — Loads a checkpoint at ckpt_step in the format of abstract_state. — documented in [simply-serving-vanilla_server](../../../concepts/simply-serving-vanilla_server.md)
- `load_checkpoint_from_manager(checkpoint_manager: ocp.CheckpointManager, abstract_state: PyTree, ckpt_step: int = -1, ckpt_format: CheckpointFormat | str = '')` — [`L416`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L416) — Loads a checkpoint at ckpt_step in the format of abstract_state. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `load_checkpoint_from_path(ckpt_path: str, abstract_state: PyTree, ckpt_format: CheckpointFormat | str = '')` — [`L583`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L583) — Loads a checkpoint in the format of abstract_state using ckpt_format. — documented in [simply-model_lib](../../../concepts/simply-model_lib.md)
- `load_data_state_from_dir(ckpt_dir: str, ckpt_step: int = -1)` — [`L702`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L702) — Loads data from a checkpoint at ckpt_step.
- `readonly_checkpoint_manager(ckpt_dir: str)` — [`L396`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L396) — Returns a readonly checkpoint manager for the given ckpt_dir.
- `resolve_checkpoint_handler_from_json(handler_in_json: PyTree, restore_concurrent_gb: int | None = None)` — [`L477`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L477) — Resolves a checkpoint handler from a handler represented in json. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `resolve_checkpoint_handler_from_path(ckpt_path: str, restore_concurrent_gb: int | None = None)` — [`L500`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L500) — Resolves a checkpoint handler from a checkpoint path.
- `save_checkpoint(checkpoint_manager: ocp.CheckpointManager, state: PyTree, ckpt_step: int, ckpt_format: CheckpointFormat = DefaultFormat(), data: PyTree | None = None, **kwargs: Any)` — [`L716`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L716) — Saves a checkpoint at ckpt_step in ckpt_format. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `transform_state_fn(stored_state: PyTree)` — [`L627`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L627)

## Module values
- `CHECKPOINT_FORMAT_KEY` — [`L41`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L41)
- `DATA_ITEM_NAME` — [`L42`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L42)
- `DefaultFormat` — [`L126`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L126)
- `PyTree` — [`L39`](../../../../../../raw/code/simply/simply/utils/checkpoint_lib.py#L39)

