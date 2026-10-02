---
title: 'Module: easydel/data/transforms/base.py'
type: catalog
provenance: extracted
module: easydel/data/transforms/base.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.data.transforms.base`/
symbols:
  Transform: Transform#
  Example: Example.
  Transform.is_filter: Transform#is_filter().
  ChainedTransform: ChainedTransform#
  ExpandTransform: ExpandTransform#
  Transform.__rshift__: Transform#__rshift__().
  ChainedTransform.__rshift__: ChainedTransform#__rshift__().
  ChainedTransform._transforms: ChainedTransform#_transforms.
  ChainedTransform.__call__: ChainedTransform#__call__().
  ChainedTransform.is_filter: ChainedTransform#is_filter().
  Transform.__call__: Transform#__call__().
  ExpandTransform.__call__: ExpandTransform#__call__().
  ChainedTransform.__init__: ChainedTransform#__init__().
  ChainedTransform.__repr__: ChainedTransform#__repr__().
  ChainedTransform.__len__: ChainedTransform#__len__().
  ChainedTransform.__iter__: ChainedTransform#__iter__().
  Transform.is_expand: Transform#is_expand().
  Transform.__repr__: Transform#__repr__().
  ExpandTransform.is_expand: ExpandTransform#is_expand().
  ExpandTransform.is_filter: ExpandTransform#is_filter().
  ExpandTransform.__repr__: ExpandTransform#__repr__().
---
# Module: [`easydel/data/transforms/base.py`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py)

## Classes
### `ChainedTransform`
- def: [`easydel/data/transforms/base.py:120`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L120)
- members:
  - `__call__(self, example: Example)` — [`L134`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L134) — Apply all transforms in sequence.
  - `__init__(self, transforms: list[Transform])` — [`L126`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L126) — Initialize ChainedTransform.
  - `__iter__(self)` — [`L167`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L167) — Iterate over transforms in the chain.
  - `__len__(self)` — [`L163`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L163) — Number of transforms in the chain.
  - `__rshift__(self, other: Transform)` — [`L150`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L150) — Append another transform to the chain.
  - `is_filter(self)` — [`L155`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L155) — True if any transform in the chain is a filter.
- protocol/private: `__repr__`[`L159`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L159), `_transforms`[`L132`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L132)
- uses (calls/refs, reference-scoped): [`Transform`](base.md#Transform), [`Example`](base.md#Example), [`is_filter`](base.md#Transform.is_filter)
- used by: [`is_filter`](base.md#Transform.is_filter), [`__rshift__`](base.md#Transform.__rshift__)

### `ExpandTransform`  ·  implements/extends ABC
- def: [`easydel/data/transforms/base.py:80`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L80)
- doc: Transform that can produce multiple examples from a single input.
- signature: `class ExpandTransform(ABC):`
- members:
  - `__call__(self, example: Example)` — [`L95`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L95) — Apply transform, yielding zero or more examples.
  - `is_expand(self)` — [`L107`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L107) — Always True for ExpandTransform.
  - `is_filter(self)` — [`L112`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L112) — ExpandTransform can filter by yielding nothing.
- protocol/private: `__repr__`[`L116`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L116)
- uses (calls/refs, reference-scoped): [`Example`](base.md#Example), [`BCOPreprocessTransform`](../../trainers/prompt_transforms.md#BCOPreprocessTransform)
- used by: [`open_shard`](source.md#TransformedShardedSource.open_shard), [`open_shard_at_row`](source.md#TransformedShardedSource.open_shard_at_row), [`BCOPreprocessTransform`](../../trainers/prompt_transforms.md#BCOPreprocessTransform), [`__init__`](source.md#TransformedShardedSource.__init__)

### `Transform`  ·  implements/extends ABC
- def: [`easydel/data/transforms/base.py:32`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L32)
- doc: Base class for all transforms.
- signature: `class Transform(ABC):`
- members:
  - `__call__(self, example: Example)` — [`L45`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L45) — Apply transform to a single example.
  - `__rshift__(self, other: Transform)` — [`L56`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L56) — Chain transforms using >> operator.
  - `is_expand(self)` — [`L72`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L72) — Whether this transform can produce multiple examples from one input.
  - `is_filter(self)` — [`L67`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L67) — Whether this transform can filter out examples (return None).
- protocol/private: `__repr__`[`L76`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L76)
- uses (calls/refs, reference-scoped): [`Example`](base.md#Example), [`GRPOPreprocessTransform`](../../trainers/prompt_transforms.md#GRPOPreprocessTransform), [`DPOPreprocessTransform`](../../trainers/prompt_transforms.md#DPOPreprocessTransform), [`ChainedTransform`](base.md#ChainedTransform), [`SFTPreprocessTransform`](../../trainers/prompt_transforms.md#SFTPreprocessTransform), [`KTOPreprocessTransform`](../../trainers/prompt_transforms.md#KTOPreprocessTransform), [`RenameFields`](field_ops.md#RenameFields), [`RewardPreprocessTransform`](../../trainers/prompt_transforms.md#RewardPreprocessTransform), [`_transforms`](base.md#ChainedTransform._transforms), [`MapTransform`](map_ops.md#MapTransform), [`ChatTemplateTransform`](chat_template.md#ChatTemplateTransform), [`EmbeddingPreprocessTransform`](../../trainers/prompt_transforms.md#EmbeddingPreprocessTransform), [`is_filter`](base.md#ChainedTransform.is_filter), [`AddField`](field_ops.md#AddField), [`CombineFields`](field_ops.md#CombineFields), [`ConvertInputOutputToChatML`](chat_template.md#ConvertInputOutputToChatML), [`ConvertToChatML`](chat_template.md#ConvertToChatML), [`DropFields`](field_ops.md#DropFields), [`ExtractField`](field_ops.md#ExtractField), [`FilterByField`](filter_ops.md#FilterByField), [`FilterNonEmpty`](filter_ops.md#FilterNonEmpty), [`FilterTransform`](filter_ops.md#FilterTransform), [`MapField`](map_ops.md#MapField), [`MaybeApplyChatTemplate`](chat_template.md#MaybeApplyChatTemplate), [`SelectFields`](field_ops.md#SelectFields), [`is_filter`](filter_ops.md#FilterByField.is_filter), [`is_filter`](filter_ops.md#FilterNonEmpty.is_filter), [`is_filter`](filter_ops.md#FilterTransform.is_filter)
- used by: [`_get_preprocess_transform`](../../trainers/base_trainer.md#BaseTrainer._get_preprocess_transform), [`GRPOPreprocessTransform`](../../trainers/prompt_transforms.md#GRPOPreprocessTransform), [`DPOPreprocessTransform`](../../trainers/prompt_transforms.md#DPOPreprocessTransform), [`ChainedTransform`](base.md#ChainedTransform), [`SFTPreprocessTransform`](../../trainers/prompt_transforms.md#SFTPreprocessTransform), [`KTOPreprocessTransform`](../../trainers/prompt_transforms.md#KTOPreprocessTransform), [`RenameFields`](field_ops.md#RenameFields), [`RewardPreprocessTransform`](../../trainers/prompt_transforms.md#RewardPreprocessTransform), [`__init__`](source.md#TransformedShardedSource.__init__), [`__rshift__`](base.md#ChainedTransform.__rshift__), [`MapTransform`](map_ops.md#MapTransform), [`ChatTemplateTransform`](chat_template.md#ChatTemplateTransform), [`EmbeddingPreprocessTransform`](../../trainers/prompt_transforms.md#EmbeddingPreprocessTransform), [`is_filter`](base.md#ChainedTransform.is_filter), [`AddField`](field_ops.md#AddField), [`CombineFields`](field_ops.md#CombineFields), [`ConvertInputOutputToChatML`](chat_template.md#ConvertInputOutputToChatML), [`ConvertToChatML`](chat_template.md#ConvertToChatML), [`DropFields`](field_ops.md#DropFields), [`ExtractField`](field_ops.md#ExtractField), [`FilterByField`](filter_ops.md#FilterByField), [`FilterNonEmpty`](filter_ops.md#FilterNonEmpty), [`FilterTransform`](filter_ops.md#FilterTransform), [`MapField`](map_ops.md#MapField), [`MaybeApplyChatTemplate`](chat_template.md#MaybeApplyChatTemplate), [`SelectFields`](field_ops.md#SelectFields), [`__init__`](base.md#ChainedTransform.__init__)

## Module values
- `Example` — [`L29`](../../../../../../../raw/code/EasyDeL/easydel/data/transforms/base.py#L29)

