---
title: 'Module: python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py'
type: catalog
provenance: extracted
module: python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py
status: fresh
symbol_base: scip-python python sglang-jax 0.0.0 `python.sgl_jax.srt.layers.attention.fla.group_rmsnorm`/GroupRMSNorm#
symbols:
  GroupRMSNorm.__call__: __call__().
  GroupRMSNorm.weight: weight.
  GroupRMSNorm.mesh: mesh.
  GroupRMSNorm.num_groups: num_groups.
  GroupRMSNorm.group_size: group_size.
  GroupRMSNorm.epsilon: epsilon.
  GroupRMSNorm: ''
  GroupRMSNorm.__init__: __init__().
  GroupRMSNorm.hidden_size: hidden_size.
  GroupRMSNorm.name: name.
---
# Module: [`python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py)

## Classes
### `GroupRMSNorm`  ·  implements/extends Module
- def: [`python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py:14`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L14)
- doc: Group RMS normalization.
- signature: `class GroupRMSNorm(nnx.Module):`
- members:
  - `epsilon` — [`L46`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L46)
  - `group_size` — [`L45`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L45)
  - `hidden_size` — [`L43`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L43)
  - `mesh` — [`L48`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L48)
  - `name` — [`L47`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L47)
  - `num_groups` — [`L44`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L44)
  - `weight` — [`L49`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L49)
- protocol/private: `__call__`[`L59`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L59), `__init__`[`L21`](../../../../../../../../../../raw/code/sglang-jax/python/sgl_jax/srt/layers/attention/fla/group_rmsnorm.py#L21)
- uses (calls/refs, reference-scoped): [`named_scope`](../../../utils/profiling_utils.md#named_scope)

