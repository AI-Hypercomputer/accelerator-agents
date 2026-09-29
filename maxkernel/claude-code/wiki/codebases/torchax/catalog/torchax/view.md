---
title: 'Module: torchax/view.py'
type: catalog
provenance: extracted
module: torchax/view.py
status: fresh
symbol_base: scip-python python torchax 0.0.0 `torchax.view`/
symbols:
  View: View#
  View.update: View#update().
  ViewInfo: ViewInfo#
  View.get_transformation_chain: View#get_transformation_chain().
  SelectInfo.__eq__: SelectInfo#__eq__().
  View.torch: View#torch().
  ViewInfo.transform_tensor: ViewInfo#transform_tensor().
  View.jax: View#jax().
  ViewInfo.update_tensor: ViewInfo#update_tensor().
  ViewInfo.calculate_output_shape: ViewInfo#calculate_output_shape().
  View.source_jax: View#source_jax().
  DiagonalInfo.__eq__: DiagonalInfo#__eq__().
  View.__new__: View#__new__().
  View.replace_source_jax: View#replace_source_jax().
  ViewInfo.__init__: ViewInfo#__init__().
  View.parent: View#parent.
  NarrowInfo.__init__: NarrowInfo#__init__().
  SelectInfo.__init__: SelectInfo#__init__().
  AsStridedInfo.__init__: AsStridedInfo#__init__().
  AsStridedInfo.__eq__: AsStridedInfo#__eq__().
  DiagonalInfo.__init__: DiagonalInfo#__init__().
  View.create_sub_view: View#create_sub_view().
  View.__setitem__: View#__setitem__().
  NarrowInfo: NarrowInfo#
  ViewInfoType: ViewInfoType#
  NarrowInfo.slices: NarrowInfo#slices.
  NarrowInfo.__eq__: NarrowInfo#__eq__().
  View.__init__: View#__init__().
  SelectInfo: SelectInfo#
  AsStridedInfo: AsStridedInfo#
  DiagonalInfo: DiagonalInfo#
  View.__str__: View#__str__().
  NarrowInfo.transform_tensor: NarrowInfo#transform_tensor().
  NarrowInfo.update_tensor: NarrowInfo#update_tensor().
  NarrowInfo.calculate_output_shape: NarrowInfo#calculate_output_shape().
  View.dim: View#dim().
  View.jax_device: View#jax_device().
  SelectInfo.dim: SelectInfo#dim.
  SelectInfo.start: SelectInfo#start.
  SelectInfo.end: SelectInfo#end.
  SelectInfo.stride: SelectInfo#stride.
  AsStridedInfo.stride: AsStridedInfo#stride.
  AsStridedInfo.offset: AsStridedInfo#offset.
  DiagonalInfo.offset: DiagonalInfo#offset.
  DiagonalInfo.dim1: DiagonalInfo#dim1.
  DiagonalInfo.dim2: DiagonalInfo#dim2.
  View.view_info: View#view_info.
  View._env: View#_env.
  ViewInfoType.INVALID: ViewInfoType#INVALID.
  ViewInfoType.NARROW: ViewInfoType#NARROW.
  ViewInfoType.SELECT: ViewInfoType#SELECT.
  ViewInfoType.AS_STRIDED: ViewInfoType#AS_STRIDED.
  ViewInfoType.DIAGONAL: ViewInfoType#DIAGONAL.
  View.ndim: View#ndim().
  ViewInfoType.NO_OP: ViewInfoType#NO_OP.
  ViewInfoType.PERMUTE: ViewInfoType#PERMUTE.
  ViewInfoType.RESHAPE: ViewInfoType#RESHAPE.
  ViewInfoType.RESIZE: ViewInfoType#RESIZE.
  ViewInfo.view_info_type: ViewInfo#view_info_type.
  SelectInfo.transform_tensor: SelectInfo#transform_tensor().
  SelectInfo.update_tensor: SelectInfo#update_tensor().
  SelectInfo.calculate_output_shape: SelectInfo#calculate_output_shape().
  AsStridedInfo.transform_tensor: AsStridedInfo#transform_tensor().
  AsStridedInfo.update_tensor: AsStridedInfo#update_tensor().
  AsStridedInfo.calculate_output_shape: AsStridedInfo#calculate_output_shape().
  DiagonalInfo.transform_tensor: DiagonalInfo#transform_tensor().
  DiagonalInfo.update_tensor: DiagonalInfo#update_tensor().
  DiagonalInfo.calculate_output_shape: DiagonalInfo#calculate_output_shape().
  View.__torch_function__: View#__torch_function__.
  View.__torch_dispatch__: View#__torch_dispatch__().
  View.device: View#device().
  View.__repr__: View#__repr__.
---
# Module: [`torchax/view.py`](../../../../../raw/code/torchax/torchax/view.py)

## Classes
### `AsStridedInfo`  ·  implements/extends ViewInfo
- def: [`torchax/view.py:167`](../../../../../raw/code/torchax/torchax/view.py#L167) — documented in [torchax-view](../../concepts/torchax-view.md)
- doc: Information for as_strided operations.
- signature: `class AsStridedInfo(ViewInfo):`
- members:
  - `calculate_output_shape(self, source: jax.Array)` — [`L188`](../../../../../raw/code/torchax/torchax/view.py#L188)
  - `transform_tensor(self, jax_array: jax.Array)` — [`L182`](../../../../../raw/code/torchax/torchax/view.py#L182)
  - `update_tensor(self, new_value: jax.Array, jax_array: jax.Array)` — [`L185`](../../../../../raw/code/torchax/torchax/view.py#L185)
  - `offset` — [`L175`](../../../../../raw/code/torchax/torchax/view.py#L175)
  - `stride` — [`L174`](../../../../../raw/code/torchax/torchax/view.py#L174)
- protocol/private: `__eq__`[`L177`](../../../../../raw/code/torchax/torchax/view.py#L177), `__init__`[`L172`](../../../../../raw/code/torchax/torchax/view.py#L172)
- uses (calls/refs, reference-scoped): [`ViewInfo`](view.md#ViewInfo), [`__init__`](view.md#ViewInfo.__init__), [`ViewInfoType`](view.md#ViewInfoType), [`AS_STRIDED`](view.md#ViewInfoType.AS_STRIDED)
- used by: [`ViewInfo`](view.md#ViewInfo), [`transform_tensor`](view.md#ViewInfo.transform_tensor), [`calculate_output_shape`](view.md#ViewInfo.calculate_output_shape), [`update_tensor`](view.md#ViewInfo.update_tensor)

### `DiagonalInfo`  ·  implements/extends ViewInfo
- def: [`torchax/view.py:192`](../../../../../raw/code/torchax/torchax/view.py#L192) — documented in [torchax-view](../../concepts/torchax-view.md)
- doc: Information for diagonal operations.
- signature: `class DiagonalInfo(ViewInfo):`
- members:
  - `__init__(self, offset: int = 0, dim1: int = 0, dim2: int = 1)` — [`L198`](../../../../../raw/code/torchax/torchax/view.py#L198) — Args:
  - `calculate_output_shape(self, source: jax.Array)` — [`L225`](../../../../../raw/code/torchax/torchax/view.py#L225)
  - `transform_tensor(self, jax_array: jax.Array)` — [`L219`](../../../../../raw/code/torchax/torchax/view.py#L219)
  - `update_tensor(self, new_value: jax.Array, jax_array: jax.Array)` — [`L222`](../../../../../raw/code/torchax/torchax/view.py#L222)
  - `dim1` — [`L207`](../../../../../raw/code/torchax/torchax/view.py#L207)
  - `dim2` — [`L208`](../../../../../raw/code/torchax/torchax/view.py#L208)
  - `offset` — [`L206`](../../../../../raw/code/torchax/torchax/view.py#L206)
- protocol/private: `__eq__`[`L210`](../../../../../raw/code/torchax/torchax/view.py#L210)
- uses (calls/refs, reference-scoped): [`ViewInfo`](view.md#ViewInfo), [`__init__`](view.md#ViewInfo.__init__), [`ViewInfoType`](view.md#ViewInfoType), [`DIAGONAL`](view.md#ViewInfoType.DIAGONAL)
- used by: [`ViewInfo`](view.md#ViewInfo), [`transform_tensor`](view.md#ViewInfo.transform_tensor), [`calculate_output_shape`](view.md#ViewInfo.calculate_output_shape), [`update_tensor`](view.md#ViewInfo.update_tensor)

### `NarrowInfo`  ·  implements/extends ViewInfo
- def: [`torchax/view.py:99`](../../../../../raw/code/torchax/torchax/view.py#L99) — documented in [torchax-view](../../concepts/torchax-view.md)
- doc: Represents a slicing operation on a tensor.
- signature: `class NarrowInfo(ViewInfo):`
- members:
  - `__init__(self, slices: slice | tuple[slice])` — [`L105`](../../../../../raw/code/torchax/torchax/view.py#L105) — Args:
  - `calculate_output_shape(self, source: jax.Array)` — [`L128`](../../../../../raw/code/torchax/torchax/view.py#L128)
  - `transform_tensor(self, jax_array: jax.Array)` — [`L119`](../../../../../raw/code/torchax/torchax/view.py#L119)
  - `update_tensor(self, new_value: jax.Array, jax_array: jax.Array)` — [`L125`](../../../../../raw/code/torchax/torchax/view.py#L125) — documented in [torchax-view](../../concepts/torchax-view.md)
  - `slices` — [`L112`](../../../../../raw/code/torchax/torchax/view.py#L112)
- protocol/private: `__eq__`[`L114`](../../../../../raw/code/torchax/torchax/view.py#L114)
- uses (calls/refs, reference-scoped): [`ViewInfo`](view.md#ViewInfo), [`__init__`](view.md#ViewInfo.__init__), [`ViewInfoType`](view.md#ViewInfoType), [`NARROW`](view.md#ViewInfoType.NARROW)
- used by: [`ViewInfo`](view.md#ViewInfo), [`getitem`](ops/jtorch.md#getitem), [`transform_tensor`](view.md#ViewInfo.transform_tensor), [`calculate_output_shape`](view.md#ViewInfo.calculate_output_shape), [`update_tensor`](view.md#ViewInfo.update_tensor), [`__setitem__`](view.md#View.__setitem__)

### `SelectInfo`  ·  implements/extends ViewInfo
- def: [`torchax/view.py:132`](../../../../../raw/code/torchax/torchax/view.py#L132) — documented in [torchax-view](../../concepts/torchax-view.md)
- doc: Represents a selection operation on a tensor.
- signature: `class SelectInfo(ViewInfo):`
- members:
  - `calculate_output_shape(self, source: jax.Array)` — [`L163`](../../../../../raw/code/torchax/torchax/view.py#L163)
  - `transform_tensor(self, jax_array: jax.Array)` — [`L157`](../../../../../raw/code/torchax/torchax/view.py#L157)
  - `update_tensor(self, new_value: jax.Array, jax_array: jax.Array)` — [`L160`](../../../../../raw/code/torchax/torchax/view.py#L160)
  - `dim` — [`L142`](../../../../../raw/code/torchax/torchax/view.py#L142)
  - `end` — [`L144`](../../../../../raw/code/torchax/torchax/view.py#L144)
  - `start` — [`L143`](../../../../../raw/code/torchax/torchax/view.py#L143)
  - `stride` — [`L145`](../../../../../raw/code/torchax/torchax/view.py#L145)
- protocol/private: `__eq__`[`L147`](../../../../../raw/code/torchax/torchax/view.py#L147), `__init__`[`L138`](../../../../../raw/code/torchax/torchax/view.py#L138)
- uses (calls/refs, reference-scoped): [`ViewInfo`](view.md#ViewInfo), [`__init__`](view.md#ViewInfo.__init__), [`ViewInfoType`](view.md#ViewInfoType), [`SELECT`](view.md#ViewInfoType.SELECT)
- used by: [`ViewInfo`](view.md#ViewInfo), [`transform_tensor`](view.md#ViewInfo.transform_tensor), [`calculate_output_shape`](view.md#ViewInfo.calculate_output_shape), [`update_tensor`](view.md#ViewInfo.update_tensor)

### `View`
- def: [`torchax/view.py:229`](../../../../../raw/code/torchax/torchax/view.py#L229) — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
- members:
  - `__new__(cls, parent: torchax.Tensor | View, view_info: ViewInfo, env: Any)` — [`L236`](../../../../../raw/code/torchax/torchax/view.py#L236) — Args: — documented in [torchax-view](../../concepts/torchax-view.md)
  - `create_sub_view(self, view_info: ViewInfo)` — [`L362`](../../../../../raw/code/torchax/torchax/view.py#L362) — Create a new view that is a child of this view. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `device(self)` — [`L388`](../../../../../raw/code/torchax/torchax/view.py#L388)
  - `dim(self)` — [`L384`](../../../../../raw/code/torchax/torchax/view.py#L384)
  - `get_transformation_chain(self)` — [`L268`](../../../../../raw/code/torchax/torchax/view.py#L268) — Get all view transformations from the source tensor to this view. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `jax(self)` — [`L371`](../../../../../raw/code/torchax/torchax/view.py#L371) — Returns a copy of the source tensor after transformations. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `jax_device(self)` — [`L392`](../../../../../raw/code/torchax/torchax/view.py#L392)
  - `ndim(self)` — [`L396`](../../../../../raw/code/torchax/torchax/view.py#L396)
  - `replace_source_jax(self, new_value: jax.Array)` — [`L290`](../../../../../raw/code/torchax/torchax/view.py#L290) — Update the source tensor with new values. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `source_jax(self)` — [`L281`](../../../../../raw/code/torchax/torchax/view.py#L281) — Returns the source tensor. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `torch(self)` — [`L300`](../../../../../raw/code/torchax/torchax/view.py#L300) — Returns a Torchax tensor representing this view after all transformations — documented in [torchax-tensor](../../concepts/torchax-tensor.md)
  - `update(self, new_values: jax.Array | View | torchax.Tensor, view_infos: list[ViewInfo] | None = None)` — [`L308`](../../../../../raw/code/torchax/torchax/view.py#L308) — Update this view with new values, propagating changes back to source. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `parent` — [`L264`](../../../../../raw/code/torchax/torchax/view.py#L264)
  - `view_info` — [`L265`](../../../../../raw/code/torchax/torchax/view.py#L265)
- protocol/private: `__init__`[`L257`](../../../../../raw/code/torchax/torchax/view.py#L257), `__repr__`[`L399`](../../../../../raw/code/torchax/torchax/view.py#L399), `__setitem__`[`L380`](../../../../../raw/code/torchax/torchax/view.py#L380), `__str__`[`L368`](../../../../../raw/code/torchax/torchax/view.py#L368), `__torch_dispatch__`[`L349`](../../../../../raw/code/torchax/torchax/view.py#L349), `__torch_function__`[`L279`](../../../../../raw/code/torchax/torchax/view.py#L279), `_env`[`L266`](../../../../../raw/code/torchax/torchax/view.py#L266)
- uses (calls/refs, reference-scoped): [`Tensor`](tensor.md#Tensor), [`ViewInfo`](view.md#ViewInfo), [`transform_tensor`](view.md#ViewInfo.transform_tensor), [`calculate_output_shape`](view.md#ViewInfo.calculate_output_shape), [`update_tensor`](view.md#ViewInfo.update_tensor), [`NarrowInfo`](view.md#NarrowInfo)
- used by: [`dispatch`](tensor.md#Environment.dispatch), [`_to_copy`](tensor.md#Environment._to_copy), [`_jax_view`](interop.md#_jax_view), [`__call__`](ops/op_base.md#InplaceOp.__call__), [`to_jax`](tensor.md#Environment.to_jax), [`__call__`](ops/op_base.md#OutVariant.__call__), [`getitem`](ops/jtorch.md#getitem), [`_aten_copy`](ops/jaten.md#_aten_copy), [`is_not_torchax_tensor`](tensor.md#Environment.is_not_torchax_tensor), [`to_tensor`](tensor.md#Environment.to_tensor), [`v2t_iso`](tensor.md#Environment.v2t_iso)  (4 test-only)

### `ViewInfo`  ·  implements/extends ABC
- def: [`torchax/view.py:40`](../../../../../raw/code/torchax/torchax/view.py#L40) — documented in [torchax-view](../../concepts/torchax-view.md)
- doc: Abstract base class for all view operations.
- signature: `class ViewInfo(ABC):`
- members:
  - `__init__(self, view_info_type: ViewInfoType = ViewInfoType.INVALID)` — [`L46`](../../../../../raw/code/torchax/torchax/view.py#L46) — Initialize a ViewInfo object.
  - `calculate_output_shape(self, source: jax.Array)` — [`L86`](../../../../../raw/code/torchax/torchax/view.py#L86) — Calculate the resulting shape after applying this view.
  - `transform_tensor(self, jax_array: jax.Array)` — [`L73`](../../../../../raw/code/torchax/torchax/view.py#L73) — Apply this view transformation to a JAX array. — documented in [torchax-view](../../concepts/torchax-view.md)
  - `update_tensor(self, new_value: jax.Array, jax_array: jax.Array)` — [`L59`](../../../../../raw/code/torchax/torchax/view.py#L59) — Apply this view transformation to a JAX array and update its value.
  - `view_info_type` — [`L56`](../../../../../raw/code/torchax/torchax/view.py#L56)
- uses (calls/refs, reference-scoped): [`NarrowInfo`](view.md#NarrowInfo), [`ViewInfoType`](view.md#ViewInfoType), [`AsStridedInfo`](view.md#AsStridedInfo), [`DiagonalInfo`](view.md#DiagonalInfo), [`SelectInfo`](view.md#SelectInfo), [`calculate_output_shape`](view.md#NarrowInfo.calculate_output_shape), [`transform_tensor`](view.md#NarrowInfo.transform_tensor), [`update_tensor`](view.md#NarrowInfo.update_tensor), [`INVALID`](view.md#ViewInfoType.INVALID), [`calculate_output_shape`](view.md#AsStridedInfo.calculate_output_shape), [`calculate_output_shape`](view.md#DiagonalInfo.calculate_output_shape), [`calculate_output_shape`](view.md#SelectInfo.calculate_output_shape), [`transform_tensor`](view.md#AsStridedInfo.transform_tensor), [`transform_tensor`](view.md#DiagonalInfo.transform_tensor), [`transform_tensor`](view.md#SelectInfo.transform_tensor), [`update_tensor`](view.md#AsStridedInfo.update_tensor), [`update_tensor`](view.md#DiagonalInfo.update_tensor), [`update_tensor`](view.md#SelectInfo.update_tensor)
- used by: [`update`](view.md#View.update), [`get_transformation_chain`](view.md#View.get_transformation_chain), [`jax`](view.md#View.jax), [`__new__`](view.md#View.__new__), [`__init__`](view.md#AsStridedInfo.__init__), [`__init__`](view.md#DiagonalInfo.__init__), [`__init__`](view.md#NarrowInfo.__init__), [`__init__`](view.md#SelectInfo.__init__), [`create_sub_view`](view.md#View.create_sub_view), [`NarrowInfo`](view.md#NarrowInfo), [`__init__`](view.md#View.__init__), [`AsStridedInfo`](view.md#AsStridedInfo), [`DiagonalInfo`](view.md#DiagonalInfo), [`SelectInfo`](view.md#SelectInfo)

### `ViewInfoType`  ·  implements/extends Enum
- def: [`torchax/view.py:28`](../../../../../raw/code/torchax/torchax/view.py#L28)
- signature: `class ViewInfoType(Enum):`
- members:
  - `AS_STRIDED` — [`L36`](../../../../../raw/code/torchax/torchax/view.py#L36)
  - `DIAGONAL` — [`L37`](../../../../../raw/code/torchax/torchax/view.py#L37)
  - `INVALID` — [`L29`](../../../../../raw/code/torchax/torchax/view.py#L29)
  - `NARROW` — [`L30`](../../../../../raw/code/torchax/torchax/view.py#L30)
  - `NO_OP` — [`L31`](../../../../../raw/code/torchax/torchax/view.py#L31)
  - `PERMUTE` — [`L32`](../../../../../raw/code/torchax/torchax/view.py#L32)
  - `RESHAPE` — [`L33`](../../../../../raw/code/torchax/torchax/view.py#L33)
  - `RESIZE` — [`L34`](../../../../../raw/code/torchax/torchax/view.py#L34)
  - `SELECT` — [`L35`](../../../../../raw/code/torchax/torchax/view.py#L35)
- used by: [`__init__`](view.md#ViewInfo.__init__), [`__init__`](view.md#AsStridedInfo.__init__), [`__init__`](view.md#DiagonalInfo.__init__), [`__init__`](view.md#NarrowInfo.__init__), [`__init__`](view.md#SelectInfo.__init__)

