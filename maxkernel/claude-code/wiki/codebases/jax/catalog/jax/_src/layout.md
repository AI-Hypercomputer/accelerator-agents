---
title: 'Module: jax/_src/layout.py'
type: catalog
provenance: extracted
module: jax/_src/layout.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.layout`/
symbols:
  Format: Format#
  Layout: Layout#
  Format.__init__: Format#__init__().
  AutoLayoutSingleton: AutoLayoutSingleton#
  get_layout_for_vmap: get_layout_for_vmap().
  Layout._to_xla_layout: Layout#_to_xla_layout().
  Layout.major_to_minor: Layout#major_to_minor.
  Layout.update: Layout#update().
  Layout.__eq__: Layout#__eq__().
  Layout.tiling: Layout#tiling.
  Layout.sub_byte_element_size_in_bits: Layout#sub_byte_element_size_in_bits.
  Layout.__init__: Layout#__init__().
  Layout.__repr__: Layout#__repr__().
  Layout.__hash__: Layout#__hash__().
  LayoutOptions: LayoutOptions.
  Layout.check_compatible_aval: Layout#check_compatible_aval().
  AutoLayout: AutoLayout.
  Layout.from_pjrt_layout: Layout#from_pjrt_layout().
  ShardingOptions: ShardingOptions.
  Layout.AUTO: Layout#AUTO.
  Format.__repr__: Format#__repr__().
  Format.__hash__: Format#__hash__().
  Format.__eq__: Format#__eq__().
  Shape: Shape.
  AutoLayoutSingleton.__repr__: AutoLayoutSingleton#__repr__().
  Format.__slots__: Format#__slots__.
---
# Module: [`jax/_src/layout.py`](../../../../../../raw/code/jax/jax/_src/layout.py)

## Classes
### `AutoLayoutSingleton`
- def: [`jax/_src/layout.py:27`](../../../../../../raw/code/jax/jax/_src/layout.py#L27)
- signature: `class AutoLayoutSingleton:`
- protocol/private: `__repr__`[`L29`](../../../../../../raw/code/jax/jax/_src/layout.py#L29)
- used by: [`make_array_from_callback`](array.md#make_array_from_callback), [`__init__`](core.md#ShapeDtypeStruct.__init__), [`const_args_layouts`](pjit.md#const_args_layouts), [`_split_layout_and_sharding`](pjit.md#_split_layout_and_sharding), [`__init__`](layout.md#Format.__init__), [`_resolve_in_layouts`](pjit.md#_resolve_in_layouts), [`check_aval_layout_compatibility`](pjit.md#check_aval_layout_compatibility), [`LayoutOptions`](layout.md#LayoutOptions), [`AutoLayout`](layout.md#AutoLayout)

### `Format`
- def: [`jax/_src/layout.py:108`](../../../../../../raw/code/jax/jax/_src/layout.py#L108)
- signature: `class Format:`
- protocol/private: `__eq__`[`L145`](../../../../../../raw/code/jax/jax/_src/layout.py#L145), `__hash__`[`L142`](../../../../../../raw/code/jax/jax/_src/layout.py#L142), `__init__`[`L111`](../../../../../../raw/code/jax/jax/_src/layout.py#L111), `__repr__`[`L139`](../../../../../../raw/code/jax/jax/_src/layout.py#L139), `__slots__`[`L109`](../../../../../../raw/code/jax/jax/_src/layout.py#L109)
- uses (calls/refs, reference-scoped): [`Sharding`](sharding.md#Sharding), [`Layout`](layout.md#Layout), [`AutoLayoutSingleton`](layout.md#AutoLayoutSingleton), [`LayoutOptions`](layout.md#LayoutOptions), [`ShardingOptions`](layout.md#ShardingOptions)
- used by: [`make_array_from_callback`](array.md#make_array_from_callback), [`device_put`](api.md#device_put), [`_sharding_constraint_impl`](pjit.md#_sharding_constraint_impl), [`_check_sharding`](api.md#_check_sharding), [`__init__`](core.md#ShapeDtypeStruct.__init__), [`_array_shard_arg`](array.md#_array_shard_arg), [`_split_layout_and_sharding`](pjit.md#_split_layout_and_sharding), [`format`](array.md#ArrayImpl.format), [`format`](core.md#ShapeDtypeStruct.format), [`_layout_constraint_impl`](pjit.md#_layout_constraint_impl)

### `Layout`
- def: [`jax/_src/layout.py:34`](../../../../../../raw/code/jax/jax/_src/layout.py#L34)
- signature: `class Layout:`
- members:
  - `check_compatible_aval(self, aval_shape: Shape)` — [`L97`](../../../../../../raw/code/jax/jax/_src/layout.py#L97)
  - `from_pjrt_layout(pjrt_layout: xc.PjRtLayout)` — [`L49`](../../../../../../raw/code/jax/jax/_src/layout.py#L49)
  - `update(self, **kwargs)` — [`L73`](../../../../../../raw/code/jax/jax/_src/layout.py#L73)
  - `AUTO` — [`L39`](../../../../../../raw/code/jax/jax/_src/layout.py#L39)
  - `major_to_minor` — [`L35`](../../../../../../raw/code/jax/jax/_src/layout.py#L35)
  - `sub_byte_element_size_in_bits` — [`L37`](../../../../../../raw/code/jax/jax/_src/layout.py#L37)
  - `tiling` — [`L36`](../../../../../../raw/code/jax/jax/_src/layout.py#L36)
- protocol/private: `__eq__`[`L66`](../../../../../../raw/code/jax/jax/_src/layout.py#L66), `__hash__`[`L62`](../../../../../../raw/code/jax/jax/_src/layout.py#L62), `__init__`[`L41`](../../../../../../raw/code/jax/jax/_src/layout.py#L41), `__repr__`[`L55`](../../../../../../raw/code/jax/jax/_src/layout.py#L55), `_to_xla_layout`[`L83`](../../../../../../raw/code/jax/jax/_src/layout.py#L83)
- uses (calls/refs, reference-scoped): [`issubdtype`](dtypes.md#issubdtype), [`iinfo`](dtypes.md#iinfo), [`AutoLayout`](layout.md#AutoLayout), [`Shape`](layout.md#Shape)
- used by: [`with_layout_constraint`](pjit.md#with_layout_constraint), [`const_args_layouts`](pjit.md#const_args_layouts), [`_split_layout_and_sharding`](pjit.md#_split_layout_and_sharding), [`__init__`](layout.md#Format.__init__), [`format`](array.md#ArrayImpl.format), [`get_layout_for_vmap`](layout.md#get_layout_for_vmap), [`_resolve_in_layouts`](pjit.md#_resolve_in_layouts), [`LayoutOptions`](layout.md#LayoutOptions), [`_resolve_out_layouts`](pjit.md#_resolve_out_layouts)

## Functions
- `get_layout_for_vmap(dim: int, layout: Layout)` — [`L152`](../../../../../../raw/code/jax/jax/_src/layout.py#L152)

## Module values
- `AutoLayout` — [`L32`](../../../../../../raw/code/jax/jax/_src/layout.py#L32)
- `LayoutOptions` — [`L104`](../../../../../../raw/code/jax/jax/_src/layout.py#L104)
- `Shape` — [`L25`](../../../../../../raw/code/jax/jax/_src/layout.py#L25)
- `ShardingOptions` — [`L105`](../../../../../../raw/code/jax/jax/_src/layout.py#L105)

