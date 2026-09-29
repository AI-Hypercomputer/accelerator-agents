---
title: 'Module: jax/_src/hijax.py'
type: catalog
provenance: extracted
module: jax/_src/hijax.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.hijax`/
symbols:
  CustomVJPTraced.vjp_fwd: CustomVJPTraced#vjp_fwd().
  CustomVJPTraced.vjp_bwd_retval: CustomVJPTraced#vjp_bwd_retval().
  custom_vjp3.__call__: custom_vjp3#__call__().
  CustomVJPTraced.jvp: CustomVJPTraced#jvp().
  call_hi_primitive_p: call_hi_primitive_p.
  _vjp_bwd_aval_mismatch_err: _vjp_bwd_aval_mismatch_err().
  HiType: HiType#
  HiPrimitive: HiPrimitive#
  VJPHiPrimitive.out_tree: VJPHiPrimitive#out_tree.
  VJPHiPrimitive.expand: VJPHiPrimitive#expand().
  VmapOf.vjp_bwd_retval: VmapOf#vjp_bwd_retval().
  CustomVJPTraced.remat: CustomVJPTraced#remat().
  VJPHiPrimitive: VJPHiPrimitive#
  VJPHiPrimitive.in_avals: VJPHiPrimitive#in_avals.
  call_hi_primitive_linearized_p: call_hi_primitive_linearized_p.
  BoxTy.new_from_loval: BoxTy#new_from_loval().
  VJPHiPrimitive.jvp: VJPHiPrimitive#jvp().
  HiPrimitive.to_lojax: HiPrimitive#to_lojax().
  VJPHiPrimitive.__init__: VJPHiPrimitive#__init__().
  HiPrimitive.abstract_eval: HiPrimitive#abstract_eval().
  VJPHiPrimitive.out_aval: VJPHiPrimitive#out_aval.
  BoxTy.update_from_loval: BoxTy#update_from_loval().
  VJPHiPrimitive.vjp_fwd: VJPHiPrimitive#vjp_fwd().
  VJPHiPrimitive.__call__: VJPHiPrimitive#__call__().
  VmapOf.__init__: VmapOf#__init__().
  VmapOf.vjp_fwd: VmapOf#vjp_fwd().
  fake_linear_op: fake_linear_op().
  _call_hi_primitive_linearize: _call_hi_primitive_linearize().
  box_get: box_get().
  VJPHiPrimitive.params: VJPHiPrimitive#params.
  VJPHiPrimitive.vjp_bwd_retval: VJPHiPrimitive#vjp_bwd_retval().
  MutableHiType.new_from_loval: MutableHiType#new_from_loval().
  BoxTy.read_loval: BoxTy#read_loval().
  NewBox.abstract_eval: NewBox#abstract_eval().
  VJPHiPrimitive.staging: VJPHiPrimitive#staging().
  VmapOf.expand: VmapOf#expand().
  VmapOf.jvp: VmapOf#jvp().
  HiPrimitive.is_high: HiPrimitive#is_high().
  MutableHiType.read_loval: MutableHiType#read_loval().
  VJPHiPrimitive.vjp_bwd: VJPHiPrimitive#vjp_bwd().
  HiVal: HiVal.
  MutableHiType.update_from_loval: MutableHiType#update_from_loval().
  Box.type_state: Box#type_state().
  VJPHiPrimitive.in_tree: VJPHiPrimitive#in_tree.
  box_set: box_set().
  BoxTypeState: BoxTypeState#
  log_extend: log_extend().
  MutableHiType.lo_ty_qdd: MutableHiType#lo_ty_qdd().
  BoxTypeState.to_tangent_qdd: BoxTypeState#to_tangent_qdd().
  BoxTypeState.normalize: BoxTypeState#normalize().
  VJPHiPrimitive.lin: VJPHiPrimitive#lin().
  VJPHiPrimitive.linearized: VJPHiPrimitive#linearized().
  VJPHiPrimitive.batch: VJPHiPrimitive#batch().
  VmapOf.batch_dim_rule: VmapOf#batch_dim_rule().
  CustomVJPTraced.__init__: CustomVJPTraced#__init__().
  CustomVJPTraced.batch_dim_rule: CustomVJPTraced#batch_dim_rule().
  custom_vjp3.defvjp: custom_vjp3#defvjp().
  OptRemat.__init__: OptRemat#__init__().
  HiPspec: HiPspec#
  QDD: QDD.
  HiPrimitive.jvp: HiPrimitive#jvp().
  HiPrimitive.transpose: HiPrimitive#transpose().
  MutableHiType.to_tangent_aval: MutableHiType#to_tangent_aval().
  BoxTy: BoxTy#
  _call_hi_primitive_linearized_transpose: _call_hi_primitive_linearized_transpose().
  _call_hi_primitive_remat: _call_hi_primitive_remat().
  _vjp_fwd_aval_mismatch_err: _vjp_fwd_aval_mismatch_err().
  Static: Static#
  LogTy: LogTy#
  HiPrimitive.__init__: HiPrimitive#__init__().
  new_box: new_box().
  BoxTypeState.leaf_avals: BoxTypeState#leaf_avals.
  BoxTypeState.treedef: BoxTypeState#treedef.
  _BoxMeta.__instancecheck__: _BoxMeta#__instancecheck__().
  Box.__new__: Box#__new__().
  BoxSet.jvp: BoxSet#jvp().
  VJPHiPrimitive.dce: VJPHiPrimitive#dce().
  VJPHiPrimitive.remat: VJPHiPrimitive#remat().
  VmapOf.prim: VmapOf#prim.
  _LogMeta.__instancecheck__: _LogMeta#__instancecheck__().
  NewLog.abstract_eval: NewLog#abstract_eval().
  MutableHiType: MutableHiType#
  register_hitype: register_hitype().
  VmapOf.in_dims: VmapOf#in_dims.
  VmapOf.out_dim: VmapOf#out_dim.
  custom_vjp3.f: custom_vjp3#f.
  log_read: log_read().
  box_effect: box_effect.
  VmapOf._vmap_params: VmapOf#_vmap_params().
  custom_vjp3.static_argnums: custom_vjp3#static_argnums.
  Log: Log#
  Box: Box#
  HiType.raise_val: HiType#raise_val().
  HiType.dec_rank: HiType#dec_rank().
  HiType.inc_rank: HiType#inc_rank().
  HiType.shard: HiType#shard().
  HiType.unshard: HiType#unshard().
  VJPHiPrimitive.batch_dim_rule: VJPHiPrimitive#batch_dim_rule().
  VmapOf.axis_data: VmapOf#axis_data.
  map_zero: map_zero().
  unmap_zero: unmap_zero().
  _call_hi_primitive_staging: _call_hi_primitive_staging().
  _call_hi_primitive_to_lojax: _call_hi_primitive_to_lojax().
  _call_hi_primitive_batcher: _call_hi_primitive_batcher().
  flatten_user_linearized: flatten_user_linearized().
  _call_hi_primitive_jvp: _call_hi_primitive_jvp().
  _call_hi_primitive_dce: _call_hi_primitive_dce().
  _call_hi_primitive_remat.rem: _call_hi_primitive_remat().rem().
  MappingSpec: MappingSpec#
  box_set_p: box_set_p.
  box_get_p: box_get_p.
  CustomVJPTraced: CustomVJPTraced#
  LoVal: LoVal.
  HiType.lower_val: HiType#lower_val().
  HiType.to_ct_aval: HiType#to_ct_aval().
  HiType.nospec: HiType#nospec().
  MutableHiType.type_state: MutableHiType#type_state.
  MutableHiType.read_loval_in: MutableHiType#read_loval_in().
  MutableHiType.read_loval_out: MutableHiType#read_loval_out().
  MutableHiType.update_from_loval2: MutableHiType#update_from_loval2().
  MutableHiType.to_ct_aval: MutableHiType#to_ct_aval().
  BoxTy.get: BoxTy#get.
  BoxTy.set: BoxTy#set.
  NewBox.to_lojax: NewBox#to_lojax().
  BoxSet.abstract_eval: BoxSet#abstract_eval().
  BoxGet.jvp: BoxGet#jvp().
  VJPHiPrimitive.__hash__: VJPHiPrimitive#__hash__().
  VJPHiPrimitive.__eq__: VJPHiPrimitive#__eq__().
  _call_hi_primitive_abstract_eval: _call_hi_primitive_abstract_eval().
  _call_hi_primitive_linearized_abstract_eval: _call_hi_primitive_linearized_abstract_eval().
  CustomVJPTraced.symbolic_zeros: CustomVJPTraced#symbolic_zeros.
  CustomVJPTraced.static_argnums: CustomVJPTraced#static_argnums.
  CustomVJPTraced.expand: CustomVJPTraced#expand().
  OptRemat.dce: OptRemat#dce().
  Log.__new__: Log#__new__().
  LogTy.append: LogTy#append.
  LogTy.extend: LogTy#extend.
  LogTy.read: LogTy#read.
  LogTy.lo_ty_qdd: LogTy#lo_ty_qdd().
  LogTy.new_from_loval: LogTy#new_from_loval().
  LogTy.update_from_loval2: LogTy#update_from_loval2().
  NewLog.to_lojax: NewLog#to_lojax().
  new_log: new_log().
  Box._new: Box#_new().
  new_box_p: new_box_p.
  log_effect: log_effect.
  log_append: log_append().
  Log._new: Log#_new().
  new_log_p: new_log_p.
  CustomVJPTraced.traced: CustomVJPTraced#traced.
  CustomVJPTraced.fwd: CustomVJPTraced#fwd.
  CustomVJPTraced.bwd: CustomVJPTraced#bwd.
  ShapedArray: ShapedArray.
  HiType.lo_ty: HiType#lo_ty().
  HiType.to_tangent_aval: HiType#to_tangent_aval().
  HiType.vspace_zero: HiType#vspace_zero().
  HiType.vspace_add: HiType#vspace_add().
  NewBox: NewBox#
  BoxSet: BoxSet#
  BoxGet: BoxGet#
  VJPHiPrimitive.check: VJPHiPrimitive#check().
  VmapOf.fwd: VmapOf#fwd().
  _call_hi_primitive_typecheck: _call_hi_primitive_typecheck().
  _call_hi_primitive_linearized_prettyprint: _call_hi_primitive_linearized_prettyprint().
  _call_hi_primitive_transpose: _call_hi_primitive_transpose().
  _replace_none: _replace_none().
  OptRemat.orig: OptRemat#orig.
  _set_up_nondiff: _set_up_nondiff().
  LogExtend: LogExtend#
  log_extend_p: log_extend_p.
  NewLog: NewLog#
  ReadLog: ReadLog#
  log_read_p: log_read_p.
  VmapOf: VmapOf#
  OptRemat: OptRemat#
  BoxEffect: BoxEffect#
  custom_vjp3.fwd: custom_vjp3#fwd.
  custom_vjp3.bwd: custom_vjp3#bwd.
  Ty: Ty.
  LoType: LoType.
  HiType.leading_axis_spec: HiType#leading_axis_spec().
  hijax_method: hijax_method().
  BoxTy.__hash__: BoxTy#__hash__().
  BoxTy.__eq__: BoxTy#__eq__().
  BoxTy.to_tangent_aval: BoxTy#to_tangent_aval().
  Box.get: Box#get().
  Box.set: Box#set().
  Box.cur_qdd: Box#cur_qdd().
  Box.ty: Box#ty().
  BoxSet.to_lojax: BoxSet#to_lojax().
  BoxGet.abstract_eval: BoxGet#abstract_eval().
  BoxGet.to_lojax: BoxGet#to_lojax().
  VJPHiPrimitive.__repr__: VJPHiPrimitive#__repr__().
  CustomVJPTraced.check: CustomVJPTraced#check().
  custom_vjp3.def_vmap: custom_vjp3#def_vmap().
  custom_vjp3.def_transpose: custom_vjp3#def_transpose().
  OptRemat.expand: OptRemat#expand().
  HiPspec.to_lo: HiPspec#to_lo().
  HiPspec.to_tangent_spec: HiPspec#to_tangent_spec().
  HiPspec.to_ct_spec: HiPspec#to_ct_spec().
  Log.append: Log#append.
  Log.extend: Log#extend.
  Log.read: Log#read.
  LogTy.__hash__: LogTy#__hash__().
  LogTy.__eq__: LogTy#__eq__().
  LogTy.to_tangent_aval: LogTy#to_tangent_aval().
  LogExtend.abstract_eval: LogExtend#abstract_eval().
  LogExtend.to_lojax: LogExtend#to_lojax().
  unsafe_map: unsafe_map.
  unsafe_zip: unsafe_zip.
  PyTreeOfAvals: PyTreeOfAvals.
  Box._val: Box#_val.
  VJPHiPrimitive.effects: VJPHiPrimitive#effects.
  VJPHiPrimitive.in_avals_flat: VJPHiPrimitive#in_avals_flat.
  custom_vjp3: custom_vjp3#
  custom_vjp3.symz: custom_vjp3#symz.
  custom_vjp3.opt_remat: custom_vjp3#opt_remat.
  Static.val: Static#val.
  Log._dct: Log#_dct.
  PyTreeDef: PyTreeDef.
  AxisName: AxisName.
  _BoxMeta: _BoxMeta#
  VJPHiPrimitive.out_avals_flat: VJPHiPrimitive#out_avals_flat.
  CustomVJPTraced.opt_remat: CustomVJPTraced#opt_remat.
  _vjp_primal_fwd_tree_mismatch_err: _vjp_primal_fwd_tree_mismatch_err().
  OptRemat.traced_fwd: OptRemat#traced_fwd.
  _LogMeta: _LogMeta#
  HiPrimitive.is_effectful: HiPrimitive#is_effectful().
  HiType.is_high: HiType#is_high.
  HiType.has_qdd: HiType#has_qdd.
  HiType.__hash__: HiType#__hash__().
  HiType.__eq__: HiType#__eq__().
  MutableHiType.is_high: MutableHiType#is_high.
  MutableHiType.has_qdd: MutableHiType#has_qdd.
  MutableHiType.is_writer: MutableHiType#is_writer.
  MutableHiType.__hash__: MutableHiType#__hash__().
  MutableHiType.__eq__: MutableHiType#__eq__().
  MutableHiType.lo_ty: MutableHiType#lo_ty().
  BoxTy.has_qdd: BoxTy#has_qdd.
  BoxTy.str_short: BoxTy#str_short().
  BoxTy.lo_ty_qdd: BoxTy#lo_ty_qdd().
  NewBox.is_high: NewBox#is_high().
  NewBox.jvp: NewBox#jvp().
  NewBox.transpose: NewBox#transpose().
  BoxSet.multiple_results: BoxSet#multiple_results.
  BoxSet.is_high: BoxSet#is_high().
  BoxSet.transpose: BoxSet#transpose().
  BoxGet.multiple_results: BoxGet#multiple_results.
  BoxGet.transpose: BoxGet#transpose().
  VJPHiPrimitive.transpose: VJPHiPrimitive#transpose().
  custom_vjp3.__init__: custom_vjp3#__init__().
  Log.cur_qdd: Log#cur_qdd().
  LogTy.has_qdd: LogTy#has_qdd.
  LogTy.is_writer: LogTy#is_writer.
  LogTy.str_short: LogTy#str_short().
  LogTy.read_loval_in: LogTy#read_loval_in().
  LogTy.read_loval_out: LogTy#read_loval_out().
  LogExtend.multiple_results: LogExtend#multiple_results.
  LogExtend.is_effectful: LogExtend#is_effectful.
  NewLog.is_high: NewLog#is_high().
  ReadLog.multiple_results: ReadLog#multiple_results.
  ReadLog.is_high: ReadLog#is_high().
  ReadLog.abstract_eval: ReadLog#abstract_eval().
  ReadLog.to_lojax: ReadLog#to_lojax().
---
# Module: [`jax/_src/hijax.py`](../../../../../../raw/code/jax/jax/_src/hijax.py)

## Classes
### `Box`
- def: [`jax/_src/hijax.py:277`](../../../../../../raw/code/jax/jax/_src/hijax.py#L277)
- members:
  - `cur_qdd(self)` — [`L300`](../../../../../../raw/code/jax/jax/_src/hijax.py#L300)
  - `get(self)` — [`L294`](../../../../../../raw/code/jax/jax/_src/hijax.py#L294)
  - `set(self, val)` — [`L297`](../../../../../../raw/code/jax/jax/_src/hijax.py#L297)
  - `ty(self)` — [`L304`](../../../../../../raw/code/jax/jax/_src/hijax.py#L304)
  - `type_state(self)` — [`L307`](../../../../../../raw/code/jax/jax/_src/hijax.py#L307)
- protocol/private: `__new__`[`L282`](../../../../../../raw/code/jax/jax/_src/hijax.py#L282), `_new`[`L289`](../../../../../../raw/code/jax/jax/_src/hijax.py#L289), `_val`[`L278`](../../../../../../raw/code/jax/jax/_src/hijax.py#L278)
- uses (calls/refs, reference-scoped): [`bind`](core.md#Primitive.bind), [`typeof`](core.md#typeof), [`tree_flatten`](tree_util.md#tree_flatten), [`box_get`](hijax.md#box_get), [`BoxTypeState`](hijax.md#BoxTypeState), [`box_set`](hijax.md#box_set), [`BoxTy`](hijax.md#BoxTy), [`new_box_p`](hijax.md#new_box_p), [`_BoxMeta`](hijax.md#_BoxMeta)
- used by: [`new_from_loval`](hijax.md#BoxTy.new_from_loval), [`to_lojax`](hijax.md#NewBox.to_lojax)

### `BoxEffect`  ·  implements/extends Effect
- def: [`jax/_src/hijax.py:314`](../../../../../../raw/code/jax/jax/_src/hijax.py#L314)
- signature: `class BoxEffect(effects.Effect):`
- used by: [`box_effect`](hijax.md#box_effect)

### `BoxGet`  ·  implements/extends HiPrimitive
- def: [`jax/_src/hijax.py:365`](../../../../../../raw/code/jax/jax/_src/hijax.py#L365)
- signature: `class BoxGet(HiPrimitive):`
- members:
  - `abstract_eval(self, box_ty, *, avals)` — [`L368`](../../../../../../raw/code/jax/jax/_src/hijax.py#L368)
  - `jvp(_, primals, tangents, *, avals)` — [`L374`](../../../../../../raw/code/jax/jax/_src/hijax.py#L374)
  - `to_lojax(_, box, *, avals)` — [`L371`](../../../../../../raw/code/jax/jax/_src/hijax.py#L371)
  - `transpose(_, *args)` — [`L381`](../../../../../../raw/code/jax/jax/_src/hijax.py#L381)
  - `multiple_results` — [`L366`](../../../../../../raw/code/jax/jax/_src/hijax.py#L366)
- uses (calls/refs, reference-scoped): [`bind`](core.md#Primitive.bind), [`tree_leaves`](tree_util.md#tree_leaves), [`HiPrimitive`](hijax.md#HiPrimitive), [`box_effect`](hijax.md#box_effect), [`box_get_p`](hijax.md#box_get_p)
- used by: [`HiPrimitive`](hijax.md#HiPrimitive), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`jvp`](hijax.md#HiPrimitive.jvp), [`transpose`](hijax.md#HiPrimitive.transpose), [`box_get_p`](hijax.md#box_get_p)

### `BoxSet`  ·  implements/extends HiPrimitive
- def: [`jax/_src/hijax.py:337`](../../../../../../raw/code/jax/jax/_src/hijax.py#L337)
- signature: `class BoxSet(HiPrimitive):`
- members:
  - `abstract_eval(self, box_ty, *leaf_avals, treedef)` — [`L342`](../../../../../../raw/code/jax/jax/_src/hijax.py#L342)
  - `is_high(self, *leaf_avals, treedef)` — [`L340`](../../../../../../raw/code/jax/jax/_src/hijax.py#L340)
  - `jvp(_, primals, tangents, *, treedef)` — [`L350`](../../../../../../raw/code/jax/jax/_src/hijax.py#L350)
  - `to_lojax(_, box, *leaves, treedef)` — [`L346`](../../../../../../raw/code/jax/jax/_src/hijax.py#L346)
  - `transpose(_, *args, treedef)` — [`L360`](../../../../../../raw/code/jax/jax/_src/hijax.py#L360)
  - `multiple_results` — [`L338`](../../../../../../raw/code/jax/jax/_src/hijax.py#L338)
- uses (calls/refs, reference-scoped): [`bind`](core.md#Primitive.bind), [`tree_unflatten`](tree_util.md#tree_unflatten), [`Zero`](ad_util.md#Zero), [`HiPrimitive`](hijax.md#HiPrimitive), [`BoxTypeState`](hijax.md#BoxTypeState), [`box_effect`](hijax.md#box_effect), [`box_set_p`](hijax.md#box_set_p)
- used by: [`HiPrimitive`](hijax.md#HiPrimitive), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`is_high`](hijax.md#HiPrimitive.is_high), [`jvp`](hijax.md#HiPrimitive.jvp), [`transpose`](hijax.md#HiPrimitive.transpose), [`box_set_p`](hijax.md#box_set_p)

### `BoxTy`  ·  implements/extends MutableHiType
- def: [`jax/_src/hijax.py:229`](../../../../../../raw/code/jax/jax/_src/hijax.py#L229)
- signature: `class BoxTy(MutableHiType):`
- members:
  - `lo_ty_qdd(self, box_state)` — [`L244`](../../../../../../raw/code/jax/jax/_src/hijax.py#L244)
  - `new_from_loval(self, box_state: BoxTypeState, *lo_vals)` — [`L247`](../../../../../../raw/code/jax/jax/_src/hijax.py#L247)
  - `read_loval(self, box_state: BoxTypeState, box)` — [`L254`](../../../../../../raw/code/jax/jax/_src/hijax.py#L254)
  - `str_short(self, short_dtypes=False, **_)` — [`L240`](../../../../../../raw/code/jax/jax/_src/hijax.py#L240)
  - `to_tangent_aval(self)` — [`L267`](../../../../../../raw/code/jax/jax/_src/hijax.py#L267)
  - `update_from_loval(self, box_state: BoxTypeState, box, *lo_vals)` — [`L260`](../../../../../../raw/code/jax/jax/_src/hijax.py#L260)
  - `get` — [`L233`](../../../../../../raw/code/jax/jax/_src/hijax.py#L233)
  - `has_qdd` — [`L230`](../../../../../../raw/code/jax/jax/_src/hijax.py#L230)
  - `set` — [`L234`](../../../../../../raw/code/jax/jax/_src/hijax.py#L234)
- protocol/private: `__eq__`[`L238`](../../../../../../raw/code/jax/jax/_src/hijax.py#L238), `__hash__`[`L237`](../../../../../../raw/code/jax/jax/_src/hijax.py#L237)
- uses (calls/refs, reference-scoped): [`tree_unflatten`](tree_util.md#tree_unflatten), [`tree_flatten`](tree_util.md#tree_flatten), [`aval_method`](core.md#aval_method), [`lo_ty`](core.md#AbstractValue.lo_ty), [`box_get`](hijax.md#box_get), [`BoxTypeState`](hijax.md#BoxTypeState), [`box_set`](hijax.md#box_set), [`leaf_avals`](hijax.md#BoxTypeState.leaf_avals), [`treedef`](hijax.md#BoxTypeState.treedef), [`MutableHiType`](hijax.md#MutableHiType), [`Box`](hijax.md#Box), [`_new`](hijax.md#Box._new)
- used by: [`str_short`](core.md#AbstractValue.str_short), [`abstract_eval`](hijax.md#NewBox.abstract_eval), [`new_from_loval`](hijax.md#MutableHiType.new_from_loval), [`read_loval`](hijax.md#MutableHiType.read_loval), [`update_from_loval`](hijax.md#MutableHiType.update_from_loval), [`lo_ty_qdd`](hijax.md#MutableHiType.lo_ty_qdd), [`to_tangent_aval`](hijax.md#MutableHiType.to_tangent_aval), [`__instancecheck__`](hijax.md#_BoxMeta.__instancecheck__), [`MutableHiType`](hijax.md#MutableHiType), [`ty`](hijax.md#Box.ty)

### `BoxTypeState`  ·  implements/extends QuasiDynamicData
- def: [`jax/_src/hijax.py:217`](../../../../../../raw/code/jax/jax/_src/hijax.py#L217)
- signature: `class BoxTypeState(QDD):`
- members:
  - `normalize(self)` — [`L225`](../../../../../../raw/code/jax/jax/_src/hijax.py#L225)
  - `to_tangent_qdd(self)` — [`L221`](../../../../../../raw/code/jax/jax/_src/hijax.py#L221)
  - `leaf_avals` — [`L218`](../../../../../../raw/code/jax/jax/_src/hijax.py#L218)
  - `treedef` — [`L219`](../../../../../../raw/code/jax/jax/_src/hijax.py#L219)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`to_tangent_aval`](core.md#AbstractValue.to_tangent_aval), [`QDD`](hijax.md#QDD), [`normalize`](core.md#AbstractValue.normalize), [`PyTreeDef`](hijax.md#PyTreeDef)
- used by: [`new_from_loval`](hijax.md#BoxTy.new_from_loval), [`update_from_loval`](hijax.md#BoxTy.update_from_loval), [`abstract_eval`](hijax.md#NewBox.abstract_eval), [`read_loval`](hijax.md#BoxTy.read_loval), [`type_state`](hijax.md#Box.type_state), [`QuasiDynamicData`](core.md#QuasiDynamicData), [`abstract_eval`](hijax.md#BoxSet.abstract_eval)

### `CustomVJPTraced`  ·  implements/extends VJPHiPrimitive
- def: [`jax/_src/hijax.py:711`](../../../../../../raw/code/jax/jax/_src/hijax.py#L711)
- signature: `class CustomVJPTraced(VJPHiPrimitive):`
- members:
  - `batch_dim_rule(self, axis_data, in_dims)` — [`L794`](../../../../../../raw/code/jax/jax/_src/hijax.py#L794)
  - `check(self, *_)` — [`L799`](../../../../../../raw/code/jax/jax/_src/hijax.py#L799)
  - `expand(self, *args)` — [`L726`](../../../../../../raw/code/jax/jax/_src/hijax.py#L726)
  - `jvp(self, primals, tangents)` — [`L777`](../../../../../../raw/code/jax/jax/_src/hijax.py#L777)
  - `remat(self, policy, *args)` — [`L805`](../../../../../../raw/code/jax/jax/_src/hijax.py#L805)
  - `vjp_bwd_retval(self, res, out_ct)` — [`L752`](../../../../../../raw/code/jax/jax/_src/hijax.py#L752)
  - `vjp_fwd(self, in_nzs, *args)` — [`L730`](../../../../../../raw/code/jax/jax/_src/hijax.py#L730)
  - `bwd` — [`L714`](../../../../../../raw/code/jax/jax/_src/hijax.py#L714)
  - `fwd` — [`L713`](../../../../../../raw/code/jax/jax/_src/hijax.py#L713)
  - `opt_remat` — [`L717`](../../../../../../raw/code/jax/jax/_src/hijax.py#L717)
  - `static_argnums` — [`L716`](../../../../../../raw/code/jax/jax/_src/hijax.py#L716)
  - `symbolic_zeros` — [`L715`](../../../../../../raw/code/jax/jax/_src/hijax.py#L715)
  - `traced` — [`L712`](../../../../../../raw/code/jax/jax/_src/hijax.py#L712)
- protocol/private: `__init__`[`L719`](../../../../../../raw/code/jax/jax/_src/hijax.py#L719)
- uses (calls/refs, reference-scoped): [`jit`](api.md#jit), [`tree_unflatten`](tree_util.md#tree_unflatten), [`Zero`](ad_util.md#Zero), [`tree_map`](tree_util.md#tree_map), [`partition_list`](util.md#partition_list), [`tree_leaves`](tree_util.md#tree_leaves), [`unzip2`](util.md#unzip2), [`remat_transform`](interpreters/remat.md#remat_transform), [`merge_lists`](util.md#merge_lists), [`_vjp_bwd_aval_mismatch_err`](hijax.md#_vjp_bwd_aval_mismatch_err), [`tracing_registry`](tree_util.md#tracing_registry), [`tree_leaves_checked`](tree_util.md#tree_leaves_checked), [`broadcast_prefix`](tree_util.md#broadcast_prefix), [`mutable_array_checks`](config.md#mutable_array_checks), [`out_tree`](hijax.md#VJPHiPrimitive.out_tree), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`_check_for_returned_refs`](custom_derivatives.md#_check_for_returned_refs), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`instantiate`](ad_util.md#instantiate), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`fake_linear_op`](hijax.md#fake_linear_op), [`params`](hijax.md#VJPHiPrimitive.params), [`in_tree`](hijax.md#VJPHiPrimitive.in_tree), [`replace_rule_output_symbolic_zeros`](ad_util.md#replace_rule_output_symbolic_zeros), [`Static`](hijax.md#Static), [`_vjp_fwd_aval_mismatch_err`](hijax.md#_vjp_fwd_aval_mismatch_err), [`replace_internal_symbolic_zeros`](ad_util.md#replace_internal_symbolic_zeros), [`trace`](pjit.md#JitWrapped.trace), [`tree_map_with_path`](tree_util.md#tree_map_with_path), [`CustomVJPPrimal`](custom_derivatives.md#CustomVJPPrimal), [`value`](custom_derivatives.md#CustomVJPPrimal.value), [`OptRemat`](hijax.md#OptRemat), [`_replace_none`](hijax.md#_replace_none), [`in_avals_flat`](hijax.md#VJPHiPrimitive.in_avals_flat), [`val`](hijax.md#Static.val), [`_vjp_primal_fwd_tree_mismatch_err`](hijax.md#_vjp_primal_fwd_tree_mismatch_err), [`out_avals_flat`](hijax.md#VJPHiPrimitive.out_avals_flat), [`perturbed`](custom_derivatives.md#CustomVJPPrimal.perturbed)
- used by: [`__call__`](hijax.md#custom_vjp3.__call__), [`expand`](hijax.md#VJPHiPrimitive.expand), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`jvp`](hijax.md#VJPHiPrimitive.jvp), [`vjp_fwd`](hijax.md#VJPHiPrimitive.vjp_fwd), [`vjp_bwd_retval`](hijax.md#VJPHiPrimitive.vjp_bwd_retval), [`remat`](hijax.md#VJPHiPrimitive.remat), [`batch_dim_rule`](hijax.md#VJPHiPrimitive.batch_dim_rule), [`check`](hijax.md#VJPHiPrimitive.check), [`orig`](hijax.md#OptRemat.orig)

### `HiPrimitive`  ·  implements/extends Primitive
- def: [`jax/_src/hijax.py:67`](../../../../../../raw/code/jax/jax/_src/hijax.py#L67)
- signature: `class HiPrimitive(core.Primitive):`
- members:
  - `abstract_eval(self, *arg_avals, **params)` — [`L80`](../../../../../../raw/code/jax/jax/_src/hijax.py#L80)
  - `is_effectful(self, params)` — [`L76`](../../../../../../raw/code/jax/jax/_src/hijax.py#L76)
  - `is_high(self, *avals, **params)` — [`L73`](../../../../../../raw/code/jax/jax/_src/hijax.py#L73)
  - `jvp(self, primals, tangents, **params)` — [`L88`](../../../../../../raw/code/jax/jax/_src/hijax.py#L88)
  - `to_lojax(self, *lotypes_wrapped_in_hitypes, **params)` — [`L84`](../../../../../../raw/code/jax/jax/_src/hijax.py#L84)
  - `transpose(self, *args, **params)` — [`L91`](../../../../../../raw/code/jax/jax/_src/hijax.py#L91)
- protocol/private: `__init__`[`L68`](../../../../../../raw/code/jax/jax/_src/hijax.py#L68)
- uses (calls/refs, reference-scoped): [`Primitive`](core.md#Primitive), [`name`](core.md#Primitive.name), [`abstract_eval`](hijax.md#NewBox.abstract_eval), [`abstract_eval`](hijax.md#NewLog.abstract_eval), [`jvp`](hijax.md#BoxSet.jvp), [`abstract_eval`](hijax.md#BoxSet.abstract_eval), [`jvp`](hijax.md#BoxGet.jvp), [`to_lojax`](hijax.md#NewBox.to_lojax), [`to_lojax`](hijax.md#NewLog.to_lojax), [`BoxGet`](hijax.md#BoxGet), [`BoxSet`](hijax.md#BoxSet), [`LogExtend`](hijax.md#LogExtend), [`NewBox`](hijax.md#NewBox), [`NewLog`](hijax.md#NewLog), [`ReadLog`](hijax.md#ReadLog), [`abstract_eval`](hijax.md#BoxGet.abstract_eval), [`abstract_eval`](hijax.md#LogExtend.abstract_eval), [`to_lojax`](hijax.md#BoxGet.to_lojax), [`to_lojax`](hijax.md#BoxSet.to_lojax), [`to_lojax`](hijax.md#LogExtend.to_lojax), [`abstract_eval`](hijax.md#ReadLog.abstract_eval), [`is_high`](hijax.md#BoxSet.is_high), [`is_high`](hijax.md#NewBox.is_high), [`is_high`](hijax.md#NewLog.is_high), [`is_high`](hijax.md#ReadLog.is_high), [`jvp`](hijax.md#NewBox.jvp), [`to_lojax`](hijax.md#ReadLog.to_lojax), [`transpose`](hijax.md#BoxGet.transpose), [`transpose`](hijax.md#BoxSet.transpose), [`transpose`](hijax.md#NewBox.transpose)
- used by: [`Primitive`](core.md#Primitive), [`is_high`](core.md#Primitive.is_high), [`pallas_call_p`](pallas/pallas_call.md#pallas_call_p), [`to_lojax`](core.md#Primitive.to_lojax), [`mpmd_map_p`](pallas/mpmd.md#mpmd_map_p), [`abstract_eval`](core.md#Primitive.abstract_eval), [`_pallas_call_typecheck_rule`](pallas/pallas_call.md#_pallas_call_typecheck_rule), [`BoxGet`](hijax.md#BoxGet), [`BoxSet`](hijax.md#BoxSet), [`LogExtend`](hijax.md#LogExtend), [`NewBox`](hijax.md#NewBox), [`NewLog`](hijax.md#NewLog), [`ReadLog`](hijax.md#ReadLog)

### `HiPspec`
- def: [`jax/_src/hijax.py:944`](../../../../../../raw/code/jax/jax/_src/hijax.py#L944)
- members:
  - `to_ct_spec(self)` — [`L947`](../../../../../../raw/code/jax/jax/_src/hijax.py#L947)
  - `to_lo(self)` — [`L945`](../../../../../../raw/code/jax/jax/_src/hijax.py#L945)
  - `to_tangent_spec(self)` — [`L946`](../../../../../../raw/code/jax/jax/_src/hijax.py#L946)
- uses (calls/refs, reference-scoped): [`PartitionSpec`](partition_spec.md#PartitionSpec)
- used by: [`_check_specs`](shard_map.md#_check_specs), [`_check_unreduced`](shard_map.md#_check_unreduced), [`check_spec`](shard_map.md#_check_specs.check_spec), [`shard`](hijax.md#HiType.shard), [`unshard`](hijax.md#HiType.unshard), [`nospec`](hijax.md#HiType.nospec)

### `HiType`  ·  implements/extends AbstractValue
- def: [`jax/_src/hijax.py:96`](../../../../../../raw/code/jax/jax/_src/hijax.py#L96)
- signature: `class HiType(core.AbstractValue):`
- members:
  - `dec_rank(self, size: int | None, spec: MappingSpec)` — [`L126`](../../../../../../raw/code/jax/jax/_src/hijax.py#L126)
  - `inc_rank(self, size: int | None, spec: MappingSpec)` — [`L128`](../../../../../../raw/code/jax/jax/_src/hijax.py#L128)
  - `leading_axis_spec(self)` — [`L132`](../../../../../../raw/code/jax/jax/_src/hijax.py#L132)
  - `lo_ty(self)` — [`L105`](../../../../../../raw/code/jax/jax/_src/hijax.py#L105)
  - `lower_val(self, hi_val: HiVal)` — [`L109`](../../../../../../raw/code/jax/jax/_src/hijax.py#L109)
  - `nospec(self, mesh, check_vma: bool, all_names: tuple[AxisName, ...])` — [`L141`](../../../../../../raw/code/jax/jax/_src/hijax.py#L141)
  - `raise_val(self, *lo_vals: LoVal)` — [`L111`](../../../../../../raw/code/jax/jax/_src/hijax.py#L111)
  - `shard(self, mesh, manual_axes: frozenset, check_vma: bool, spec: HiPspec)` — [`L136`](../../../../../../raw/code/jax/jax/_src/hijax.py#L136)
  - `to_ct_aval(self)` — [`L117`](../../../../../../raw/code/jax/jax/_src/hijax.py#L117)
  - `to_tangent_aval(self)` — [`L115`](../../../../../../raw/code/jax/jax/_src/hijax.py#L115)
  - `unshard(self, mesh, check_vma: bool, spec: HiPspec)` — [`L139`](../../../../../../raw/code/jax/jax/_src/hijax.py#L139)
  - `vspace_add(self, x: HiVal, y: HiVal)` — [`L122`](../../../../../../raw/code/jax/jax/_src/hijax.py#L122)
  - `vspace_zero(self)` — [`L120`](../../../../../../raw/code/jax/jax/_src/hijax.py#L120)
  - `has_qdd` — [`L98`](../../../../../../raw/code/jax/jax/_src/hijax.py#L98)
  - `is_high` — [`L97`](../../../../../../raw/code/jax/jax/_src/hijax.py#L97)
- protocol/private: `__eq__`[`L102`](../../../../../../raw/code/jax/jax/_src/hijax.py#L102), `__hash__`[`L101`](../../../../../../raw/code/jax/jax/_src/hijax.py#L101)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`HiVal`](hijax.md#HiVal), [`HiPspec`](hijax.md#HiPspec), [`MappingSpec`](hijax.md#MappingSpec), [`LoVal`](hijax.md#LoVal), [`AxisName`](hijax.md#AxisName)
- used by: [`AbstractValue`](core.md#AbstractValue), [`like`](core.md#ShapeDtypeStruct.like), [`_convert_out_shape_to_aval`](pallas/core.md#_convert_out_shape_to_aval), [`to_ct_aval`](core.md#AbstractValue.to_ct_aval), [`zeros_like_aval`](ad_util.md#zeros_like_aval), [`add_jaxvals`](ad_util.md#add_jaxvals), [`to_tangent_aval`](core.md#AbstractValue.to_tangent_aval), [`unmapped_aval`](core.md#unmapped_aval), [`shard_aval`](core.md#shard_aval), [`unshard_aval`](core.md#unshard_aval), [`mapped_aval`](core.md#mapped_aval), [`empty_like_aval`](ad_util.md#empty_like_aval), [`lo_ty`](core.md#AbstractValue.lo_ty), [`_infer_scan_length`](lax/control_flow/loops.md#_infer_scan_length), [`to_tangent_aval`](hijax.md#MutableHiType.to_tangent_aval), [`dec_rank`](core.md#AbstractValue.dec_rank), [`inc_rank`](core.md#AbstractValue.inc_rank), [`shard`](core.md#AbstractValue.shard), [`to_ct_aval`](hijax.md#MutableHiType.to_ct_aval), [`unshard`](core.md#AbstractValue.unshard), [`vspace_add`](core.md#AbstractValue.vspace_add), [`leading_axis_spec`](core.md#AbstractValue.leading_axis_spec)

### `Log`
- def: [`jax/_src/hijax.py:969`](../../../../../../raw/code/jax/jax/_src/hijax.py#L969)
- signature: `class Log(metaclass=_LogMeta):`
- members:
  - `cur_qdd(self)` — [`L981`](../../../../../../raw/code/jax/jax/_src/hijax.py#L981)
  - `append` — [`L984`](../../../../../../raw/code/jax/jax/_src/hijax.py#L984)
  - `extend` — [`L985`](../../../../../../raw/code/jax/jax/_src/hijax.py#L985)
  - `read` — [`L986`](../../../../../../raw/code/jax/jax/_src/hijax.py#L986)
- protocol/private: `__new__`[`L972`](../../../../../../raw/code/jax/jax/_src/hijax.py#L972), `_dct`[`L970`](../../../../../../raw/code/jax/jax/_src/hijax.py#L970), `_new`[`L976`](../../../../../../raw/code/jax/jax/_src/hijax.py#L976)
- uses (calls/refs, reference-scoped): [`bind`](core.md#Primitive.bind), [`log_extend`](hijax.md#log_extend), [`log_read`](hijax.md#log_read), [`log_append`](hijax.md#log_append), [`new_log_p`](hijax.md#new_log_p), [`_LogMeta`](hijax.md#_LogMeta)
- used by: [`new_from_loval`](hijax.md#LogTy.new_from_loval), [`to_lojax`](hijax.md#NewLog.to_lojax), [`update_from_loval2`](hijax.md#LogTy.update_from_loval2)

### `LogExtend`  ·  implements/extends HiPrimitive
- def: [`jax/_src/hijax.py:1027`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1027)
- signature: `class LogExtend(HiPrimitive):`
- members:
  - `abstract_eval(self, log_ty, *val_tys, treedef)` — [`L1031`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1031)
  - `to_lojax(_, log, *vals, treedef)` — [`L1034`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1034)
  - `is_effectful` — [`L1029`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1029)
  - `multiple_results` — [`L1028`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1028)
- uses (calls/refs, reference-scoped): [`tree_unflatten`](tree_util.md#tree_unflatten), [`HiPrimitive`](hijax.md#HiPrimitive), [`log_effect`](hijax.md#log_effect)
- used by: [`HiPrimitive`](hijax.md#HiPrimitive), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`log_extend_p`](hijax.md#log_extend_p)

### `LogTy`
- def: [`jax/_src/hijax.py:988`](../../../../../../raw/code/jax/jax/_src/hijax.py#L988)
- members:
  - `lo_ty_qdd(self, qdd: QDD,)` — [`L1004`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1004)
  - `new_from_loval(self, qdd)` — [`L1015`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1015)
  - `read_loval_in(self, qdd, log)` — [`L1007`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1007)
  - `read_loval_out(self, qdd, log)` — [`L1011`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1011)
  - `str_short(self, short_dtypes=False, **_)` — [`L998`](../../../../../../raw/code/jax/jax/_src/hijax.py#L998)
  - `to_tangent_aval(self)` — [`L1001`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1001)
  - `update_from_loval2(self, qdd, log: Log, lo_ft)` — [`L1019`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1019)
  - `append` — [`L992`](../../../../../../raw/code/jax/jax/_src/hijax.py#L992)
  - `extend` — [`L993`](../../../../../../raw/code/jax/jax/_src/hijax.py#L993)
  - `has_qdd` — [`L989`](../../../../../../raw/code/jax/jax/_src/hijax.py#L989)
  - `is_writer` — [`L990`](../../../../../../raw/code/jax/jax/_src/hijax.py#L990)
  - `read` — [`L994`](../../../../../../raw/code/jax/jax/_src/hijax.py#L994)
- protocol/private: `__eq__`[`L997`](../../../../../../raw/code/jax/jax/_src/hijax.py#L997), `__hash__`[`L996`](../../../../../../raw/code/jax/jax/_src/hijax.py#L996)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`aval_method`](core.md#aval_method), [`log_extend`](hijax.md#log_extend), [`QDD`](hijax.md#QDD), [`MutableHiType`](hijax.md#MutableHiType), [`log_read`](hijax.md#log_read), [`Log`](hijax.md#Log), [`_new`](hijax.md#Log._new), [`log_append`](hijax.md#log_append), [`_dct`](hijax.md#Log._dct)
- used by: [`str_short`](core.md#AbstractValue.str_short), [`new_from_loval`](hijax.md#MutableHiType.new_from_loval), [`lo_ty_qdd`](hijax.md#MutableHiType.lo_ty_qdd), [`to_tangent_aval`](hijax.md#MutableHiType.to_tangent_aval), [`__instancecheck__`](hijax.md#_LogMeta.__instancecheck__), [`abstract_eval`](hijax.md#NewLog.abstract_eval), [`read_loval_in`](hijax.md#MutableHiType.read_loval_in), [`read_loval_out`](hijax.md#MutableHiType.read_loval_out), [`update_from_loval2`](hijax.md#MutableHiType.update_from_loval2)

### `MappingSpec`
- def: [`jax/_src/hijax.py:943`](../../../../../../raw/code/jax/jax/_src/hijax.py#L943)
- used by: [`vmap`](api.md#vmap), [`dec_rank`](hijax.md#HiType.dec_rank), [`inc_rank`](hijax.md#HiType.inc_rank), [`leading_axis_spec`](hijax.md#HiType.leading_axis_spec)

### `MutableHiType`  ·  implements/extends AbstractValue
- def: [`jax/_src/hijax.py:146`](../../../../../../raw/code/jax/jax/_src/hijax.py#L146)
- signature: `class MutableHiType(core.AbstractValue):`
- members:
  - `lo_ty(self)` — [`L159`](../../../../../../raw/code/jax/jax/_src/hijax.py#L159)
  - `lo_ty_qdd(self, state: QDD,)` — [`L157`](../../../../../../raw/code/jax/jax/_src/hijax.py#L157)
  - `new_from_loval(self, state: QDD,, *vals: LoVal)` — [`L163`](../../../../../../raw/code/jax/jax/_src/hijax.py#L163)
  - `read_loval(self, state: QDD, val: HiVal,)` — [`L165`](../../../../../../raw/code/jax/jax/_src/hijax.py#L165)
  - `read_loval_in(self, state, val,)` — [`L168`](../../../../../../raw/code/jax/jax/_src/hijax.py#L168)
  - `read_loval_out(self, qdd, hi,)` — [`L170`](../../../../../../raw/code/jax/jax/_src/hijax.py#L170)
  - `to_ct_aval(self)` — [`L186`](../../../../../../raw/code/jax/jax/_src/hijax.py#L186)
  - `to_tangent_aval(self)` — [`L181`](../../../../../../raw/code/jax/jax/_src/hijax.py#L181)
  - `update_from_loval(self, state: QDD, val: HiVal,, *lo_vals: LoVal)` — [`L174`](../../../../../../raw/code/jax/jax/_src/hijax.py#L174)
  - `update_from_loval2(self, state, val, lo_vals_ft,)` — [`L177`](../../../../../../raw/code/jax/jax/_src/hijax.py#L177)
  - `has_qdd` — [`L148`](../../../../../../raw/code/jax/jax/_src/hijax.py#L148)
  - `is_high` — [`L147`](../../../../../../raw/code/jax/jax/_src/hijax.py#L147)
  - `is_writer` — [`L149`](../../../../../../raw/code/jax/jax/_src/hijax.py#L149)
  - `type_state` — [`L150`](../../../../../../raw/code/jax/jax/_src/hijax.py#L150)
- protocol/private: `__eq__`[`L154`](../../../../../../raw/code/jax/jax/_src/hijax.py#L154), `__hash__`[`L153`](../../../../../../raw/code/jax/jax/_src/hijax.py#L153)
- uses (calls/refs, reference-scoped): [`AbstractValue`](core.md#AbstractValue), [`aval_method`](core.md#aval_method), [`HiType`](hijax.md#HiType), [`new_from_loval`](hijax.md#BoxTy.new_from_loval), [`cur_qdd`](core.md#cur_qdd), [`update_from_loval`](hijax.md#BoxTy.update_from_loval), [`read_loval`](hijax.md#BoxTy.read_loval), [`HiVal`](hijax.md#HiVal), [`BoxTy`](hijax.md#BoxTy), [`QDD`](hijax.md#QDD), [`LoVal`](hijax.md#LoVal), [`lo_ty_qdd`](hijax.md#LogTy.lo_ty_qdd), [`new_from_loval`](hijax.md#LogTy.new_from_loval), [`update_from_loval2`](hijax.md#LogTy.update_from_loval2), [`to_tangent_aval`](hijax.md#BoxTy.to_tangent_aval), [`to_tangent_aval`](hijax.md#LogTy.to_tangent_aval), [`lo_ty_qdd`](hijax.md#BoxTy.lo_ty_qdd), [`read_loval_in`](hijax.md#LogTy.read_loval_in), [`read_loval_out`](hijax.md#LogTy.read_loval_out)
- used by: [`AbstractValue`](core.md#AbstractValue), [`to_ct_aval`](core.md#AbstractValue.to_ct_aval), [`to_tangent_aval`](core.md#AbstractValue.to_tangent_aval), [`lo_ty`](core.md#AbstractValue.lo_ty), [`BoxTy`](hijax.md#BoxTy), [`LogTy`](hijax.md#LogTy), [`lo_ty_qdd`](core.md#AbstractValue.lo_ty_qdd)

### `NewBox`  ·  implements/extends HiPrimitive
- def: [`jax/_src/hijax.py:319`](../../../../../../raw/code/jax/jax/_src/hijax.py#L319)
- signature: `class NewBox(HiPrimitive):`
- members:
  - `abstract_eval(self, *, treedef)` — [`L322`](../../../../../../raw/code/jax/jax/_src/hijax.py#L322)
  - `is_high(self, *, treedef)` — [`L320`](../../../../../../raw/code/jax/jax/_src/hijax.py#L320)
  - `jvp(_, primals, tangents, *, treedef)` — [`L330`](../../../../../../raw/code/jax/jax/_src/hijax.py#L330)
  - `to_lojax(_, *, treedef)` — [`L327`](../../../../../../raw/code/jax/jax/_src/hijax.py#L327)
  - `transpose(_, *args, treedef)` — [`L333`](../../../../../../raw/code/jax/jax/_src/hijax.py#L333)
- uses (calls/refs, reference-scoped): [`tree_flatten`](tree_util.md#tree_flatten), [`HiPrimitive`](hijax.md#HiPrimitive), [`AvalQDD`](core.md#AvalQDD), [`BoxTypeState`](hijax.md#BoxTypeState), [`BoxTy`](hijax.md#BoxTy), [`Box`](hijax.md#Box), [`box_effect`](hijax.md#box_effect), [`_new`](hijax.md#Box._new)
- used by: [`HiPrimitive`](hijax.md#HiPrimitive), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`is_high`](hijax.md#HiPrimitive.is_high), [`jvp`](hijax.md#HiPrimitive.jvp), [`transpose`](hijax.md#HiPrimitive.transpose), [`new_box_p`](hijax.md#new_box_p)

### `NewLog`  ·  implements/extends HiPrimitive
- def: [`jax/_src/hijax.py:1041`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1041)
- signature: `class NewLog(HiPrimitive):`
- members:
  - `abstract_eval(self)` — [`L1044`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1044)
  - `is_high(self)` — [`L1042`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1042)
  - `to_lojax(_)` — [`L1048`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1048)
- uses (calls/refs, reference-scoped): [`HiPrimitive`](hijax.md#HiPrimitive), [`AvalQDD`](core.md#AvalQDD), [`LogTy`](hijax.md#LogTy), [`Log`](hijax.md#Log), [`_new`](hijax.md#Log._new), [`log_effect`](hijax.md#log_effect)
- used by: [`HiPrimitive`](hijax.md#HiPrimitive), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`is_high`](hijax.md#HiPrimitive.is_high), [`new_log_p`](hijax.md#new_log_p)

### `OptRemat`
- def: [`jax/_src/hijax.py:906`](../../../../../../raw/code/jax/jax/_src/hijax.py#L906)
- members:
  - `dce(self, used_outs)` — [`L919`](../../../../../../raw/code/jax/jax/_src/hijax.py#L919)
  - `expand(self, *primals)` — [`L916`](../../../../../../raw/code/jax/jax/_src/hijax.py#L916)
  - `orig` — [`L907`](../../../../../../raw/code/jax/jax/_src/hijax.py#L907)
  - `traced_fwd` — [`L908`](../../../../../../raw/code/jax/jax/_src/hijax.py#L908)
- protocol/private: `__init__`[`L910`](../../../../../../raw/code/jax/jax/_src/hijax.py#L910)
- uses (calls/refs, reference-scoped): [`tree_leaves`](tree_util.md#tree_leaves), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`params`](hijax.md#VJPHiPrimitive.params), [`CustomVJPTraced`](hijax.md#CustomVJPTraced)
- used by: [`jvp`](hijax.md#CustomVJPTraced.jvp), [`expand`](hijax.md#VJPHiPrimitive.expand), [`dce`](hijax.md#VJPHiPrimitive.dce)

### `ReadLog`  ·  implements/extends HiPrimitive
- def: [`jax/_src/hijax.py:1056`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1056)
- signature: `class ReadLog(HiPrimitive):`
- members:
  - `abstract_eval(self, log_qdd)` — [`L1061`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1061)
  - `is_high(self, _)` — [`L1059`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1059)
  - `to_lojax(_, log)` — [`L1064`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1064)
  - `multiple_results` — [`L1057`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1057)
- uses (calls/refs, reference-scoped): [`HiPrimitive`](hijax.md#HiPrimitive)
- used by: [`HiPrimitive`](hijax.md#HiPrimitive), [`to_lojax`](hijax.md#HiPrimitive.to_lojax), [`abstract_eval`](hijax.md#HiPrimitive.abstract_eval), [`is_high`](hijax.md#HiPrimitive.is_high), [`log_read_p`](hijax.md#log_read_p)

### `Static`
- def: [`jax/_src/hijax.py:940`](../../../../../../raw/code/jax/jax/_src/hijax.py#L940)
- members:
  - `val` — [`L941`](../../../../../../raw/code/jax/jax/_src/hijax.py#L941)
- uses (calls/refs, reference-scoped): [`register_static`](tree_util.md#register_static)
- used by: [`vjp_fwd`](hijax.md#CustomVJPTraced.vjp_fwd), [`vjp_bwd_retval`](hijax.md#CustomVJPTraced.vjp_bwd_retval), [`__call__`](hijax.md#custom_vjp3.__call__), [`vjp_fwd`](hijax.md#VmapOf.vjp_fwd), [`expand`](hijax.md#CustomVJPTraced.expand)

### `VJPHiPrimitive`
- def: [`jax/_src/hijax.py:388`](../../../../../../raw/code/jax/jax/_src/hijax.py#L388)
- signature: `class VJPHiPrimitive:`
- members:
  - `batch(self, axis_data, args, dims)` — [`L448`](../../../../../../raw/code/jax/jax/_src/hijax.py#L448)
  - `batch_dim_rule(self, axis_data, dims,)` — [`L452`](../../../../../../raw/code/jax/jax/_src/hijax.py#L452)
  - `check(self, *arg_tys)` — [`L473`](../../../../../../raw/code/jax/jax/_src/hijax.py#L473)
  - `dce(self, used_outs)` — [`L457`](../../../../../../raw/code/jax/jax/_src/hijax.py#L457)
  - `expand(self, *args)` — [`L411`](../../../../../../raw/code/jax/jax/_src/hijax.py#L411)
  - `jvp(self, primals, tangents)` — [`L430`](../../../../../../raw/code/jax/jax/_src/hijax.py#L430)
  - `lin(self, nzs_in, *primals)` — [`L434`](../../../../../../raw/code/jax/jax/_src/hijax.py#L434)
  - `linearized(self, residuals, *tangents)` — [`L438`](../../../../../../raw/code/jax/jax/_src/hijax.py#L438)
  - `remat(self, _policy, *args)` — [`L465`](../../../../../../raw/code/jax/jax/_src/hijax.py#L465)
  - `staging(self, trace, source_info, *args)` — [`L476`](../../../../../../raw/code/jax/jax/_src/hijax.py#L476)
  - `transpose(self, out_ct, *maybe_accums)` — [`L443`](../../../../../../raw/code/jax/jax/_src/hijax.py#L443)
  - `vjp_bwd(self, res, outgrad,, *arg_accums)` — [`L419`](../../../../../../raw/code/jax/jax/_src/hijax.py#L419)
  - `vjp_bwd_retval(self, res, outgrad,)` — [`L424`](../../../../../../raw/code/jax/jax/_src/hijax.py#L424)
  - `vjp_fwd(self, nzs_in,, *args)` — [`L415`](../../../../../../raw/code/jax/jax/_src/hijax.py#L415)
  - `effects` — [`L392`](../../../../../../raw/code/jax/jax/_src/hijax.py#L392)
  - `in_avals` — [`L389`](../../../../../../raw/code/jax/jax/_src/hijax.py#L389)
  - `in_avals_flat` — [`L405`](../../../../../../raw/code/jax/jax/_src/hijax.py#L405)
  - `in_tree` — [`L405`](../../../../../../raw/code/jax/jax/_src/hijax.py#L405)
  - `out_aval` — [`L390`](../../../../../../raw/code/jax/jax/_src/hijax.py#L390)
  - `out_avals_flat` — [`L406`](../../../../../../raw/code/jax/jax/_src/hijax.py#L406)
  - `out_tree` — [`L406`](../../../../../../raw/code/jax/jax/_src/hijax.py#L406)
  - `params` — [`L391`](../../../../../../raw/code/jax/jax/_src/hijax.py#L391)
- protocol/private: `__call__`[`L468`](../../../../../../raw/code/jax/jax/_src/hijax.py#L468), `__eq__`[`L488`](../../../../../../raw/code/jax/jax/_src/hijax.py#L488), `__hash__`[`L485`](../../../../../../raw/code/jax/jax/_src/hijax.py#L485), `__init__`[`L394`](../../../../../../raw/code/jax/jax/_src/hijax.py#L394), `__repr__`[`L482`](../../../../../../raw/code/jax/jax/_src/hijax.py#L482)
- uses (calls/refs, reference-scoped): [`bind`](core.md#Primitive.bind), [`tree_unflatten`](tree_util.md#tree_unflatten), [`tree_map`](tree_util.md#tree_map), [`vjp_fwd`](hijax.md#CustomVJPTraced.vjp_fwd), [`vjp_bwd_retval`](hijax.md#CustomVJPTraced.vjp_bwd_retval), [`jvp`](hijax.md#CustomVJPTraced.jvp), [`call_hi_primitive_p`](hijax.md#call_hi_primitive_p), [`tracing_registry`](tree_util.md#tracing_registry), [`vjp_fwd`](ad_checkpoint.md#RematTraced.vjp_fwd), [`tree_leaves_checked`](tree_util.md#tree_leaves_checked), [`jvp`](lax/control_flow/loops.md#Scan3.jvp), [`lin`](ad_checkpoint.md#RematTraced.lin), [`remat`](ad_checkpoint.md#CheckpointName.remat), [`remat`](hijax.md#CustomVJPTraced.remat), [`vjp_bwd_retval`](hijax.md#VmapOf.vjp_bwd_retval), [`expand`](lax/control_flow/loops.md#Scan3.expand), [`vjp_fwd`](hijax.md#VmapOf.vjp_fwd), [`expand`](hijax.md#VmapOf.expand), [`expand`](pallas/einshape.md#Einshape.expand), [`jvp`](ad_checkpoint.md#CustomRemat.jvp), [`jvp`](ad_checkpoint.md#RematTraced.jvp), [`jvp`](hijax.md#VmapOf.jvp), [`batch`](ad_checkpoint.md#RematTraced.batch), [`batch_dim_rule`](hijax.md#CustomVJPTraced.batch_dim_rule), [`batch_dim_rule`](hijax.md#VmapOf.batch_dim_rule), [`remat`](ad_checkpoint.md#CustomRemat.remat), [`jvp`](ad_checkpoint.md#CheckpointName.jvp), [`lin`](ad_checkpoint.md#CheckpointName.lin), [`linearized`](ad_checkpoint.md#RematTraced.linearized), [`vjp_fwd`](ad_checkpoint.md#CheckpointName.vjp_fwd), [`CustomVJPTraced`](hijax.md#CustomVJPTraced), [`dce`](hijax.md#OptRemat.dce), [`expand`](ad_checkpoint.md#CustomRemat.expand), [`expand`](ad_checkpoint.md#RematTraced.expand), [`expand`](hijax.md#CustomVJPTraced.expand), [`Scan3`](lax/control_flow/loops.md#Scan3), [`Einshape`](pallas/einshape.md#Einshape), [`PrimalLeftTangentRight`](ad_checkpoint.md#PrimalLeftTangentRight), [`VmapOf`](hijax.md#VmapOf), [`check`](hijax.md#CustomVJPTraced.check)  (+17 more)
- used by: [`vjp_fwd`](hijax.md#CustomVJPTraced.vjp_fwd), [`vjp_bwd_retval`](hijax.md#CustomVJPTraced.vjp_bwd_retval), [`jvp`](hijax.md#CustomVJPTraced.jvp), [`vjp_fwd`](ad_checkpoint.md#RematTraced.vjp_fwd), [`remat`](ad_checkpoint.md#CheckpointName.remat), [`remat`](hijax.md#CustomVJPTraced.remat), [`__init__`](hijax.md#VmapOf.__init__), [`__init__`](lax/control_flow/loops.md#Scan3.__init__), [`__init__`](pallas/einshape.md#Einshape.__init__), [`expand`](lax/control_flow/loops.md#Scan3.expand), [`__init__`](ad_checkpoint.md#CheckpointName.__init__), [`__init__`](ad_checkpoint.md#CustomRemat.__init__), [`__init__`](ad_checkpoint.md#PrimalLeftTangentRight.__init__), [`__init__`](ad_checkpoint.md#RematTraced.__init__), [`__init__`](hijax.md#CustomVJPTraced.__init__), [`__init__`](hijax.md#OptRemat.__init__), [`batch_dim_rule`](hijax.md#CustomVJPTraced.batch_dim_rule), [`CheckpointName`](ad_checkpoint.md#CheckpointName), [`assert_is_tile_preserving`](pallas/einshape.md#Einshape.assert_is_tile_preserving), [`jvp`](ad_checkpoint.md#CheckpointName.jvp), [`lin`](ad_checkpoint.md#CheckpointName.lin), [`vjp_fwd`](ad_checkpoint.md#CheckpointName.vjp_fwd), [`CustomVJPTraced`](hijax.md#CustomVJPTraced), [`RematTraced`](ad_checkpoint.md#RematTraced), [`Scan3`](lax/control_flow/loops.md#Scan3), [`CustomRemat`](ad_checkpoint.md#CustomRemat), [`Einshape`](pallas/einshape.md#Einshape), [`OptRemat`](hijax.md#OptRemat), [`PrimalLeftTangentRight`](ad_checkpoint.md#PrimalLeftTangentRight), [`VmapOf`](hijax.md#VmapOf)

### `VmapOf`
- def: [`jax/_src/hijax.py:492`](../../../../../../raw/code/jax/jax/_src/hijax.py#L492)
- members:
  - `batch_dim_rule(self, axis_data, in_dims)` — [`L543`](../../../../../../raw/code/jax/jax/_src/hijax.py#L543)
  - `expand(self, *args)` — [`L512`](../../../../../../raw/code/jax/jax/_src/hijax.py#L512)
  - `fwd(*args)` — [`L524`](../../../../../../raw/code/jax/jax/_src/hijax.py#L524)
  - `jvp(self, primals, tangents)` — [`L516`](../../../../../../raw/code/jax/jax/_src/hijax.py#L516)
  - `vjp_bwd_retval(self, res_, g)` — [`L533`](../../../../../../raw/code/jax/jax/_src/hijax.py#L533)
  - `vjp_fwd(self, in_nzs, *args)` — [`L522`](../../../../../../raw/code/jax/jax/_src/hijax.py#L522)
  - `axis_data` — [`L494`](../../../../../../raw/code/jax/jax/_src/hijax.py#L494)
  - `in_dims` — [`L495`](../../../../../../raw/code/jax/jax/_src/hijax.py#L495)
  - `out_dim` — [`L496`](../../../../../../raw/code/jax/jax/_src/hijax.py#L496)
  - `prim` — [`L493`](../../../../../../raw/code/jax/jax/_src/hijax.py#L493)
- protocol/private: `__init__`[`L498`](../../../../../../raw/code/jax/jax/_src/hijax.py#L498), `_vmap_params`[`L508`](../../../../../../raw/code/jax/jax/_src/hijax.py#L508)
- uses (calls/refs, reference-scoped): [`Primitive`](core.md#Primitive), [`tree_map`](tree_util.md#tree_map), [`vmap`](api.md#vmap), [`VJPHiPrimitive`](hijax.md#VJPHiPrimitive), [`in_avals`](hijax.md#VJPHiPrimitive.in_avals), [`unmapped_aval`](core.md#unmapped_aval), [`__init__`](hijax.md#VJPHiPrimitive.__init__), [`out_aval`](hijax.md#VJPHiPrimitive.out_aval), [`params`](hijax.md#VJPHiPrimitive.params), [`Static`](hijax.md#Static), [`map_zero`](hijax.md#map_zero), [`unmap_zero`](hijax.md#unmap_zero)
- used by: [`expand`](hijax.md#VJPHiPrimitive.expand), [`jvp`](hijax.md#VJPHiPrimitive.jvp), [`vjp_fwd`](hijax.md#VJPHiPrimitive.vjp_fwd), [`vjp_bwd_retval`](hijax.md#VJPHiPrimitive.vjp_bwd_retval), [`batch`](hijax.md#VJPHiPrimitive.batch), [`batch_dim_rule`](hijax.md#VJPHiPrimitive.batch_dim_rule)

### `_BoxMeta`  ·  implements/extends type
- def: [`jax/_src/hijax.py:271`](../../../../../../raw/code/jax/jax/_src/hijax.py#L271)
- signature: `class _BoxMeta(type):`
- protocol/private: `__instancecheck__`[`L272`](../../../../../../raw/code/jax/jax/_src/hijax.py#L272)
- uses (calls/refs, reference-scoped): [`typeof`](core.md#typeof), [`Tracer`](core.md#Tracer), [`BoxTy`](hijax.md#BoxTy)
- used by: [`Box`](hijax.md#Box)

### `_LogMeta`  ·  implements/extends type
- def: [`jax/_src/hijax.py:963`](../../../../../../raw/code/jax/jax/_src/hijax.py#L963)
- signature: `class _LogMeta(type):`
- protocol/private: `__instancecheck__`[`L964`](../../../../../../raw/code/jax/jax/_src/hijax.py#L964)
- uses (calls/refs, reference-scoped): [`typeof`](core.md#typeof), [`Tracer`](core.md#Tracer), [`LogTy`](hijax.md#LogTy)
- used by: [`Log`](hijax.md#Log)

### `custom_vjp3`
- def: [`jax/_src/hijax.py:862`](../../../../../../raw/code/jax/jax/_src/hijax.py#L862)
- signature: `class custom_vjp3:`
- members:
  - `def_transpose(self, rule,)` — [`L904`](../../../../../../raw/code/jax/jax/_src/hijax.py#L904)
  - `def_vmap(self, rule,)` — [`L903`](../../../../../../raw/code/jax/jax/_src/hijax.py#L903)
  - `defvjp(self, fwd, bwd, *, symbolic_zeros=False, optimize_remat=False)` — [`L873`](../../../../../../raw/code/jax/jax/_src/hijax.py#L873)
  - `bwd` — [`L864`](../../../../../../raw/code/jax/jax/_src/hijax.py#L864)
  - `f` — [`L869`](../../../../../../raw/code/jax/jax/_src/hijax.py#L869)
  - `fwd` — [`L863`](../../../../../../raw/code/jax/jax/_src/hijax.py#L863)
  - `opt_remat` — [`L866`](../../../../../../raw/code/jax/jax/_src/hijax.py#L866)
  - `static_argnums` — [`L870`](../../../../../../raw/code/jax/jax/_src/hijax.py#L870)
  - `symz` — [`L865`](../../../../../../raw/code/jax/jax/_src/hijax.py#L865)
- protocol/private: `__call__`[`L880`](../../../../../../raw/code/jax/jax/_src/hijax.py#L880), `__init__`[`L868`](../../../../../../raw/code/jax/jax/_src/hijax.py#L868)
- uses (calls/refs, reference-scoped): [`jit`](api.md#jit), [`typeof`](core.md#typeof), [`tree_map`](tree_util.md#tree_map), [`Tracer`](core.md#Tracer), [`aval`](core.md#Tracer.aval), [`str_short`](core.md#AbstractValue.str_short), [`UnexpectedTracerError`](errors.md#UnexpectedTracerError), [`Static`](hijax.md#Static), [`trace`](pjit.md#JitWrapped.trace), [`resolve_kwargs`](api_util.md#resolve_kwargs), [`CustomVJPTraced`](hijax.md#CustomVJPTraced), [`_set_up_nondiff`](hijax.md#_set_up_nondiff)
- used by: [`__new__`](custom_derivatives.md#custom_vjp.__new__)

## Functions
- `_call_hi_primitive_abstract_eval(*_args, _prim)` — [`L566`](../../../../../../raw/code/jax/jax/_src/hijax.py#L566)
- `_call_hi_primitive_batcher(axis_data, args_flat, dims_flat, _prim)` — [`L590`](../../../../../../raw/code/jax/jax/_src/hijax.py#L590)
- `_call_hi_primitive_dce(used_outs_flat, eqn)` — [`L682`](../../../../../../raw/code/jax/jax/_src/hijax.py#L682)
- `_call_hi_primitive_jvp(primals, tangents, *, _prim)` — [`L666`](../../../../../../raw/code/jax/jax/_src/hijax.py#L666)
- `_call_hi_primitive_linearize(is_vjp, nz_in_flat, *args_flat, _prim)` — [`L599`](../../../../../../raw/code/jax/jax/_src/hijax.py#L599)
- `_call_hi_primitive_linearized_abstract_eval(*_args, _prim, residuals_tree, nz_in_flat, nz_out_flat)` — [`L638`](../../../../../../raw/code/jax/jax/_src/hijax.py#L638)
- `_call_hi_primitive_linearized_prettyprint(eqn, context, settings)` — [`L660`](../../../../../../raw/code/jax/jax/_src/hijax.py#L660)
- `_call_hi_primitive_linearized_transpose(cts_flat_, *args, _prim, residuals_tree, nz_in_flat, nz_out_flat)` — [`L642`](../../../../../../raw/code/jax/jax/_src/hijax.py#L642)
- `_call_hi_primitive_remat(policy, *args_flat, _prim)` — [`L700`](../../../../../../raw/code/jax/jax/_src/hijax.py#L700)
- `_call_hi_primitive_staging(trace, source_info, *args_flat, _prim)` — [`L577`](../../../../../../raw/code/jax/jax/_src/hijax.py#L577)
- `_call_hi_primitive_to_lojax(*args_flat, _prim)` — [`L584`](../../../../../../raw/code/jax/jax/_src/hijax.py#L584)
- `_call_hi_primitive_transpose(cts_flat, *primals_flat, _prim)` — [`L675`](../../../../../../raw/code/jax/jax/_src/hijax.py#L675)
- `_call_hi_primitive_typecheck(_ctx_factory, *in_atoms_flat, _prim)` — [`L569`](../../../../../../raw/code/jax/jax/_src/hijax.py#L569)
- `_replace_none(primal_in_aval, maybe_ct)` — [`L856`](../../../../../../raw/code/jax/jax/_src/hijax.py#L856)
- `_set_up_nondiff(f, argnums_, argnames)` — [`L931`](../../../../../../raw/code/jax/jax/_src/hijax.py#L931)
- `_vjp_bwd_aval_mismatch_err(path, primal_aval, ct)` — [`L839`](../../../../../../raw/code/jax/jax/_src/hijax.py#L839)
- `_vjp_fwd_aval_mismatch_err(path, primal_aval, fwd_val)` — [`L834`](../../../../../../raw/code/jax/jax/_src/hijax.py#L834)
- `_vjp_primal_fwd_tree_mismatch_err(self, tree)` — [`L819`](../../../../../../raw/code/jax/jax/_src/hijax.py#L819)
- `box_get(box)` — [`L205`](../../../../../../raw/code/jax/jax/_src/hijax.py#L205)
- `box_set(box, val)` — [`L210`](../../../../../../raw/code/jax/jax/_src/hijax.py#L210)
- `fake_linear_op(prim, nz_in_flat, nz_out_flat, rs, *tangents)` — [`L615`](../../../../../../raw/code/jax/jax/_src/hijax.py#L615)
- `flatten_user_linearized(prim, residuals, *tangents_flat)` — [`L628`](../../../../../../raw/code/jax/jax/_src/hijax.py#L628)
- `hijax_method(f)` — [`L193`](../../../../../../raw/code/jax/jax/_src/hijax.py#L193)
- `log_append(log, key, val)` — [`L957`](../../../../../../raw/code/jax/jax/_src/hijax.py#L957)
- `log_extend(log, dct)` — [`L953`](../../../../../../raw/code/jax/jax/_src/hijax.py#L953)
- `log_read(log)` — [`L960`](../../../../../../raw/code/jax/jax/_src/hijax.py#L960)
- `map_zero(axis_data, d, ct)` — [`L549`](../../../../../../raw/code/jax/jax/_src/hijax.py#L549)
- `new_box()` — [`L201`](../../../../../../raw/code/jax/jax/_src/hijax.py#L201)
- `new_log()` — [`L1052`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1052)
- `register_hitype(val_cls, typeof_fn)` — [`L189`](../../../../../../raw/code/jax/jax/_src/hijax.py#L189)
- `rem(*args_flat)` — [`L703`](../../../../../../raw/code/jax/jax/_src/hijax.py#L703)
- `unmap_zero(axis_data, d, ct)` — [`L554`](../../../../../../raw/code/jax/jax/_src/hijax.py#L554)

## Module values
- `AxisName` — [`L94`](../../../../../../raw/code/jax/jax/_src/hijax.py#L94)
- `HiVal` — [`L55`](../../../../../../raw/code/jax/jax/_src/hijax.py#L55)
- `LoType` — [`L63`](../../../../../../raw/code/jax/jax/_src/hijax.py#L63)
- `LoVal` — [`L54`](../../../../../../raw/code/jax/jax/_src/hijax.py#L54)
- `PyTreeDef` — [`L53`](../../../../../../raw/code/jax/jax/_src/hijax.py#L53)
- `PyTreeOfAvals` — [`L52`](../../../../../../raw/code/jax/jax/_src/hijax.py#L52)
- `QDD` — [`L64`](../../../../../../raw/code/jax/jax/_src/hijax.py#L64)
- `ShapedArray` — [`L65`](../../../../../../raw/code/jax/jax/_src/hijax.py#L65)
- `Ty` — [`L62`](../../../../../../raw/code/jax/jax/_src/hijax.py#L62)
- `box_effect` — [`L315`](../../../../../../raw/code/jax/jax/_src/hijax.py#L315)
- `box_get_p` — [`L383`](../../../../../../raw/code/jax/jax/_src/hijax.py#L383)
- `box_set_p` — [`L362`](../../../../../../raw/code/jax/jax/_src/hijax.py#L362)
- `call_hi_primitive_linearized_p` — [`L634`](../../../../../../raw/code/jax/jax/_src/hijax.py#L634)
- `call_hi_primitive_p` — [`L561`](../../../../../../raw/code/jax/jax/_src/hijax.py#L561)
- `log_effect` — [`L951`](../../../../../../raw/code/jax/jax/_src/hijax.py#L951)
- `log_extend_p` — [`L1039`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1039)
- `log_read_p` — [`L1066`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1066)
- `new_box_p` — [`L335`](../../../../../../raw/code/jax/jax/_src/hijax.py#L335)
- `new_log_p` — [`L1050`](../../../../../../raw/code/jax/jax/_src/hijax.py#L1050)
- `unsafe_map` — [`L49`](../../../../../../raw/code/jax/jax/_src/hijax.py#L49)
- `unsafe_zip` — [`L50`](../../../../../../raw/code/jax/jax/_src/hijax.py#L50)

