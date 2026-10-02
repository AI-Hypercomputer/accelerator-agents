---
title: 'Module: jax/_src/tree_util.py'
type: catalog
provenance: extracted
module: jax/_src/tree_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.tree_util`/
symbols:
  tree_unflatten: tree_unflatten().
  tree_flatten: tree_flatten().
  tree_map: tree_map().
  PyTreeDef: PyTreeDef.
  tree_leaves: tree_leaves().
  tree_structure: tree_structure().
  keystr: keystr().
  register_dataclass: register_dataclass().
  export: export.
  tracing_registry: tracing_registry.
  tree_leaves_checked: tree_leaves_checked().
  broadcast_prefix: broadcast_prefix().
  register_pytree_node: register_pytree_node().
  T: T.
  tree_flatten_with_path: tree_flatten_with_path().
  _prefix_error: _prefix_error().
  register_pytree_with_keys: register_pytree_with_keys().
  register_pytree_node_class: register_pytree_node_class().
  equality_errors_pytreedef: equality_errors_pytreedef().
  default_registry: default_registry.
  KeyPath: KeyPath.
  treedef_tuple: treedef_tuple().
  tree_transpose: tree_transpose().
  _equality_errors: _equality_errors().
  _child_keys: _child_keys().
  _simple_entrystr: _simple_entrystr().
  generate_key_paths: generate_key_paths.
  none_leaf_registry: none_leaf_registry.
  tree_reduce_associative: tree_reduce_associative().
  Partial: Partial#
  prefix_errors: prefix_errors().
  _all_registries: _all_registries.
  treedef_children: treedef_children().
  treedef_is_leaf: treedef_is_leaf().
  _registry._registry: _registry._registry.
  _parallel_reduce: _parallel_reduce().
  tree_leaves_with_path: tree_leaves_with_path().
  tree_reduce: tree_reduce().
  dispatch_registry: dispatch_registry.
  register_static: register_static().
  _RegistryEntry: _RegistryEntry.
  broadcast_flattened_prefix_with_treedef: broadcast_flattened_prefix_with_treedef().
  flatten_one_level: flatten_one_level().
  equality_errors: equality_errors().
  treedef_is_strict_leaf: treedef_is_strict_leaf().
  flatten_one_level_with_keys: flatten_one_level_with_keys().
  Unspecified: Unspecified#
  tree_map_with_path: tree_map_with_path().
  all_leaves: all_leaves().
  tree_broadcast: tree_broadcast().
  register_pytree_with_keys_class: register_pytree_with_keys_class().
  Typ: Typ.
  _AuxData: _AuxData.
  _HashableCallableShim.fun: _HashableCallableShim#fun.
  KeyLeafPair: KeyLeafPair.
  treedef_tuple_tracing_registry: treedef_tuple_tracing_registry().
  is_tree_node: is_tree_node().
  tree_all: tree_all().
  _HashableCallableShim.__eq__: _HashableCallableShim#__eq__().
  broadcast_flattened_prefix_with_treedef._broadcast: broadcast_flattened_prefix_with_treedef()._broadcast().
  Leaf: Leaf.
  KeyLeafPairs: KeyLeafPairs.
  _flatten_defaultdict_with_keys: _flatten_defaultdict_with_keys().
  FlattenedIndexKey.FlattenedIndexKey: FlattenedIndexKey.FlattenedIndexKey.
  PyTree: PyTree.
  KeyEntry: KeyEntry.
  DictKey.DictKey: DictKey.DictKey.
  _HashableCallableShim.__call__: _HashableCallableShim#__call__().
  _HashableCallableShim.__hash__: _HashableCallableShim#__hash__().
  _HashableCallableShim.__repr__: _HashableCallableShim#__repr__().
  Partial.__new__: Partial#__new__().
  H: H.
  _Children: _Children.
  _HashableCallableShim: _HashableCallableShim#
  register_pytree_with_keys.flatten_func_impl: register_pytree_with_keys().flatten_func_impl().
  register_dataclass.unflatten_func: register_dataclass().unflatten_func().
  register_dataclass.flatten_func: register_dataclass().flatten_func().
  SequenceKey.SequenceKey: SequenceKey.SequenceKey.
  GetAttrKey.GetAttrKey: GetAttrKey.GetAttrKey.
  _HashableCallableShim.__init__: _HashableCallableShim#__init__().
  _ensure_inbounds: _ensure_inbounds().
---
# Module: [`jax/_src/tree_util.py`](../../../../../../raw/code/jax/jax/_src/tree_util.py)

## Classes
### `Partial`  ·  implements/extends partial
- def: [`jax/_src/tree_util.py:513`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L513)
- doc: A version of functools.partial that works in pytrees.
- signature: `class Partial(functools.partial):`
- protocol/private: `__new__`[`L562`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L562)
- uses (calls/refs, reference-scoped): [`export`](tree_util.md#export), [`_HashableCallableShim`](tree_util.md#_HashableCallableShim)
- used by: [`remat_transform`](interpreters/remat.md#remat_transform), [`_vjp3_callable`](api.md#_vjp3_callable), [`linear_transpose`](api.md#linear_transpose), [`dce`](ad_checkpoint.md#dce)

### `Unspecified`
- def: [`jax/_src/tree_util.py:437`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L437)
- signature: `class Unspecified:`
- used by: [`tree_reduce_associative`](tree_util.md#tree_reduce_associative), [`_parallel_reduce`](tree_util.md#_parallel_reduce), [`tree_reduce`](tree_util.md#tree_reduce)

### `_HashableCallableShim`
- def: [`jax/_src/tree_util.py:491`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L491)
- doc: Object that delegates __call__, __hash__, and __eq__ to another object.
- signature: `class _HashableCallableShim:`
- members:
  - `fun` — [`L495`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L495)
- protocol/private: `__call__`[`L497`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L497), `__eq__`[`L503`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L503), `__hash__`[`L500`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L500), `__init__`[`L494`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L494), `__repr__`[`L508`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L508)
- used by: [`__new__`](tree_util.md#Partial.__new__)

## Functions
- `_broadcast(broadcast_fn, leaf_start, leaf_end, prefix_treedef, treedef)` — [`L658`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L658)
- `_child_keys(pytree: Any)` — [`L1272`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1272)
- `_ensure_inbounds(allow_invalid: bool, num_args: int, argnums: Sequence[int])` — [`L1373`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1373) — Ensure argnum is within bounds. Also resolves negative argnums.
- `_equality_errors(path, t1, t2, is_leaf)` — [`L771`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L771)
- `_flatten_defaultdict_with_keys(d)` — [`L1180`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1180)
- `_parallel_reduce(sequence: list[T], operation: Callable[[T, T], T], identity: T | Unspecified = Unspecified())` — [`L453`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L453)
- `_prefix_error(key_path: KeyPath, prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L1277`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1277)
- `_simple_entrystr(key: KeyEntry)` — [`L867`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L867)
- `all_leaves(iterable: Iterable[Any], is_leaf: Callable[[Any], bool] | None = None)` — [`L204`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L204) — Tests whether all elements in the given iterable are all leaves.
- `broadcast_flattened_prefix_with_treedef(prefix_leaves: list[Any], prefix_treedef: PyTreeDef, full_treedef: PyTreeDef)` — [`L627`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L627) — Broadcasts tree prefix leaves into the full set of leaves for a given full treedef.
- `broadcast_prefix(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L597`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L597) — Broadcasts tree prefix leaves into the full set of leaves for a given full tree.
- `equality_errors(tree1: Any, tree2: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L739`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L739) — Helper to describe structural differences between two pytrees.
- `equality_errors_pytreedef(tree1: PyTreeDef, tree2: PyTreeDef)` — [`L760`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L760) — Like `equality_errors` but invoked on PyTreeDef.
- `flatten_func(x)` — [`L1163`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1163)
- `flatten_func_impl(tree)` — [`L941`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L941)
- `flatten_one_level(tree: Any)` — [`L689`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L689) — Flatten the given pytree node by one level.
- `flatten_one_level_with_keys(tree: Any)` — [`L720`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L720) — Flatten the given pytree node by one level, with keys.
- `is_tree_node(typ: type)` — [`L235`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L235) — Returns True if the type is a registered PyTree node type.
- `keystr(keys: KeyPath, *, simple: bool = False, separator: str = '')` — [`L835`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L835) — Helper to pretty-print a tuple of keys.
- `prefix_errors(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L732`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L732)
- `register_dataclass(nodetype: Typ, data_fields: Sequence[str] | None = None, meta_fields: Sequence[str] | None = None, drop_fields: Sequence[str] = ())` — [`L1001`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1001) — Extends the set of types that are considered internal nodes in pytrees.
- `register_pytree_node(nodetype: type[T], flatten_func: Callable[[T], tuple[_Children, _AuxData]], unflatten_func: Callable[[_AuxData, _Children], T], flatten_with_keys_func: (Callable[[T], tuple[KeyLeafPairs, _AuxData]] | None) = None)` — [`L257`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L257) — Extends the set of types that are considered internal nodes in pytrees.
- `register_pytree_node_class(cls: Typ)` — [`L337`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L337) — Extends the set of types that are considered internal nodes in pytrees.
- `register_pytree_with_keys(nodetype: type[T], flatten_with_keys: Callable[[T], tuple[Iterable[KeyLeafPair], _AuxData]], unflatten_func: Callable[[_AuxData, Iterable[Any]], T], flatten_func: None | Callable[[T], tuple[Iterable[Any], _AuxData]] = None)` — [`L881`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L881) — Extends the set of types that are considered internal nodes in pytrees.
- `register_pytree_with_keys_class(cls: Typ)` — [`L952`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L952) — Extends the set of types that are considered internal nodes in pytrees.
- `register_static(cls: type[H])` — [`L1192`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1192) — Registers `cls` as a pytree with no leaves.
- `tree_all(tree: Any, *, is_leaf: Callable[[Any], bool] | None = None)` — [`L486`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L486) — Alias of :func:`jax.tree.all`.
- `tree_broadcast(prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L588`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L588) — Alias of :func:`jax.tree.broadcast`.
- `tree_flatten(tree: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L86`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L86) — Alias of :func:`jax.tree.flatten`. — documented in [jax-_src-tree_util](../../../concepts/jax-_src-tree_util.md)
- `tree_flatten_with_path(tree: Any, is_leaf: Callable[..., bool] | None = None, is_leaf_takes_path: bool = False)` — [`L1230`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1230) — Alias of :func:`jax.tree.flatten_with_path`.
- `tree_leaves(tree: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L100`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L100) — Alias of :func:`jax.tree.leaves`.
- `tree_leaves_checked(treedef_expected: PyTreeDef, tree: Any)` — [`L108`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L108)
- `tree_leaves_with_path(tree: Any, is_leaf: Callable[..., bool] | None = None, is_leaf_takes_path: bool = False)` — [`L1242`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1242) — Alias of :func:`jax.tree.leaves_with_path`.
- `tree_map(f: Callable[..., Any], tree: Any, *rest: Any, is_leaf: Callable[[Any], bool] | None = None)` — [`L389`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L389) — Alias of :func:`jax.tree.map`. — documented in [jax-_src-tree_util](../../../concepts/jax-_src-tree_util.md)
- `tree_map_with_path(f: Callable[..., Any], tree: Any, *rest: Any, is_leaf: Callable[..., bool] | None = None, is_leaf_takes_path: bool = False)` — [`L1252`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1252) — Alias of :func:`jax.tree.map_with_path`.
- `tree_reduce(function: Callable[[T, Any], T], tree: Any, initializer: T | Unspecified = Unspecified(), is_leaf: Callable[[Any], bool] | None = None)` — [`L442`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L442) — Alias of :func:`jax.tree.reduce`.
- `tree_reduce_associative(operation: Callable[[T, T], T], tree: Any, *, identity: T | Unspecified = Unspecified(), is_leaf: Callable[[Any], bool] | None = None)` — [`L473`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L473) — Alias of :func:`jax.tree.reduce_associative`.
- `tree_structure(tree: Any, is_leaf: None | Callable[[Any], bool] = None)` — [`L115`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L115) — Alias of :func:`jax.tree.structure`.
- `tree_transpose(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef | None, pytree_to_transpose: Any)` — [`L404`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L404) — Alias of :func:`jax.tree.transpose`.
- `tree_unflatten(treedef: PyTreeDef, leaves: Iterable[Leaf])` — [`L94`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L94) — Alias of :func:`jax.tree.unflatten`. — documented in [jax-_src-tree_util](../../../concepts/jax-_src-tree_util.md)
- `treedef_children(treedef: PyTreeDef)` — [`L152`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L152) — Return a list of treedefs for immediate children
- `treedef_is_leaf(treedef: PyTreeDef)` — [`L177`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L177) — Return True if the treedef represents a leaf.
- `treedef_is_strict_leaf(treedef: PyTreeDef)` — [`L199`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L199)
- `treedef_tuple(treedefs: Iterable[PyTreeDef])` — [`L126`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L126) — Makes a tuple treedef from an iterable of child treedefs.
- `treedef_tuple_tracing_registry(treedefs: Iterable[PyTreeDef])` — [`L122`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L122)
- `unflatten_func(meta, data)` — [`L1157`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1157)

## Module values
- `DictKey` — [`L829`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L829)
- `FlattenedIndexKey` — [`L831`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L831)
- `GetAttrKey` — [`L830`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L830)
- `H` — [`L37`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L37)
- `KeyEntry` — [`L250`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L250)
- `KeyLeafPair` — [`L251`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L251)
- `KeyLeafPairs` — [`L252`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L252)
- `KeyPath` — [`L253`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L253)
- `Leaf` — [`L39`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L39)
- `PyTree` — [`L40`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L40)
- `PyTreeDef` — [`L41`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L41) — documented in [jax-_src-tree_util](../../../concepts/jax-_src-tree_util.md)
- `SequenceKey` — [`L828`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L828)
- `T` — [`L35`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L35)
- `Typ` — [`L36`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L36)
- `_AuxData` — [`L249`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L249)
- `_Children` — [`L248`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L248)
- `_RegistryEntry` — [`L427`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L427)
- `_all_registries` — [`L77`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L77)
- `_registry` — [`L428`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L428)
- `default_registry` — [`L43`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L43)
- `dispatch_registry` — [`L66`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L66)
- `export` — [`L31`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L31)
- `generate_key_paths` — [`L1248`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L1248)
- `none_leaf_registry` — [`L50`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L50)
- `tracing_registry` — [`L72`](../../../../../../raw/code/jax/jax/_src/tree_util.py#L72)

