---
title: 'Module: easydel/utils/traversals.py'
type: catalog
provenance: extracted
module: easydel/utils/traversals.py
status: fresh
symbol_base: scip-python python easydel 0.0.0 `easydel.utils.traversals`/
symbols:
  iter_module_search: iter_module_search().
  flatten_dict: flatten_dict().
  unflatten_dict: unflatten_dict().
  deepcopy_model: deepcopy_model().
  merge_state_and_tree: merge_state_and_tree().
  get_module_from_path: get_module_from_path().
  recreate_meta_values: recreate_meta_values().
  set_module_from_path: set_module_from_path().
  is_flatten: is_flatten().
  string_key_to_int: string_key_to_int().
  differentiate_state: differentiate_state().
  _dict_flatten_dict._flatten: _dict_flatten_dict()._flatten().
  int_key_to_string: int_key_to_string().
  merge_model_and_tree: merge_model_and_tree().
  tree_path_to_string: tree_path_to_string().
  tree_apply: tree_apply().
  flatten_tree: flatten_tree().
  named_tree_map: named_tree_map().
  empty_node: empty_node.
  nnx_init: nnx_init().
  specs_to_name_sharding: specs_to_name_sharding().
  recursive_merge: recursive_merge().
  M: M.
  MetaValueRecreator.get_count: MetaValueRecreator#get_count().
  MetaValueRecreator.get_rng: MetaValueRecreator#get_rng().
  _dict_flatten_dict: _dict_flatten_dict().
  _dict_unflatten_dict: _dict_unflatten_dict().
  validate_state: validate_state().
  refine_graphs: refine_graphs().
  PyTree: PyTree.
  create_graphdef: create_graphdef().
  init_graphstate: init_graphstate().
  T: T.
  ModulePath: ModulePath.
  TreeDict: TreeDict.
  logger: logger.
  MetaValueRecreator._count: MetaValueRecreator#_count.
  MetaValueRecreator._rng: MetaValueRecreator#_rng.
  StateValidationResult: StateValidationResult#
  _dict_flatten_dict._key: _dict_flatten_dict()._key().
  FnDict: FnDict.
  Path: Path.
  MetaValueRecreator: MetaValueRecreator#
  _EmptyNode: _EmptyNode#
  StateValidationResult.is_valid: StateValidationResult#is_valid.
  StateValidationResult.missing_keys: StateValidationResult#missing_keys.
  StateValidationResult.invalid_types: StateValidationResult#invalid_types.
  MetaValueRecreator.__init__: MetaValueRecreator#__init__().
  is_iterable: is_iterable().
  redefine_state: redefine_state().
---
# Module: [`easydel/utils/traversals.py`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py)

## Classes
### `MetaValueRecreator`
- def: [`easydel/utils/traversals.py:43`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L43)
- doc: Helper for recreating nnx meta values (RNG keys/counts) deterministically.
- signature: `class MetaValueRecreator:`
- members:
  - `get_count(self)` — [`L58`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L58) — Return the next counter value as a uint32 array and increment.
  - `get_rng(self)` — [`L64`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L64) — Split the internal PRNG key and return one half.
- protocol/private: `__init__`[`L54`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L54), `_count`[`L55`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L55), `_rng`[`L56`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L56)
- used by: [`recreate_meta_values`](traversals.md#recreate_meta_values)

### `StateValidationResult`
- def: [`easydel/utils/traversals.py:76`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L76)
- doc: Result of validating a state dictionary against a reference.
- signature: `class StateValidationResult:`
- members:
  - `invalid_types` — [`L87`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L87)
  - `is_valid` — [`L85`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L85)
  - `missing_keys` — [`L86`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L86)
- used by: [`differentiate_state`](traversals.md#differentiate_state), [`validate_state`](traversals.md#validate_state)

### `_EmptyNode`
- def: [`easydel/utils/traversals.py:71`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L71)
- signature: `class _EmptyNode:`
- used by: [`empty_node`](traversals.md#empty_node)

## Functions
- `_dict_flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None, fumap=False)` — [`L137`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L137)
- `_dict_unflatten_dict(xs, sep=None)` — [`L170`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L170)
- `_flatten(xs, prefix)` — [`L147`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L147)
- `_key(path)` — [`L142`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L142)
- `create_graphdef(module: nnx.Module, _add_rngs: bool = True, _rng_key: str = "rngs", _seed: int = 0, **kwargs)` — [`L281`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L281) — Creates a graph definition from an nnx module.
- `deepcopy_model(model)` — [`L724`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L724) — Creates a deep copy of a JAX model.
- `differentiate_state(state: dict[str, tp.Any], init_state: dict[str, tp.Any], validate: bool = True)` — [`L380`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L380) — Enhanced state differentiation with validation and error handling.
- `flatten_dict(xs: dict | Mapping, keep_empty_nodes: bool = False, is_leaf: tp.Callable[[tuple, tp.Any], bool] | None = None, sep: str | None = None, fumap: bool = False)` — [`L188`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L188) — Enhanced dictionary flattening with better type handling and validation.
- `flatten_tree(xs: PyTree, is_leaf: tp.Callable[[tp.Any], bool] | None = None, sep: str | None = None)` — [`L670`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L670) — Flatten a JAX tree and convert paths to strings.
- `get_module_from_path(model: nn.Module, path: ModulePath)` — [`L805`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L805) — Retrieves a module from a model given its path.
- `init_graphstate(module: nnx.Module, _add_rngs: bool = True, _rng_key: str = "rngs", _seed: int = 0, _lazy: bool = True, **kwargs)` — [`L318`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L318) — Initializes the graph state of an nnx module.
- `int_key_to_string(xs)` — [`L94`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L94) — Convert all integer keys in a (possibly nested) dictionary to strings.
- `is_flatten(tree: dict)` — [`L458`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L458) — Checks if a dictionary represents a flattened tree.
- `is_iterable(obj)` — [`L165`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L165) — Check whether ``obj`` is an iterable (excluding strings).
- `iter_module_search(model: nn.Module, instance: type[T] | None = None)` — [`L779`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L779) — Iterates through a model and yields paths and modules of a specific type.
- `merge_model_and_tree(model: M, tree: dict, *, silence: bool = False)` — [`L582`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L582) — Attaches a parameter tree to an nnx model.
- `merge_state_and_tree(tree: dict, state: nnx.State, *, silence: bool = False)` — [`L533`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L533) — Attaches a parameter tree to an nnx state.
- `named_tree_map(f: tp.Callable[[str, tp.Any, tp.Any], tp.Any], tree: PyTree, *rest: tp.Any, is_leaf: tp.Callable[[tp.Any], bool] | None = None, sep: str | None = None)` — [`L693`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L693) — An extended version of `jax.tree_util.tree_map`.
- `nnx_init(module: type[M], _add_rngs: bool = True, _rng_key: str = "rngs", _seed: int = 0, _lazy: bool = True, **kwargs)` — [`L247`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L247) — Initializes an nnx module with lazy initialization support.
- `recreate_meta_values(values: dict[str, tp.Any], seed: int | None = None)` — [`L474`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L474) — Enhanced meta value recreation with better state management.
- `recursive_merge(full_tree, updates)` — [`L745`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L745) — Recursively merge two PyTrees where updates may have fewer parameters.
- `redefine_state(state: dict, missings: dict[str, nnx.VariableState])` — [`L418`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L418) — Redefines missing attributes in a state dictionary.
- `refine_graphs(*graphs: dict)` — [`L508`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L508) — Refines and merges multiple graph representations into a single nnx.State.
- `set_module_from_path(model: nn.Module, path: ModulePath, new_value: tp.Any)` — [`L828`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L828) — Sets a module at a specific path in the model.
- `specs_to_name_sharding(tree: dict, mesh: Mesh | None = None)` — [`L611`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L611) — Converts a dictionary of specifications to a dictionary of NamedSharding objects.
- `string_key_to_int(xs)` — [`L115`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L115) — Convert digit-only string keys in a dictionary back to integers.
- `tree_apply(fns: FnDict, tree: TreeDict)` — [`L628`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L628) — Apply a dictionary of functions to a corresponding PyTree.
- `tree_path_to_string(path: Path, sep: str | None = None)` — [`L642`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L642) — Convert a JAX tree path to a string representation.
- `unflatten_dict(xs, sep=None)` — [`L232`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L232) — Reconstruct a nested dictionary from a flattened one.
- `validate_state(state: dict[str, tp.Any], init_state: dict[str, tp.Any])` — [`L358`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L358) — Validate a state dictionary against a reference init state.

## Module values
- `FnDict` — [`L35`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L35)
- `M` — [`L91`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L91)
- `ModulePath` — [`L32`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L32)
- `Path` — [`L37`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L37)
- `PyTree` — [`L34`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L34)
- `T` — [`L31`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L31)
- `TreeDict` — [`L36`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L36)
- `empty_node` — [`L90`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L90)
- `logger` — [`L40`](../../../../../../raw/code/EasyDeL/easydel/utils/traversals.py#L40)

