---
title: 'Module: simply/utils/pytree.py'
type: catalog
provenance: extracted
module: simply/utils/pytree.py
status: fresh
symbol_base: scip-python python simply 0.0.0 `simply.utils.pytree`/
symbols:
  load: load().
  dump: dump().
  traverse_tree_with_path: traverse_tree_with_path().
  PyTree: PyTree.
  set_tree_value: set_tree_value().
  tree_is_sequence: tree_is_sequence().
  tree_is_mapping: tree_is_mapping().
  tree_value: tree_value().
  concatenate_pytrees: concatenate_pytrees().
  trim_none: trim_none().
  load_pytree_from: load_pytree_from().
  check_trees_match_mapping_keys: check_trees_match_mapping_keys().
  save_pytree_to: save_pytree_to().
  check_trees_match_sequence_length: check_trees_match_sequence_length().
  to_flat_dict: to_flat_dict().
  dump_dataclasses: dump_dataclasses().
  load_dataclasses: load_dataclasses().
  construct_tree_with_path_value: construct_tree_with_path_value().
  tree_leaves_with_tag: tree_leaves_with_tag().
  convert_string_path_to_key_path: convert_string_path_to_key_path().
  tree_leaves_with_tag._get_raw: tree_leaves_with_tag()._get_raw().
---
# Module: [`simply/utils/pytree.py`](../../../../../../raw/code/simply/simply/utils/pytree.py)

## Functions
- `_get_raw(path)` — [`L221`](../../../../../../raw/code/simply/simply/utils/pytree.py#L221)
- `check_trees_match_mapping_keys(trees: Sequence[PyTree], keys: Set[str])` — [`L141`](../../../../../../raw/code/simply/simply/utils/pytree.py#L141) — Raises error if trees are not consistent mapping with keys.
- `check_trees_match_sequence_length(trees: Sequence[PyTree], length: int)` — [`L155`](../../../../../../raw/code/simply/simply/utils/pytree.py#L155) — Raises error if trees are not consistent sequence with length.
- `concatenate_pytrees(trees: Sequence[PyTree])` — [`L368`](../../../../../../raw/code/simply/simply/utils/pytree.py#L368) — Concatenates multiple pytrees' underlining sequences into one. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `construct_tree_with_path_value(path: jax.tree_util.KeyPath, value: PyTree)` — [`L77`](../../../../../../raw/code/simply/simply/utils/pytree.py#L77) — Constructs a tree with path and value. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `convert_string_path_to_key_path(path: str)` — [`L34`](../../../../../../raw/code/simply/simply/utils/pytree.py#L34) — Converts a string path to a KeyPath.
- `dump(ptree: Any, only_dump_basic: bool = True)` — [`L314`](../../../../../../raw/code/simply/simply/utils/pytree.py#L314) — Dumps data in pytree into a json-like tree. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `dump_dataclasses(ptree: Any, only_dump_basic: bool = True)` — [`L363`](../../../../../../raw/code/simply/simply/utils/pytree.py#L363)
- `load(jtree: PyTree)` — [`L274`](../../../../../../raw/code/simply/simply/utils/pytree.py#L274) — Loads data objects (dataclasses and numpy arrays) in a json-like tree. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `load_dataclasses(jtree: PyTree)` — [`L358`](../../../../../../raw/code/simply/simply/utils/pytree.py#L358)
- `load_pytree_from(path: epath.PathLike)` — [`L473`](../../../../../../raw/code/simply/simply/utils/pytree.py#L473) — Loads a pytree from a file. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `save_pytree_to(tree: Any, path: epath.PathLike)` — [`L466`](../../../../../../raw/code/simply/simply/utils/pytree.py#L466) — Saves a pytree to a file. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `set_tree_value(tree: PyTree, path: jax.tree_util.KeyPath | str, value: PyTree)` — [`L94`](../../../../../../raw/code/simply/simply/utils/pytree.py#L94) — Sets value at path in tree. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `to_flat_dict(tree: PyTree, sep: str = '', is_leaf: Callable[[Any], bool] | None = None)` — [`L236`](../../../../../../raw/code/simply/simply/utils/pytree.py#L236) — Converts a tree into a flattened dictionary. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)
- `traverse_tree_with_path(fn: Callable[..., PyTree], *trees: PyTree, root_path: str = '')` — [`L168`](../../../../../../raw/code/simply/simply/utils/pytree.py#L168) — Traverses tree with path, with fn applied to each leaf node.
- `tree_is_mapping(tree: PyTree)` — [`L133`](../../../../../../raw/code/simply/simply/utils/pytree.py#L133)
- `tree_is_sequence(tree: PyTree)` — [`L137`](../../../../../../raw/code/simply/simply/utils/pytree.py#L137)
- `tree_leaves_with_tag(tree, tag='loss')` — [`L211`](../../../../../../raw/code/simply/simply/utils/pytree.py#L211) — Yields leaves and their paths from a pytree if the path contains a tag.
- `tree_value(tree: PyTree, path: jax.tree_util.KeyPath | str)` — [`L57`](../../../../../../raw/code/simply/simply/utils/pytree.py#L57) — Gets tree value by path. — documented in [simply-utils-checkpoint_lib](../../../concepts/simply-utils-checkpoint_lib.md)
- `trim_none(tree: PyTree)` — [`L432`](../../../../../../raw/code/simply/simply/utils/pytree.py#L432) — Trim pytree with None values. — documented in [simply-utils-pytree](../../../concepts/simply-utils-pytree.md)

## Module values
- `PyTree` — [`L31`](../../../../../../raw/code/simply/simply/utils/pytree.py#L31)

