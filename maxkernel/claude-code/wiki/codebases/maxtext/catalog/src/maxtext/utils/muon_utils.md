---
title: 'Module: src/maxtext/utils/muon_utils.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/muon_utils.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.muon_utils`/
symbols:
  get_muon_weight_dimension_numbers: get_muon_weight_dimension_numbers().
  scan_layers_arg: scan_layers_arg.
  get_transform_tree: get_transform_tree().
  transform_logic: transform_logic().
  get_muon_weight_dimension_numbers.apply_transform_nnx: get_muon_weight_dimension_numbers().apply_transform_nnx().
  _print_structure_debug: _print_structure_debug().
  get_model_mdn: get_model_mdn().
  _is_path_contain_any: _is_path_contain_any().
  _print_structure_debug.get_leaf_info: _print_structure_debug().get_leaf_info().
  model_name_arg: model_name_arg.
---
# Module: [`src/maxtext/utils/muon_utils.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py)

## Functions
- `_is_path_contain_any(tuples, path)` — [`L42`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L42) — Checks if any element in 'tuples' is present in 'path'.
- `_print_structure_debug(abstract_param, muon_weight_dimension_numbers)` — [`L135`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L135) — Prints the model structure and the resulting Muon config.
- `apply_transform_nnx(path: Tuple[jax.tree_util.KeyEntry, ...], leaf)` — [`L112`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L112)
- `get_leaf_info(leaf)` — [`L138`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L138)
- `get_model_mdn(model_name, scan_layers=True, verbose=False, pure_nnx=False)` — [`L160`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L160) — Initializes a model and retrieves its Muon dimension numbers.
- `get_muon_weight_dimension_numbers(model, config, verbose=False)` — [`L106`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L106) — Extract muon dimension number from model structure.
- `get_transform_tree(tree, path=())` — [`L98`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L98) — Extraction utility via recursion.
- `transform_logic(path: Tuple[str, ...])` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L47) — Determines Muon dimension numbers based on the parameter's hierarchical path.

## Module values
- `model_name_arg` — [`L206`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L206)
- `scan_layers_arg` — [`L207`](../../../../../../../raw/code/maxtext/src/maxtext/utils/muon_utils.py#L207)

