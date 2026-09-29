---
title: 'Module: src/maxdiffusion/models/ltx_video/utils/torch_compat.py'
type: catalog
provenance: extracted
module: src/maxdiffusion/models/ltx_video/utils/torch_compat.py
status: fresh
symbol_base: scip-python python maxdiffusion 0.0.0 `src.maxdiffusion.models.ltx_video.utils.torch_compat`/
symbols:
  torch_statedict_to_jax: torch_statedict_to_jax().
  jax_statedict_to_torch: jax_statedict_to_torch().
  StateDict: StateDict.
  ConvertAction.__call__: ConvertAction#__call__().
  ConvertAction.apply_group_by: ConvertAction#apply_group_by().
  ConvertAction.rename: ConvertAction#rename.
  ConvertAction: ConvertAction#
  ConvertAction.apply_transpose: ConvertAction#apply_transpose().
  ConvertAction.apply_rename: ConvertAction#apply_rename().
  ConvertAction.apply_split_by: ConvertAction#apply_split_by().
  should_be_scan_repeatable: should_be_scan_repeatable().
  torch_statedict_to_jax.nested_insert: torch_statedict_to_jax().nested_insert().
  AnyTensor: AnyTensor.
  JaxParams: JaxParams.
  unbox_logically_partioned: unbox_logically_partioned().
  is_kernel_2d: is_kernel_2d().
  is_affine_scale_param: is_affine_scale_param().
  extract_scan_keywords: extract_scan_keywords().
  ScanRepeatableCarryBlock: ScanRepeatableCarryBlock.
  ConvertAction.transpose: ConvertAction#transpose.
  is_scan_repeatable: is_scan_repeatable().
  ConvertAction.split_by: ConvertAction#split_by.
  is_stack_or_tensor: is_stack_or_tensor().
  convert_tensor_stack_to_tensor: convert_tensor_stack_to_tensor().
  ConvertAction.group_by: ConvertAction#group_by.
  ConvertAction.jax_groups: ConvertAction#jax_groups.
  torch_tensor_to_jax_array: torch_tensor_to_jax_array().
  is_scale_shift_table: is_scale_shift_table().
---
# Module: [`src/maxdiffusion/models/ltx_video/utils/torch_compat.py`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py)

## Classes
### `ConvertAction`
- def: [`src/maxdiffusion/models/ltx_video/utils/torch_compat.py:78`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L78)
- doc: Defines a set of actions to be done on a given parameter.
- signature: `class ConvertAction:`
- members:
  - `__call__(self, mini_statedict: StateDict, new_params: List, full_statedict: StateDict, delim: str)` — [`L244`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L244) — Given a state dict, applies the transformations defined in the ConvertAction.
  - `apply_group_by(self, mini_statedict: StateDict, new_params: List, full_statedict: StateDict, delim: str)` — [`L207`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L207) — Applies the group_by action if defined
  - `apply_rename(self, mini_statedict: StateDict, delim: str)` — [`L143`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L143) — Applies the rename action if defined
  - `apply_split_by(self, mini_statedict: StateDict, new_params: List, delim: str)` — [`L176`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L176) — Applies the split_by action if defined
  - `apply_transpose(self, mini_statedict: StateDict)` — [`L128`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L128) — Applies the transpose action if defined
  - `group_by` — [`L107`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L107) — ---
  - `jax_groups` — [`L120`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L120) — ---
  - `rename` — [`L92`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L92) — ---
  - `split_by` — [`L100`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L100) — ---
  - `transpose` — [`L86`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L86) — ---
- uses (calls/refs, reference-scoped): [`StateDict`](torch_compat.md#StateDict), [`extract_scan_keywords`](torch_compat.md#extract_scan_keywords)
- used by: [`torch_statedict_to_jax`](torch_compat.md#torch_statedict_to_jax), [`jax_statedict_to_torch`](torch_compat.md#jax_statedict_to_torch)

## Functions
- `convert_tensor_stack_to_tensor(param: Union[List[torch.Tensor], torch.Tensor])` — [`L63`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L63) — Converts a list of torch tensors to a single torch tensor.
- `extract_scan_keywords(param_name: str, jax_flattened_keys: List[str], delim: str)` — [`L332`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L332) — Extracts the keywords from the scan repeatable carry block parameter (if exists)
- `is_affine_scale_param(param_name: str, parameter: AnyTensor, jax_flattened_keys: List[str])` — [`L313`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L313) — Checks if the parameter is an affine scale parameter.
- `is_kernel_2d(param_name: str, param: AnyTensor)` — [`L270`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L270) — Checks if the parameter is a 2D kernel (weight) or not.
- `is_scale_shift_table(param_name: str, _)` — [`L299`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L299) — Checks if the parameter is a scale shift table parameter.
- `is_scan_repeatable(param_name: str, _)` — [`L285`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L285) — Checks if the parameter is a scan repeatable carry block parameter.
- `is_stack_or_tensor(param: Any)` — [`L54`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L54) — Returns True if param is of type tensor or list/tuple of tensors (stack of tensors)
- `jax_statedict_to_torch(jax_params: JaxParams, rulebook: Optional[Dict[Callable[[str, AnyTensor], bool], ConvertAction]] = None)` — [`L388`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L388) — Converts a JAX state dict to a torch state dict.
- `nested_insert(param_name: str, param: torch.Tensor, nested_dict: Dict)` — [`L497`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L497) — Inserts a parameter into a nested dictionary. (to fit Jax format)
- `should_be_scan_repeatable(param_name: str, param: AnyTensor, jax_flattened_keys: List[str], delim: str)` — [`L369`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L369) — Checks if the parameter should be a scan repeatable carry block or not.
- `torch_statedict_to_jax(jax_params: JaxParams, torch_params: Dict[str, torch.Tensor])` — [`L443`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L443) — Converts a torch state dict to a JAX state dict.
- `torch_tensor_to_jax_array(data: torch.Tensor)` — [`L46`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L46)
- `unbox_logically_partioned(statedict: JaxParams)` — [`L38`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L38)

## Module values
- `AnyTensor` — [`L30`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L30)
- `JaxParams` — [`L35`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L35)
- `ScanRepeatableCarryBlock` — [`L33`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L33)
- `StateDict` — [`L31`](../../../../../../../../../raw/code/maxdiffusion/src/maxdiffusion/models/ltx_video/utils/torch_compat.py#L31)

