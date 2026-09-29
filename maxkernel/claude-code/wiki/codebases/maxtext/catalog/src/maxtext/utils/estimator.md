---
title: 'Module: src/maxtext/utils/estimator.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/estimator.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.estimator`/
symbols:
  main: main().
  RematPolicy.next_policy: RematPolicy#next_policy().
  RematPolicy.previous_policy: RematPolicy#previous_policy().
  search_policy_only: search_policy_only().
  search: search().
  Action: Action#
  RematPolicy: RematPolicy#
  is_oom: is_oom().
  RematPolicy.to_dict: RematPolicy#to_dict().
  RematPolicy.tensors: RematPolicy#tensors.
  build_argv: build_argv().
  generate_remat_config: generate_remat_config().
  Action.REMAT: Action#REMAT.
  RematPolicy.__init__: RematPolicy#__init__().
  largest_batch_size: largest_batch_size().
  RematPolicy.tensor_order: RematPolicy#tensor_order.
  generate_priority_list: generate_priority_list().
  find_batch_size: find_batch_size().
  find_remat_policy_tensor_names: find_remat_policy_tensor_names().
  RematPolicy.__repr__: RematPolicy#__repr__().
  Action.OFFLOAD: Action#OFFLOAD.
  Action.DEVICE: Action#DEVICE.
  get_parameter_value: get_parameter_value().
  tensor_score: tensor_score().
  find_pdb_scalar: find_pdb_scalar().
  generate_pdb_config: generate_pdb_config().
---
# Module: [`src/maxtext/utils/estimator.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py)

## Classes
### `Action`  ·  implements/extends IntEnum
- def: [`src/maxtext/utils/estimator.py:47`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L47)
- signature: `class Action(IntEnum):`
- members:
  - `DEVICE` — [`L50`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L50)
  - `OFFLOAD` — [`L49`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L49)
  - `REMAT` — [`L48`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L48)
- used by: [`main`](estimator.md#main), [`next_policy`](estimator.md#RematPolicy.next_policy), [`previous_policy`](estimator.md#RematPolicy.previous_policy), [`search`](estimator.md#search), [`search_policy_only`](estimator.md#search_policy_only), [`to_dict`](estimator.md#RematPolicy.to_dict), [`__init__`](estimator.md#RematPolicy.__init__)

### `RematPolicy`
- def: [`src/maxtext/utils/estimator.py:53`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L53)
- doc: RematPolicy representing different remat policy combinations
- signature: `class RematPolicy:`
- members:
  - `next_policy(self)` — [`L66`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L66) — Moves from Remat -> Offload -> Device.
  - `previous_policy(self)` — [`L84`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L84) — Moves from Device -> Offload -> Remat.
  - `to_dict(self)` — [`L61`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L61) — Convert internal action to strings for MaxText
  - `tensor_order` — [`L58`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L58)
  - `tensors` — [`L57`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L57)
- protocol/private: `__init__`[`L56`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L56), `__repr__`[`L99`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L99)
- uses (calls/refs, reference-scoped): [`Action`](estimator.md#Action), [`REMAT`](estimator.md#Action.REMAT), [`DEVICE`](estimator.md#Action.DEVICE), [`OFFLOAD`](estimator.md#Action.OFFLOAD)
- used by: [`main`](estimator.md#main), [`search`](estimator.md#search), [`search_policy_only`](estimator.md#search_policy_only), [`is_oom`](estimator.md#is_oom), [`generate_remat_config`](estimator.md#generate_remat_config)

## Functions
- `build_argv(base_argv, remat_policy, pdb: float)` — [`L391`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L391) — Builds the argument vector for train_compile.
- `find_batch_size(base_argv)` — [`L443`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L443) — Parses the base arguments to find the ``per_device_batch_size``.
- `find_pdb_scalar(config)` — [`L170`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L170) — Calculates the scaling factor to normalize the Per-Device Batch (PDB) size.
- `find_remat_policy_tensor_names(base_argv)` — [`L461`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L461) — Finds tensors explicitly provided as flags in the command line.
- `generate_pdb_config(pdb: float)` — [`L386`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L386) — Generate batch size configs
- `generate_priority_list(config, provided_tensor_names)` — [`L103`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L103) — Generates a sorted list of tensors based on their scores.
- `generate_remat_config(policy)` — [`L377`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L377) — Generate remat-related configs from a RematPolicy or a dict.
- `get_parameter_value(config_tuple, prefix)` — [`L417`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L417) — Searches a tuple for an item starting with a specific prefix
- `is_oom(base_argv, policy: RematPolicy, pdb: float)` — [`L253`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L253) — Checks if the given policy and batch size cause an OOM error.
- `largest_batch_size(base_argv, policy, min_pdb=None, max_pdb=32, pdb_scalar=1)` — [`L197`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L197) — Finds the largest possible ``per_device_batch_size`` (pdb) that does not cause
- `main(argv_list: Sequence[str])` — [`L494`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L494) — Main entry point for the remat policy estimation script.
- `search(tensor_names, base_argv, min_pdb: float | None = None, max_pdb: float = 64, init_policy: RematPolicy = None, pdb_scalar: float = 1)` — [`L338`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L338) — Performs the core search algorithm to find the Pareto frontier points.
- `search_policy_only(tensor_names, base_argv, pdb, init_policy: RematPolicy = None)` — [`L296`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L296) — Finds the "lightest" remat policy that fits in memory for a *fixed* batch size.
- `tensor_score(tensor_name: str, config)` — [`L124`](../../../../../../../raw/code/maxtext/src/maxtext/utils/estimator.py#L124) — Calculates a score for a given tensor.

