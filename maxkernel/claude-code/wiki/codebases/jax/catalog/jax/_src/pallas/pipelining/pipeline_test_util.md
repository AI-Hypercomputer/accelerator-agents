---
title: 'Module: jax/_src/pallas/pipelining/pipeline_test_util.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/pipelining/pipeline_test_util.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.pipelining.pipeline_test_util`/
symbols:
  print_stage: print_stage().
  compare_lists: compare_lists().
  AnyOrder: AnyOrder#
  AnyOrder.elements: AnyOrder#elements.
---
# Module: [`jax/_src/pallas/pipelining/pipeline_test_util.py`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/pipeline_test_util.py)

## Classes
### `AnyOrder`
- def: [`jax/_src/pallas/pipelining/pipeline_test_util.py:36`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/pipeline_test_util.py#L36)
- doc: A helper class to mark the order of elements as unimportant.
- signature: `class AnyOrder:`
- members:
  - `elements` — [`L38`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/pipeline_test_util.py#L38)
- used by: [`compare_lists`](pipeline_test_util.md#compare_lists)

## Functions
- `compare_lists(result, expected)` — [`L41`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/pipeline_test_util.py#L41) — Returns if two lists are equal while respecting ``AnyOrder`` elements.
- `print_stage(ctx: schedulers.PipelineContext, stage: internal.PipelineStage, *args)` — [`L26`](../../../../../../../../raw/code/jax/jax/_src/pallas/pipelining/pipeline_test_util.py#L26) — Evaluation function that prints the stage name and iteration number.

