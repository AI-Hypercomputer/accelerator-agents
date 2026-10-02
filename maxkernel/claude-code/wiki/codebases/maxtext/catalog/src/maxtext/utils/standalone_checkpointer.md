---
title: 'Module: src/maxtext/utils/standalone_checkpointer.py'
type: catalog
provenance: extracted
module: src/maxtext/utils/standalone_checkpointer.py
status: fresh
symbol_base: scip-python python maxtext 0.0.0 `src.maxtext.utils.standalone_checkpointer`/
symbols:
  checkpoint_loop: checkpoint_loop().
  checkpoint_loop.init_state_fn: checkpoint_loop().init_state_fn().
  main: main().
  add_entropy_to_checkpoint: add_entropy_to_checkpoint().
  Transformer: Transformer.
---
# Module: [`src/maxtext/utils/standalone_checkpointer.py`](../../../../../../../raw/code/maxtext/src/maxtext/utils/standalone_checkpointer.py)

## Functions
- `add_entropy_to_checkpoint(state)` — [`L120`](../../../../../../../raw/code/maxtext/src/maxtext/utils/standalone_checkpointer.py#L120) — Replace adam mu/nu with cos/sin of params.
- `checkpoint_loop(config, state=None)` — [`L47`](../../../../../../../raw/code/maxtext/src/maxtext/utils/standalone_checkpointer.py#L47) — Save/restore exerciser.
- `init_state_fn()` — [`L62`](../../../../../../../raw/code/maxtext/src/maxtext/utils/standalone_checkpointer.py#L62)
- `main(argv: Sequence[str])` — [`L155`](../../../../../../../raw/code/maxtext/src/maxtext/utils/standalone_checkpointer.py#L155)

## Module values
- `Transformer` — [`L44`](../../../../../../../raw/code/maxtext/src/maxtext/utils/standalone_checkpointer.py#L44)

