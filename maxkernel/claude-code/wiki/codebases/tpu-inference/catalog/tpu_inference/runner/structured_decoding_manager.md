---
title: 'Module: tpu_inference/runner/structured_decoding_manager.py'
type: catalog
provenance: extracted
module: tpu_inference/runner/structured_decoding_manager.py
status: fresh
symbol_base: scip-python python tpu-inference 0.0.0 `tpu_inference.runner.structured_decoding_manager`/StructuredDecodingManager#
symbols:
  StructuredDecodingManager.prepare_structured_decoding_input: prepare_structured_decoding_input().
  StructuredDecodingManager.runner: runner.
  StructuredDecodingManager._apply_grammar_bitmask_kernel: _apply_grammar_bitmask_kernel().
  StructuredDecodingManager.structured_decode_fn: structured_decode_fn().
  StructuredDecodingManager: ''
  StructuredDecodingManager.__init__: __init__().
---
# Module: [`tpu_inference/runner/structured_decoding_manager.py`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py)

## Classes
### `StructuredDecodingManager`
- def: [`tpu_inference/runner/structured_decoding_manager.py:28`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py#L28)
- signature: `class StructuredDecodingManager:`
- members:
  - `prepare_structured_decoding_input(self, logits: jax.Array, grammar_output: GrammarOutput)` — [`L65`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py#L65)
  - `structured_decode_fn(self, require_struct_decoding: jax.Array, grammar_bitmask: jax.Array, logits: jax.Array, arange: jax.Array)` — [`L34`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py#L34)
  - `runner` — [`L31`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py#L31)
- protocol/private: `__init__`[`L30`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py#L30), `_apply_grammar_bitmask_kernel`[`L44`](../../../../../../raw/code/tpu-inference/tpu_inference/runner/structured_decoding_manager.py#L44)
- uses (calls/refs, reference-scoped): `mesh`, `input_batch`, [`req_id_to_index`](input_batch.md#InputBatch.req_id_to_index), `vocab_size`, `TPUModelRunner`, `grammar_bitmask_cpu`, `require_structured_out_cpu`, `structured_decode_arange`
- used by: `sample_tokens`, [`_precompile_structured_decoding`](compilation_manager.md#CompilationManager._precompile_structured_decoding), `structured_decoding_manager`

