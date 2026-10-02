---
title: 'Module: recurrentgemma/torch/test_utils.py'
type: catalog
provenance: extracted
module: recurrentgemma/torch/test_utils.py
status: fresh
symbol_base: scip-python python recurrentgemma 0.0.0 `recurrentgemma.torch.test_utils`/
symbols:
  numerically_compare_modules: numerically_compare_modules().
  compare_jax_to_torch: compare_jax_to_torch().
  generate_input: generate_input().
---
# Module: [`recurrentgemma/torch/test_utils.py`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/test_utils.py)

## Functions
- `compare_jax_to_torch(jax_outputs: Any, torch_outputs: Any, tols: dict[str, float] | None = None)` — [`L40`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/test_utils.py#L40) — Compares numerically Jax values to PyTorch values.
- `generate_input(rng: jax.Array, input_shape: tuple[int, ...] | list[int], dtype: str, vocab_size: int)` — [`L26`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/test_utils.py#L26)
- `numerically_compare_modules(jax_module, torch_module, uses_segment_pos: bool, has_cache: bool, input_shape: tuple[int, ...] | list[int], dtype: str, seed: int, num_unroll_steps: int = 2, vocab_size: int | None = None, tols: dict[str, float] | None = None)` — [`L59`](../../../../../../raw/code/recurrentgemma/recurrentgemma/torch/test_utils.py#L59) — Compares numerically Jax and PyTorch modules.

