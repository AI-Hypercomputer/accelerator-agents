---
title: 'Module: pallas_forge/kernels/_utils.py'
type: catalog
provenance: extracted
module: pallas_forge/kernels/_utils.py
status: fresh
symbol_base: scip-python python pallas-forge 0.0.0 `pallas_forge.kernels._utils`/
symbols:
  pad_to_multiple: pad_to_multiple().
  unpad: unpad().
  next_multiple: next_multiple().
  ceildiv: ceildiv().
  check_alignment: check_alignment().
---
# Module: [`pallas_forge/kernels/_utils.py`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/_utils.py)

## Functions
- `ceildiv(a: int, b: int)` — [`L43`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/_utils.py#L43) — Ceiling division: ceildiv(7, 3) = 3.
- `check_alignment(shape: tuple[int, ...], alignment: int = 128)` — [`L13`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/_utils.py#L13) — Check if all dimensions of a shape meet the alignment requirement.
- `next_multiple(n: int, multiple: int)` — [`L48`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/_utils.py#L48) — Round n up to the next multiple. E.g., next_multiple(100, 128) = 128.
- `pad_to_multiple(x: jax.Array, multiple: int, axis: int)` — [`L22`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/_utils.py#L22) — Pad a JAX array along `axis` so its size is a multiple of `multiple`.
- `unpad(x: jax.Array, target_shape: tuple[int, ...])` — [`L37`](../../../../../../raw/code/pallas-forge/pallas_forge/kernels/_utils.py#L37) — Remove padding by slicing back to the target shape.

