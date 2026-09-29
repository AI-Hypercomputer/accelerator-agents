---
title: 'Module: jax/_src/pallas/helpers.py'
type: catalog
provenance: extracted
module: jax/_src/pallas/helpers.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.pallas.helpers`/
symbols:
  when: when().
  empty_ref_like: empty_ref_like().
  kernel: kernel().
  loop: loop().
  select_ref: select_ref().
  empty: empty.
  empty_like: empty_like().
  _T: _T.
  loop.decorator: loop().decorator().
  when._wrapped: when()._wrapped().
  with_scoped: with_scoped().
  with_scoped.decorator: with_scoped().decorator().
  with_scoped.decorator.inner: with_scoped().decorator().inner().
---
# Module: [`jax/_src/pallas/helpers.py`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py)

## Functions
- `_wrapped(f)` — [`L81`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L81)
- `decorator(body)` — [`L140`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L140)
- `decorator(f)` — [`L298`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L298)
- `empty_like(x: object)` — [`L39`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L39) — Create an empty PyTree of possibly uninitialized values.
- `empty_ref_like(x: object)` — [`L56`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L56) — Returns an empty array Ref with same shape/dtype/memory space as x.
- `inner(*args)` — [`L299`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L299)
- `kernel(body: Callable | Sequence[Callable] | api.NotSpecified = api.NotSpecified(), out_type: object = (), *, mesh: pl_core.Mesh | Sequence[pl_core.Mesh], scratch_types: pl_core.ScratchShapeTree = (), compiler_params: pl_core.CompilerParams | None = None, interpret: Any = False, cost_estimate: pl_core.CostEstimate | None = None, debug: bool = False, name: str | None = None, metadata: dict[str, str] | None = None)` — [`L156`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L156) — Entry point for creating a Pallas kernel.
- `loop(lower: jax_typing.ArrayLike, upper: jax_typing.ArrayLike, *, init_carry: None = ..., step: jax_typing.ArrayLike = ..., unroll: int | bool | None = ...)` — [`L94`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L94)
- `select_ref(idx: jax_typing.Array, *refs)` — [`L310`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L310) — Selects a ref from a list of refs based on the runtime value of a scalar index.
- `when(condition: bool | jax_typing.ArrayLike,)` — [`L68`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L68) — Calls the decorated function when the condition is met.
- `with_scoped(*types: Any, collective_axes: Hashable | tuple[Hashable, ...] = (), **kw_types: Any)` — [`L268`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L268) — Returns a function decorator that runs a function with provided allocations.

## Module values
- `_T` — [`L90`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L90)
- `empty` — [`L35`](../../../../../../../raw/code/jax/jax/_src/pallas/helpers.py#L35)

