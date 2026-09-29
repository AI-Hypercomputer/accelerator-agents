---
title: 'Module: jax/_src/numpy/vectorize.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/vectorize.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.vectorize`/
symbols:
  vectorize.wrapped: vectorize().wrapped().
  _parse_input_dimensions: _parse_input_dimensions().
  vectorize: vectorize().
  _parse_gufunc_signature: _parse_gufunc_signature().
  _check_output_dims.wrapped: _check_output_dims().wrapped().
  CoreDims: CoreDims.
  _update_dim_sizes: _update_dim_sizes().
  _check_output_dims: _check_output_dims().
  _ARGUMENT: _ARGUMENT.
  _apply_excluded: _apply_excluded().
  export: export.
  _CORE_DIMENSION_LIST: _CORE_DIMENSION_LIST.
  _ARGUMENT_LIST: _ARGUMENT_LIST.
  _SIGNATURE: _SIGNATURE.
  _DIMENSION_NAME: _DIMENSION_NAME.
  NDArray: NDArray.
  _apply_excluded.new_func: _apply_excluded().new_func().
---
# Module: [`jax/_src/numpy/vectorize.py`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py)

## Functions
- `_apply_excluded(func: Callable[..., Any], excluded: Collection[int | str], args: Sequence[Any], kwargs: dict[str, Any])` — [`L166`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L166) — Partially apply positional arguments in `excluded` to a function.
- `_check_output_dims(func: Callable, dim_sizes: dict[str, int], expected_output_core_dims: list[CoreDims], error_context: str = "")` — [`L136`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L136) — Check that output core dimensions match the signature.
- `_parse_gufunc_signature(signature: str)` — [`L45`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L45) — Parse string signatures for a generalized universal function.
- `_parse_input_dimensions(args: tuple[NDArray, ...], input_core_dims: list[CoreDims], error_context: str = "")` — [`L104`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L104) — Parse broadcast and core dimensions for vectorize with a signature.
- `_update_dim_sizes(dim_sizes: dict[str, int], shape: tuple[int, ...], core_dims: CoreDims, error_context: str = "", *, is_input: bool)` — [`L66`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L66) — Incrementally check and update core dimension sizes for a single argument.
- `new_func(*args, **kwargs)` — [`L180`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L180)
- `vectorize(pyfunc, *, excluded=frozenset(), signature=None)` — [`L190`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L190) — Define a vectorized function with broadcasting.
- `wrapped(*args)` — [`L143`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L143)
- `wrapped(*args, **kwargs)` — [`L272`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L272)

## Module values
- `CoreDims` — [`L41`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L41)
- `NDArray` — [`L42`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L42)
- `_ARGUMENT` — [`L36`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L36)
- `_ARGUMENT_LIST` — [`L37`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L37)
- `_CORE_DIMENSION_LIST` — [`L35`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L35)
- `_DIMENSION_NAME` — [`L34`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L34)
- `_SIGNATURE` — [`L38`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L38)
- `export` — [`L31`](../../../../../../../raw/code/jax/jax/_src/numpy/vectorize.py#L31)

