---
title: 'Module: jax/_src/numpy/reductions.py'
type: catalog
provenance: extracted
module: jax/_src/numpy/reductions.py
status: fresh
symbol_base: scip-python python jax 0.0.0 `jax._src.numpy.reductions`/
symbols:
  _quantile: _quantile().
  _reduction: _reduction().
  _var: _var().
  Axis: Axis.
  _logsumexp: _logsumexp().
  _nanvar: _nanvar().
  _cumulative_reduction: _cumulative_reduction().
  nanmean: nanmean().
  _mean: _mean().
  _average: _average().
  sum: sum().
  export: export.
  cumulative_prod: cumulative_prod().
  _nan_reduction: _nan_reduction().
  percentile: percentile().
  cumulative_sum: cumulative_sum().
  _reduce_bitwise_and: _reduce_bitwise_and().
  nanstd: nanstd().
  cumsum: cumsum().
  _reduce_sum: _reduce_sum().
  _std: _std().
  nanvar: nanvar().
  _ensure_optional_axes: _ensure_optional_axes().
  mean: mean().
  all: all().
  count_nonzero: count_nonzero().
  max: max().
  any: any().
  _count: _count().
  _reduce_prod: _reduce_prod().
  _reduce_bitwise_or: _reduce_bitwise_or().
  _reduce_bitwise_xor: _reduce_bitwise_xor().
  _reduce_logical_and: _reduce_logical_and().
  _reduce_logical_or: _reduce_logical_or().
  _reduce_logical_xor: _reduce_logical_xor().
  nansum: nansum().
  nanprod: nanprod().
  _reduce_max: _reduce_max().
  _reduce_min: _reduce_min().
  _logsumexp2: _logsumexp2().
  var: var().
  prod: prod().
  check_where: check_where().
  std: std().
  ptp: ptp().
  cumprod: cumprod().
  _reduce_all: _reduce_all().
  _reduce_any: _reduce_any().
  _moveaxis: _moveaxis().
  _cast_to_bool: _cast_to_bool().
  nanmin: nanmin().
  nanmax: nanmax().
  nancumsum: nancumsum().
  nancumprod: nancumprod().
  nanpercentile: nanpercentile().
  _reduction_init_val: _reduction_init_val().
  min: min().
  quantile: quantile().
  nanquantile: nanquantile().
  median: median().
  _ptp: _ptp().
  _cumsum_with_promotion: _cumsum_with_promotion().
  amin: amin().
  amax: amax().
  nanmedian: nanmedian().
  _var_promote_types: _var_promote_types().
  _promote_integer_dtype: _promote_integer_dtype().
  _reduction_dims: _reduction_dims().
  average: average().
  _upcast_f16: _upcast_f16().
  _require_integer: _require_integer().
  _axis_size: _axis_size().
  _cast_to_numeric: _cast_to_numeric().
  _canonicalize_axis_allow_named: _canonicalize_axis_allow_named().
  percentile.internal_quantile: percentile().internal_quantile().
  _isscalar: _isscalar().
  ReductionOp: ReductionOp.
  _all: _all.
  _ensure_optional_axes.force: _ensure_optional_axes().force().
---
# Module: [`jax/_src/numpy/reductions.py`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py)

## Functions
- `_average(a: ArrayLike, axis: Axis = None, weights: ArrayLike | None = None, returned: bool = False, keepdims: bool = False)` — [`L999`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L999)
- `_axis_size(a: ArrayLike, axis: int | Sequence[int])` — [`L796`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L796)
- `_canonicalize_axis_allow_named(x, rank)` — [`L167`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L167)
- `_cast_to_bool(operand: Array)` — [`L205`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L205)
- `_cast_to_numeric(operand: Array)` — [`L210`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L210)
- `_count(a: ArrayLike, axis: Axis, keepdims: bool, where: ArrayLike | None, dtype: DTypeLike)` — [`L875`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L875)
- `_cumsum_with_promotion(a: ArrayLike, axis: int | None = None, dtype: DTypeLike | None = None, out: None = None)` — [`L2257`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2257) — Utility function to compute cumsum with integer promotion.
- `_cumulative_reduction(name: str, reduction: Callable[..., Array], a: ArrayLike, axis: int | None, dtype: DTypeLike | None, out: None = None, fill_nan: bool = False, fill_value: ArrayLike = 0, promote_integers: bool = False)` — [`L2033`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2033) — Helper function for implementing cumulative reductions.
- `_ensure_optional_axes(x: Axis)` — [`L218`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L218)
- `_isscalar(element: Any)` — [`L53`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L53)
- `_logsumexp(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L740`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L740) — Compute log(sum(exp(a))) while avoiding precision loss.
- `_logsumexp2(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L763`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L763) — Compute log2(sum(2 ** a)) via logsumexp.
- `_mean(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, *, upcast_f16_for_computation: bool = True, where: ArrayLike | None = None)` — [`L894`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L894)
- `_moveaxis(a: ArrayLike, source: int, destination: int)` — [`L59`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L59)
- `_nan_reduction(a: ArrayLike, name: str, jnp_reduction: Callable[..., Array], init_val: ArrayLike, nan_if_all_nan: bool, axis: Axis = None, keepdims: bool = False, where: ArrayLike | None = None, **kwargs)` — [`L1385`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1385)
- `_nanvar(a: Array, *, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, ddof: int = 0, keepdims: bool = False, where: ArrayLike | None = None, a_mean: ArrayLike | None = None)` — [`L1922`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1922)
- `_promote_integer_dtype(dtype: DType)` — [`L73`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L73)
- `_ptp(a: Array, axis: Axis = None, out: None = None, keepdims: bool = False)` — [`L1332`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1332)
- `_quantile(a: Array, q: Array, axis: int | tuple[int, ...] | None, method: str, keepdims: bool, squash_nans: bool, weights: Array | None = None)` — [`L2515`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2515)
- `_reduce_all(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, *, where: ArrayLike | None = None)` — [`L570`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L570)
- `_reduce_any(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, *, where: ArrayLike | None = None)` — [`L627`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L627)
- `_reduce_bitwise_and(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L685`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L685)
- `_reduce_bitwise_or(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L696`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L696)
- `_reduce_bitwise_xor(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L705`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L705)
- `_reduce_logical_and(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L714`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L714)
- `_reduce_logical_or(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L723`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L723)
- `_reduce_logical_xor(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L732`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L732)
- `_reduce_max(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L405`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L405)
- `_reduce_min(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L488`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L488)
- `_reduce_prod(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None, promote_integers: bool = True)` — [`L318`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L318)
- `_reduce_sum(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None, promote_integers: bool = True)` — [`L231`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L231)
- `_reduction(a: ArrayLike, name: str, op: ReductionOp, init_val: ArrayLike, *, has_identity: bool = True, preproc: Callable[[Array], Array] | None = None, bool_op: ReductionOp | None = None, upcast_f16_for_computation: bool = False, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where_: ArrayLike | None = None, parallel_reduce: Callable[..., Array] | None = None, promote_integers: bool = False)` — [`L100`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L100)
- `_reduction_dims(a: ArrayLike, axis: Axis)` — [`L170`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L170)
- `_reduction_init_val(a: Array, init_val: Any)` — [`L187`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L187)
- `_require_integer(arr: Array)` — [`L213`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L213)
- `_std(a: Array, *, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, correction: int | float = 0, keepdims: bool = False, where: ArrayLike | None = None, mean: ArrayLike | None = None)` — [`L1275`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1275)
- `_upcast_f16(dtype: DTypeLike)` — [`L68`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L68)
- `_var(a: Array, *, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, correction: int | float = 0, keepdims: bool = False, where: ArrayLike | None = None, a_mean: ArrayLike | None = None)` — [`L1127`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1127)
- `_var_promote_types(a_dtype: DTypeLike, dtype: DTypeLike | None)` — [`L1166`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1166)
- `all(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, *, where: ArrayLike | None = None)` — [`L577`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L577) — Test whether all array elements along a given axis evaluate to True.
- `amax(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L789`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L789) — Alias of :func:`jax.numpy.max`.
- `amin(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L781`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L781) — Alias of :func:`jax.numpy.min`.
- `any(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, *, where: ArrayLike | None = None)` — [`L634`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L634) — Test whether any of the array elements along a given axis evaluate to True.
- `average(a: ArrayLike, axis: Axis = None, weights: ArrayLike | None = None, returned: Literal[False] = False, keepdims: bool = False)` — [`L927`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L927)
- `check_where(name: str, where: ArrayLike | None)` — [`L88`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L88)
- `count_nonzero(a: ArrayLike, axis: Axis = None, keepdims: bool = False)` — [`L1343`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1343) — Return the number of nonzero elements along a given axis.
- `cumprod(a: ArrayLike, axis: int | None = None, dtype: DTypeLike | None = None, out: None = None)` — [`L2123`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2123) — Cumulative product of elements along an axis.
- `cumsum(a: ArrayLike, axis: int | None = None, dtype: DTypeLike | None = None, out: None = None)` — [`L2086`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2086) — Cumulative sum of elements along an axis.
- `cumulative_prod(x: ArrayLike,, *, axis: int | None = None, dtype: DTypeLike | None = None, include_initial: bool = False)` — [`L2327`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2327) — Cumulative product along the axis of an array.
- `cumulative_sum(x: ArrayLike,, *, axis: int | None = None, dtype: DTypeLike | None = None, include_initial: bool = False)` — [`L2265`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2265) — Cumulative sum along the axis of an array.
- `force(x)` — [`L219`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L219)
- `internal_quantile(x, y, w)` — [`L2775`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2775)
- `max(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L414`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L414) — Return the maximum of the array elements along a given axis.
- `mean(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, *, where: ArrayLike | None = None)` — [`L809`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L809) — Return the mean of array elements along a given axis.
- `median(a: ArrayLike, axis: int | tuple[int, ...] | None = None, out: None = None, overwrite_input: bool = False, keepdims: bool = False)` — [`L2853`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2853) — Return the median of array elements along a given axis.
- `min(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L497`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L497) — Return the minimum of array elements along a given axis.
- `nancumprod(a: ArrayLike, axis: int | None = None, dtype: DTypeLike | None = None, out: None = None)` — [`L2209`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2209) — Cumulative product of elements along an axis, ignoring NaN values.
- `nancumsum(a: ArrayLike, axis: int | None = None, dtype: DTypeLike | None = None, out: None = None)` — [`L2159`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2159) — Cumulative sum of elements along an axis, ignoring NaN values.
- `nanmax(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L1488`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1488) — Return the maximum of the array elements along a given axis, ignoring NaNs.
- `nanmean(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, where: ArrayLike | None = None)` — [`L1742`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1742) — Return the mean of the array elements along a given axis, ignoring NaNs.
- `nanmedian(a: ArrayLike, axis: int | tuple[int, ...] | None = None, out: None = None, overwrite_input: bool = False, keepdims: bool = False)` — [`L2905`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2905) — Return the median of array elements along a given axis, ignoring NaNs.
- `nanmin(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L1405`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1405) — Return the minimum of the array elements along a given axis, ignoring NaNs.
- `nanpercentile(a: ArrayLike, q: ArrayLike, axis: int | tuple[int, ...] | None = None, out: None = None, overwrite_input: bool = False, method: str = "linear", keepdims: bool = False, *, weights: ArrayLike | None = None)` — [`L2789`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2789) — Compute the percentile of the data along the specified axis, ignoring NaN values.
- `nanprod(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L1657`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1657) — Return the product of the array elements along a given axis, ignoring NaNs.
- `nanquantile(a: ArrayLike, q: ArrayLike, axis: int | tuple[int, ...] | None = None, out: None = None, overwrite_input: bool = False, method: str = "linear", keepdims: bool = False, *, weights: ArrayLike | None = None)` — [`L2453`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2453) — Compute the quantile of the data along the specified axis, ignoring NaNs.
- `nanstd(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, ddof: int = 0, keepdims: bool = False, where: ArrayLike | None = None, mean: ArrayLike | None = None)` — [`L1951`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1951) — Compute the standard deviation along a given axis, ignoring NaNs.
- `nansum(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None)` — [`L1571`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1571) — Return the sum of the array elements along a given axis, ignoring NaNs.
- `nanvar(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, ddof: int = 0, keepdims: bool = False, where: ArrayLike | None = None, mean: ArrayLike | None = None)` — [`L1834`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1834) — Compute the variance of array elements along a given axis, ignoring NaNs.
- `percentile(a: ArrayLike, q: ArrayLike, axis: int | tuple[int, ...] | None = None, out: None = None, overwrite_input: bool = False, method: str = "linear", keepdims: bool = False, *, weights: ArrayLike | None = None, out_sharding: NamedSharding | P | None = None)` — [`L2718`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2718) — Compute the percentile of the data along the specified axis.
- `prod(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None, promote_integers: bool = True)` — [`L329`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L329) — Return product of the array elements over a given axis.
- `ptp(a: ArrayLike, axis: Axis = None, out: None = None, keepdims: bool = False)` — [`L1290`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1290) — Return the peak-to-peak range along a given axis.
- `quantile(a: ArrayLike, q: ArrayLike, axis: int | tuple[int, ...] | None = None, out: None = None, overwrite_input: bool = False, method: str = "linear", keepdims: bool = False, *, weights: ArrayLike | None = None)` — [`L2391`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L2391) — Compute the quantile of the data along the specified axis.
- `std(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, ddof: int = 0, keepdims: bool = False, *, where: ArrayLike | None = None, mean: ArrayLike | None = None, correction: int | float | None = None)` — [`L1188`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1188) — Compute the standard deviation along a given axis.
- `sum(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False, initial: ArrayLike | None = None, where: ArrayLike | None = None, promote_integers: bool = True)` — [`L243`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L243) — Sum of the elements of the array over a given axis.
- `var(a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None, out: None = None, ddof: int = 0, keepdims: bool = False, *, where: ArrayLike | None = None, mean: ArrayLike | None = None, correction: int | float | None = None)` — [`L1036`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L1036) — Compute the variance along a given axis.

## Module values
- `Axis` — [`L51`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L51)
- `ReductionOp` — [`L98`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L98)
- `_all` — [`L48`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L48)
- `export` — [`L46`](../../../../../../../raw/code/jax/jax/_src/numpy/reductions.py#L46)

